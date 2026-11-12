"""Prepare the official Franka FR3v2 + hand model for PyBullet, without ROS.

pip install -e '.[sim,fr3]'
python tools/prepare_fr3.py

Downloads a pinned Apache-2.0 franka_description revision (or use --source for
an existing checkout). Keeps official kinematics, inertias and collision meshes;
converts visual COLLADA to OBJ for Bullet. Assets stay in an ignored cache.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from copy import deepcopy
from pathlib import Path

import trimesh
import xacro

REVISION = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"
REPOSITORY = "https://github.com/frankarobotics/franka_description"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("robots/common/", "robots/fr3v2/", "end_effectors/common/",
               "end_effectors/franka_hand/", "meshes/robots/fr3v2/",
               "meshes/robot_ee/franka_hand_white/")


def prepare(output: Path, source: Path | None = None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grasp-fr3-") as temporary:
        work = Path(temporary)
        package = work / "franka_description"
        if source is None:
            archive = work / "source.zip"
            print(f"Downloading official franka_description @ {REVISION[:12]}", flush=True)
            request = urllib.request.Request(
                f"https://codeload.github.com/frankarobotics/franka_description/zip/{REVISION}",
                headers={"User-Agent": "grasp-sampler-fr3-setup"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    relative = Path(*Path(entry.filename).parts[1:])
                    if not relative.parts or entry.is_dir() or ".." in relative.parts:
                        continue
                    if relative.as_posix().startswith(SOURCE_DIRS) or relative.name in ("LICENSE", "NOTICE"):
                        target = package / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(bundle.read(entry))
        else:
            for directory in SOURCE_DIRS:
                shutil.copytree(source / directory, package / directory)
            for name in ("LICENSE", "NOTICE"):
                shutil.copy2(source / name, package / name)
        # Resolve package substitution in a private copy, without ROS/ament and
        # without modifying the upstream checkout or its model parameters.
        for path in package.rglob("*.xacro"):
            path.write_text(path.read_text().replace("$(find franka_description)", str(package)))
        document = xacro.process_file(str(package / "robots/fr3v2/fr3v2.urdf.xacro"),
                                      mappings={"hand": "true", "with_sc": "false"})
        robot = ET.fromstring(document.toxml())
        visual_parts = {}
        for link in robot.findall("link"):
            for visual in list(link.findall("visual")):
                mesh_node = visual.find("geometry/mesh")
                if mesh_node is None or not mesh_node.attrib["filename"].endswith(".dae"):
                    continue
                relative = Path(mesh_node.attrib["filename"].removeprefix("package://franka_description/"))
                if relative not in visual_parts:
                    scene = trimesh.load(package / relative, force="scene")
                    parts = []
                    # Bullet applies a single OBJ material to a URDF visual.
                    # Split material regions into separate visuals, baking only
                    # scene-node transforms and preserving each original color.
                    for i, node_name in enumerate(scene.graph.nodes_geometry):
                        transform, geometry_name = scene.graph[node_name]
                        mesh = scene.geometry[geometry_name].copy()
                        mesh.apply_transform(transform)
                        color = mesh.visual.material.main_color / 255.
                        part = relative.with_name(f"{relative.stem}_part{i}.obj")
                        target = output / part
                        target.parent.mkdir(parents=True, exist_ok=True)
                        mesh.export(str(target), file_type="obj", mtl_name=f"{target.stem}.mtl")
                        parts.append((part.as_posix(), color))
                    visual_parts[relative] = parts
                link.remove(visual)
                for i, (filename, color) in enumerate(visual_parts[relative]):
                    part_visual = deepcopy(visual)
                    part_visual.attrib["name"] = f"{link.attrib['name']}_part{i}"
                    part_visual.find("geometry/mesh").attrib["filename"] = filename
                    material = part_visual.find("material")
                    if material is not None:
                        part_visual.remove(material)
                    material = ET.SubElement(part_visual, "material", name=f"{link.attrib['name']}_mat{i}")
                    ET.SubElement(material, "color", rgba=" ".join(map(str, color)))
                    link.append(part_visual)
        for node in robot.iter("mesh"):
            if not node.attrib["filename"].startswith("package://franka_description/"):
                continue
            relative = Path(node.attrib["filename"].removeprefix("package://franka_description/"))
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / relative, target)
            node.attrib["filename"] = target.relative_to(output).as_posix()
        # Fixed utility frames have no physical mass in the upstream model.
        for link in robot.findall("link"):
            if link.find("inertial") is None:
                inertia = ET.SubElement(link, "inertial")
                ET.SubElement(inertia, "mass", value="0")
                ET.SubElement(inertia, "inertia", ixx="0", iyy="0", izz="0", ixy="0", ixz="0", iyz="0")
        ET.ElementTree(robot).write(output / "fr3v2.urdf", encoding="utf-8", xml_declaration=True)
        for name in ("LICENSE", "NOTICE"):
            shutil.copy2(package / name, output / name)
        (output / "provenance.json").write_text(json.dumps({
            "repository": REPOSITORY, "revision": REVISION,
            "model": "fr3v2", "end_effector": "franka_hand_white",
            "changes": ("ROS-free paths; per-material COLLADA visuals converted to separate OBJ regions; "
                        "massless fixed frames"),
        }, indent=2) + "\n")
    print(f"Prepared official FR3v2: {output / 'fr3v2.urdf'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/robots/fr3v2")
    parser.add_argument("--source", type=Path, help="existing official checkout at the pinned revision")
    args = parser.parse_args()
    prepare(args.output, args.source)


if __name__ == "__main__":
    main()
