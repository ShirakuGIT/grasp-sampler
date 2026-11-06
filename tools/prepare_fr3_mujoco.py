"""Cache native Menagerie FR3v2 + standard Franka Hand assets for the workstation.

python tools/prepare_fr3_mujoco.py
Uses a pinned, sparse upstream checkout and complete official Franka visual
regions; preserves source licenses. No ROS or PyBullet dependency. --source
accepts an existing Menagerie checkout. Install the mujoco and fr3 extras first.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

REVISION = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
ROOT = Path(__file__).resolve().parents[1]


def prepare(output, source=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grasp-menagerie-") as tmp:
        if source is None:
            source = Path(tmp) / "menagerie"
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            REPOSITORY, str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", REVISION],
                            check=True)
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
            subprocess.run(["git", "-C", str(source), "sparse-checkout", "set",
                            "franka_fr3_v2", "franka_emika_panda"], check=True)
        source = Path(source)
        for folder in ("franka_fr3_v2", "franka_emika_panda"):
            shutil.copytree(source / folder, output / folder, dirs_exist_ok=True)
    arm = ET.parse(output / "franka_fr3_v2/fr3v2.xml").getroot()
    panda = ET.parse(output / "franka_emika_panda/panda.xml").getroot()
    # Resolve each native model's assets before merging them.
    for model, folder in ((arm, "franka_fr3_v2"), (panda, "franka_emika_panda")):
        for mesh in model.findall("asset/mesh"):
            mesh.attrib.setdefault("name", Path(mesh.attrib["file"]).stem)
            mesh.attrib["file"] = str(output / folder / "assets" / mesh.attrib["file"])
    # Isolate the hand's default classes, materials and names from the arm's.
    names = {node.attrib["name"] for node in panda.iter() if "name" in node.attrib}
    names |= {node.attrib["class"] for node in panda.iter("default") if "class" in node.attrib}
    reference_attributes = {"name", "class", "childclass", "mesh", "material", "joint", "joint1",
                            "joint2", "tendon", "body", "body1", "body2"}
    for node in panda.iter():
        for key, value in list(node.attrib.items()):
            if key in reference_attributes and value in names:
                node.attrib[key] = "gripper_" + value
    hand = deepcopy(panda.find(".//body[@name='gripper_hand']"))
    hand.attrib.update(pos="0 0 0", childclass="gripper_panda")
    ET.SubElement(hand, "site", name="hand_tcp", pos="0 0 0.1034", size="0.004", rgba="1 .7 .1 1")
    arm.find(".//body[@name='fr3v2_link8']").append(hand)
    arm.find("default").append(deepcopy(panda.find("default/default")))
    used_meshes = {node.attrib["mesh"] for node in hand.iter("geom") if "mesh" in node.attrib}
    for asset in panda.find("asset"):
        if asset.tag == "material" or asset.attrib.get("name") in used_meshes:
            arm.find("asset").append(deepcopy(asset))
    for section in ("tendon", "equality"):
        arm.append(deepcopy(panda.find(section)))
    gripper_actuator = deepcopy(panda.find("actuator/general[@name='gripper_actuator8']"))
    gripper_actuator.attrib["forcerange"] = "-40 40"
    arm.find("actuator").append(gripper_actuator)
    # The streamlined Menagerie arm omits several small visual regions. Use
    # the complete official Franka visual regions for the workstation while
    # retaining Menagerie's native collision geometry, dynamics and actuators.
    official = ROOT / "assets/robots/fr3v2/fr3v2.urdf"
    if not official.exists():
        from prepare_fr3 import prepare as prepare_visuals
        prepare_visuals(official.parent)
    description = ET.parse(official).getroot()
    visual_cache = output / "official_visuals"
    for name in ("LICENSE", "NOTICE", "provenance.json"):
        visual_cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(official.parent / name, visual_cache / name)
    for link in description.findall("link"):
        name = link.attrib["name"]
        if not name.startswith("fr3v2_link") or "accelerometer" in name:
            continue
        body = arm.find(f".//body[@name='{name}']")
        if body is None:
            continue
        for geom in list(body.findall("geom[@class='visual']")):
            body.remove(geom)
        for index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            original = official.parent / mesh.attrib["filename"]
            target = visual_cache / original.name
            shutil.copy2(original, target)
            mesh_name = f"full_{name}_{index}"
            ET.SubElement(arm.find("asset"), "mesh", name=mesh_name, file=str(target), inertia="shell")
            color = visual.find("material/color").attrib["rgba"]
            ET.SubElement(body, "geom", mesh=mesh_name, rgba=color, **{"class": "visual"})
    arm.remove(arm.find("keyframe"))
    arm.find("compiler").attrib["meshdir"] = "."
    ET.SubElement(arm, "option", timestep="0.002", integrator="implicitfast", cone="elliptic")
    ET.ElementTree(arm).write(output / "fr3v2_with_hand.xml", encoding="utf-8", xml_declaration=True)
    (output / "provenance.json").write_text(json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "arm": "franka_fr3_v2",
        "hand": "standard Franka Hand from franka_emika_panda", "tcp_offset_m": .1034,
        "changes": ("Hand attached at FR3v2 flange; complete official Franka visual regions; "
                    "native Menagerie physics; gripper force limited to 40N tendon total"),
    }, indent=2) + "\n")
    print(f"Prepared native MuJoCo model: {output / 'fr3v2_with_hand.xml'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/robots/fr3v2_mujoco")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.source)


if __name__ == "__main__":
    main()
