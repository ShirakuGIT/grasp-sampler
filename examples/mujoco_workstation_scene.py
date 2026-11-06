"""Native MJCF tabletop workstation around the cached official FR3v2 model."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from mujoco_mount import add_mount

from grasp_sampler import load_mesh

TABLE_Z, BASE_Z = .72, .755
SHELF_Z, UPPER_Z = .84, 1.12
ROOT = Path(__file__).resolve().parents[1]
COLORS = [[.94, .25, .16, 1], [.12, .48, .94, 1], [.98, .65, .12, 1],
          [.56, .28, .86, 1], [.08, .72, .49, 1], [.95, .28, .56, 1]]


SUITES = {"classic": ("mustard", "can", "foam"), "household": ("ritz", "mug", "cup", "plate"),
          "expanded": ("ritz", "mug", "cup", "plate", "bowl", "mustard", "vase")}
LABELS = {"ritz": "Ritz cracker box", "mug": "mug", "cup": "cup", "plate": "plate",
          "bowl": "bowl", "vase": "curved vase", "mustard": "mustard bottle", "can": "potted meat can", "foam": "foam brick"}


def object_mesh(i, suite="classic"):
    name = SUITES[suite][i]
    if suite == "classic":
        old_name = ("mustard_bottle", "potted_meat_can", "foam_brick")[i]
        return load_mesh(ROOT/f"assets/meshes/ycb_{old_name}_metric_fp.glb").mesh, LABELS[name]
    path = ROOT / "assets/household" / f"{name}.glb"
    if not path.exists():
        raise FileNotFoundError(f"Missing original household asset: {path}")
    return load_mesh(path, process=False).mesh, LABELS[name]


def numbers(values):
    return " ".join(f"{v:.8g}" for v in values)


def build_scene(count, suite="classic"):
    path = ROOT / "assets/robots/fr3v2_mujoco/fr3v2_with_hand.xml"
    if not path.exists():
        raise FileNotFoundError("Run python tools/prepare_fr3_mujoco.py before launching the workstation")
    root = ET.parse(path).getroot()
    root.attrib["model"] = "FR3v2 grasp sampling workstation"
    world, asset = root.find("worldbody"), root.find("asset")
    world.find("body[@name='base']").attrib["pos"] = f"0 0 {BASE_Z}"
    ET.SubElement(root, "size", njmax="4000", nconmax="2000")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000", azimuth="135", elevation="-28")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="2")
    ET.SubElement(visual, "headlight", ambient=".25 .25 .25", diffuse=".6 .6 .6", specular=".3 .3 .3")
    ET.SubElement(asset, "texture", name="studio_sky", type="skybox", builtin="gradient",
                  rgb1=".14 .18 .23", rgb2=".025 .04 .065", width="512", height="3072")
    ET.SubElement(asset, "material", name="table_material", rgba=".42 .51 .56 1",
                  specular=".35", shininess=".25", reflectance="0")
    ET.SubElement(asset, "material", name="steel_material", rgba=".22 .29 .35 1", specular=".65", shininess=".5")
    ET.SubElement(world, "light", pos="0 -1 3", dir=".1 .3 -1", diffuse=".8 .85 .9", castshadow="true")
    ET.SubElement(world, "light", pos="1 1 2.5", dir="-.4 -.2 -1", diffuse=".5 .6 .7", castshadow="false")
    ET.SubElement(world, "geom", name="floor", type="plane", size="5 5 .05", rgba=".07 .09 .12 1")
    fixtures = []

    def box(name, size, pos, color=".06 .09 .13 1", material=None, collision=True):
        attrs = dict(name=name, type="box", size=numbers(np.array(size) / 2), pos=numbers(pos),
                     rgba=color, friction="1.2 .01 .001", group="0")
        if material:
            attrs["material"] = material
            attrs.pop("rgba")
        if not collision:
            attrs.update(contype="0", conaffinity="0")
        else:
            fixtures.append(name)
        return ET.SubElement(world, "geom", **attrs)

    box("table", [1.28, 1.40 if suite == "expanded" else 1.20 if suite != "classic" else 1.06, .045], [.44, 0, TABLE_Z - .0225], material="table_material")
    for x in [-.13, 1.01]:
        for y in [-.45, .45]:
            box(f"leg_{x}_{y}", [.055, .055, .67], [x, y, .36], material="steel_material")
            box(f"foot_{x}_{y}", [.09, .09, .03], [x, y, .015])
    for y in [-.45, .45]:
        box(f"rail_{y}", [1.18, .04, .065], [.44, y, .6], material="steel_material")
    add_mount(world, asset, TABLE_Z, BASE_Z)
    box("pick_mat", [.36, .29, .002], [.45, -.18, TABLE_Z + .001], collision=False)
    # Two constrained cabinet bays. The 28 cm tier spacing rules out a full
    # top-down pregrasp for the tall object; the arm approaches through -X.
    shelf_width = 1.04 if suite == "expanded" else .88 if suite != "classic" else .64
    for name, z in (("shelf_floor", SHELF_Z), ("upper_tier", UPPER_Z), ("shelf_roof", 1.52 if suite == "expanded" else 1.40)):
        box(name, [.35, shelf_width, .018], [.60, -.04, z-.009], color=".70 .74 .76 1")
        box(name+"_edge", [.012, shelf_width, .025], [.425, -.04, z-.014], color=".28 .34 .39 1")
    box("shelf_back", [.016, shelf_width, .80 if suite == "expanded" else .68], [.785, -.04, 1.12 if suite == "expanded" else 1.06], color=".20 .25 .30 1")
    for y in [-.04-shelf_width/2-.01, -.04+shelf_width/2+.01]:
        box(f"shelf_side_{y}", [.36, .016, .80 if suite == "expanded" else .68], [.60, y, 1.12 if suite == "expanded" else 1.06], color=".55 .61 .65 1")
    for x in [.45, .75]:
        for y in [-.33, .25]:
            box(f"shelf_foot_{x}_{y}", [.025, .025, .11], [x, y, .775], material="steel_material")
    objects = []
    for i in range(count):
        mesh, label = object_mesh(i, suite)
        name = SUITES[suite][i]
        size = mesh.extents
        position = [.50 if i == 0 else .44, -.16 + .16 * i, SHELF_Z + size[2] / 2 + .001]
        goal = (np.array([.50, -.16, UPPER_Z + size[2] / 2]) if i == 0 else
                np.array([.30, -.28 if i == 1 else .35, TABLE_Z + size[2] / 2]))
        yaw = .08 * (i - 1)
        if suite != "classic":
            position = ([.50, -.23, SHELF_Z+size[2]/2+.001],
                        [.54, -.06, UPPER_Z+size[2]/2+.001],
                        [.49, .17, SHELF_Z+size[2]/2+.001],
                        [.54, .20, UPPER_Z+size[2]/2+.001],
                        [.27, .46, TABLE_Z+size[2]/2+.001],
                        [.55, -.40, UPPER_Z+size[2]/2+.001],
                        [.54, .33, SHELF_Z+size[2]/2+.001])[i]
            goal = np.array(([.50, -.23, UPPER_Z+size[2]/2],
                             [.24, -.50, TABLE_Z+size[2]/2],
                             [.27, .43, TABLE_Z+size[2]/2],
                             [.26, .10, TABLE_Z+size[2]/2],
                             [.30, -.25, TABLE_Z+size[2]/2],
                             [.27, .43, TABLE_Z+size[2]/2],
                             [.57, .35, UPPER_Z+size[2]/2])[i])
            yaw = np.pi if name == "mug" else 0.
        if suite == "expanded":
            if i == 2:
                goal = np.array([.60, -.04, UPPER_Z+size[2]/2])
        body = ET.SubElement(world, "body", name=f"item_{i}", pos=numbers(position),
                             quat=numbers([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        ET.SubElement(body, "freejoint", name=f"item_{i}_free")
        kwargs = dict(
                      friction="1.6 .015 .002", condim="4", solref=".004 1", solimp=".95 .99 .001")
        directory = ROOT / "assets/household"
        visual_name = f"classic_{name}" if suite == "classic" else name
        ET.SubElement(asset, "texture", name=f"item_{i}_texture", type="2d", file=str(directory/f"{visual_name}.png"))
        ET.SubElement(asset, "material", name=f"item_{i}_material", texture=f"item_{i}_texture",
                      rgba="1 1 1 1", specular=".15", shininess=".15")
        ET.SubElement(asset, "mesh", name=f"item_{i}_visual", file=str(directory/f"{visual_name}.obj"))
        ET.SubElement(body, "geom", name=f"item_{i}_visual_geom", type="mesh", mesh=f"item_{i}_visual",
                      material=f"item_{i}_material", contype="0", conaffinity="0", group="1", mass="0")
        if suite == "classic":
            ET.SubElement(asset, "mesh", name=f"item_{i}_mesh", vertex=numbers(mesh.vertices.reshape(-1)),
                          face=" ".join(map(str, mesh.faces.reshape(-1))))
            ET.SubElement(body, "geom", name=f"item_{i}_geom", mass=".06", group="3",
                          type="mesh", mesh=f"item_{i}_mesh", **kwargs)
        else:
            parts = np.load(directory/f"{name}_collision.npz", allow_pickle=False)
            part_count = len(parts.files)//2
            mass = .03 if name == "plate" else .06
            weights = np.ones(part_count)
            if name in ("bowl", "vase"):
                for part in range(part_count):
                    triangles = parts[f"vertices_{part}"][parts[f"faces_{part}"]]
                    weights[part] = abs(np.einsum("ij,ij->", triangles[:, 0],
                        np.cross(triangles[:, 1], triangles[:, 2])))/6
                weights /= weights.sum()
            else:
                weights /= part_count
            for part in range(part_count):
                mesh_name = f"item_{i}_part_{part}"
                ET.SubElement(asset, "mesh", name=mesh_name,
                              vertex=numbers(parts[f"vertices_{part}"].reshape(-1)),
                              face=" ".join(map(str, parts[f"faces_{part}"].reshape(-1))))
                ET.SubElement(body, "geom", name=f"item_{i}_geom" if part == 0 else mesh_name,
                              type="mesh", mesh=mesh_name, group="3", mass=str(mass*weights[part]), **kwargs)
            parts.close()
        floor_z = float(goal[2]-size[2]/2)
        box(f"slot_{i}", [.10, .12, .0005], [goal[0], goal[1], floor_z + .0005],
            color=numbers([*COLORS[i % len(COLORS)][:3], .45]), collision=False)
        objects.append((mesh, label, goal))
    # Grip pads retain the native hand's contact geometry, with high-friction
    # rubber contact coefficients appropriate for this light-object example.
    for node in root.findall(".//default"):
        if "fingertip_pad_collision" in node.attrib.get("class", ""):
            geom = node.find("geom")
            geom.attrib.update(friction="2 .02 .002", condim="4", solref=".004 1")
            if suite != "classic":
                # Rolling resistance represents the finite 17 mm rubber pad,
                # rather than a torque-free point contact on a thin rim.
                geom.attrib.update(condim="6", friction="2 .02 .008")
                if node.attrib["class"].endswith("_1"):
                    # The stock central proxy leaves a 3 mm dead gap even at
                    # zero opening. Let the rubber pads meet for thin rims.
                    geom.attrib["size"] = ".0085 .0055 .0085"
    if suite != "classic":
        # Preload lets the native force-limited fingers hold thin handles/rims.
        # The same 80 mm opening and +/-40 N actuator limits are retained.
        actuator = root.find("actuator/general[@name='gripper_actuator8']")
        actuator.attrib.update(gainprm=f"{36/255} 0 0", biasprm="-16 -500 -20")
    return ET.tostring(root, encoding="unicode"), objects, fixtures
