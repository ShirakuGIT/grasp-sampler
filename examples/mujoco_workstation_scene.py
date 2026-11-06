"""Native MJCF tabletop workstation around the cached official FR3v2 model."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from grasp_sampler import load_mesh

TABLE_Z, BASE_Z = .72, .755
SHELF_Z, UPPER_Z = .84, 1.12
ROOT = Path(__file__).resolve().parents[1]
COLORS = [[.94, .25, .16, 1], [.12, .48, .94, 1], [.98, .65, .12, 1],
          [.56, .28, .86, 1], [.08, .72, .49, 1], [.95, .28, .56, 1]]


def object_mesh(i):
    name = ("ycb_mustard_bottle", "ycb_potted_meat_can", "ycb_foam_brick")[i]
    obj = load_mesh(ROOT / "assets/meshes" / f"{name}_metric_fp.glb")
    return obj.mesh, name.removeprefix("ycb_").replace("_", " ")


def numbers(values):
    return " ".join(f"{v:.8g}" for v in values)


def build_scene(count):
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
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient=".25 .25 .25", diffuse=".6 .6 .6", specular=".3 .3 .3")
    ET.SubElement(asset, "texture", name="studio_sky", type="skybox", builtin="gradient",
                  rgb1=".14 .18 .23", rgb2=".025 .04 .065", width="512", height="3072")
    ET.SubElement(asset, "material", name="table_material", rgba=".42 .51 .56 1",
                  specular=".35", shininess=".25", reflectance=".025")
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

    box("table", [1.28, 1.06, .045], [.44, 0, TABLE_Z - .0225], material="table_material")
    for x in [-.13, 1.01]:
        for y in [-.45, .45]:
            box(f"leg_{x}_{y}", [.055, .055, .67], [x, y, .36], material="steel_material")
            box(f"foot_{x}_{y}", [.09, .09, .03], [x, y, .015])
    for y in [-.45, .45]:
        box(f"rail_{y}", [1.18, .04, .065], [.44, y, .6], material="steel_material")
    box("mount", [.19, .19, BASE_Z - TABLE_Z], [0, 0, (BASE_Z + TABLE_Z) / 2],
        material="steel_material", collision=False)
    for x in [-.075, .075]:
        for y in [-.075, .075]:
            ET.SubElement(world, "geom", type="cylinder", size=".009 .002", pos=f"{x} {y} {BASE_Z+.002}",
                          rgba=".06 .08 .1 1", contype="0", conaffinity="0")
    box("pick_mat", [.36, .29, .002], [.45, -.18, TABLE_Z + .001], collision=False)
    # Two constrained cabinet bays. The 28 cm tier spacing rules out a full
    # top-down pregrasp for the tall object; the arm approaches through -X.
    for name, z in (("shelf_floor", SHELF_Z), ("upper_tier", UPPER_Z), ("shelf_roof", 1.40)):
        box(name, [.35, .64, .018], [.60, -.04, z-.009], color=".70 .74 .76 1")
        box(name+"_edge", [.012, .64, .025], [.425, -.04, z-.014], color=".28 .34 .39 1")
    box("shelf_back", [.016, .64, .68], [.785, -.04, 1.06], color=".20 .25 .30 1")
    for y in [-.37, .29]:
        box(f"shelf_side_{y}", [.36, .016, .68], [.60, y, 1.06], color=".55 .61 .65 1")
    for x in [.45, .75]:
        for y in [-.33, .25]:
            box(f"shelf_foot_{x}_{y}", [.025, .025, .11], [x, y, .775], material="steel_material")
    objects = []
    for i in range(count):
        mesh, label = object_mesh(i)
        size = mesh.extents
        position = [.50 if i == 0 else .44, -.16 + .16 * i, SHELF_Z + size[2] / 2 + .001]
        goal = (np.array([.50, -.16, UPPER_Z + size[2] / 2]) if i == 0 else
                np.array([.30, -.28 if i == 1 else .35, TABLE_Z + size[2] / 2]))
        yaw = .08 * (i - 1)
        body = ET.SubElement(world, "body", name=f"item_{i}", pos=numbers(position),
                             quat=numbers([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        ET.SubElement(body, "freejoint", name=f"item_{i}_free")
        kwargs = dict(name=f"item_{i}_geom", mass=".06", rgba=numbers(COLORS[i]),
                      friction="1.6 .015 .002", condim="4", solref=".004 1", solimp=".95 .99 .001")
        ET.SubElement(asset, "mesh", name=f"item_{i}_mesh", vertex=numbers(mesh.vertices.reshape(-1)),
                      face=" ".join(map(str, mesh.faces.reshape(-1))))
        kwargs.update(type="mesh", mesh=f"item_{i}_mesh")
        ET.SubElement(body, "geom", **kwargs)
        floor_z = UPPER_Z if i == 0 else TABLE_Z
        box(f"slot_{i}", [.10, .12, .0005], [goal[0], goal[1], floor_z + .0005],
            color=numbers([*COLORS[i][:3], .45]), collision=False)
        objects.append((mesh, label, goal))
    # Grip pads retain the native hand's contact geometry, with high-friction
    # rubber contact coefficients appropriate for this light-object example.
    for node in root.findall(".//default"):
        if "fingertip_pad_collision" in node.attrib.get("class", ""):
            geom = node.find("geom")
            geom.attrib.update(friction="2 .02 .002", condim="4", solref=".004 1")
    return ET.tostring(root, encoding="unicode"), objects, fixtures
