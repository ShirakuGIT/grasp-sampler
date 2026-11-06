"""Object inventory and appearance-preserving native viewer assets."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from grasp_sampler import load_mesh

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GalleryItem:
    path: Path
    label: str
    variant: str
    mesh_path: Path
    texture_path: Path
    extents: np.ndarray
    scale: float


def inventory(root=ROOT / "assets"):
    """Include every object GLB, retaining distinct compact/scanned variants.

    OBJ/MTL/PNG sidecars are representations of the same object; robot link
    assets and convex collision pieces are not separate graspable objects.
    """
    return [path for folder in ("meshes", "household")
            for path in sorted((Path(root)/folder).glob("*.glb"))]


def vertex_color_texture(mesh):
    """Bake interpolated original vertex colors into separate triangle tiles."""
    tile = 12
    side = int(np.ceil(np.sqrt(len(mesh.faces))))
    pixels = np.zeros((side*tile, side*tile, 4), np.uint8)
    pixels[..., 3] = 255
    a, b = np.meshgrid(np.linspace(0, 1, tile), np.linspace(0, 1, tile))
    weights = np.maximum(np.stack([1-a-b, a, b], axis=-1), 0)
    weights /= weights.sum(axis=-1, keepdims=True)
    colors = mesh.visual.vertex_colors[mesh.faces]
    uv = []
    for i, face_colors in enumerate(colors):
        x, y = i % side, i // side
        pixels[y*tile:(y+1)*tile, x*tile:(x+1)*tile] = weights @ face_colors
        uv.extend([[(x*tile+.5)/(side*tile), 1-(y*tile+.5)/(side*tile)],
                   [((x+1)*tile-.5)/(side*tile), 1-(y*tile+.5)/(side*tile)],
                   [(x*tile+.5)/(side*tile), 1-((y+1)*tile-.5)/(side*tile)]])
    out = trimesh.Trimesh(mesh.vertices[mesh.faces].reshape(-1, 3),
                          np.arange(len(mesh.faces)*3).reshape(-1, 3), process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), image=Image.fromarray(pixels))
    return out


def prepare_item(path, cache):
    obj = load_mesh(path, process=False)
    mesh = obj.mesh.copy()
    key = hashlib.sha256(f"gallery-v1:{path}:{path.stat().st_mtime_ns}".encode()).hexdigest()[:16]
    directory = Path(cache)/key
    directory.mkdir(parents=True, exist_ok=True)
    obj_path, png_path = directory/"visual.obj", directory/"visual.png"
    restored = ROOT/"assets/gallery_colors"/path.stem
    if path.parent.name == "meshes" and restored.with_suffix(".obj").exists():
        obj_path, png_path = restored.with_suffix(".obj"), restored.with_suffix(".png")
    if not obj_path.exists() or not png_path.exists():
        if mesh.visual.kind != "texture":
            mesh = vertex_color_texture(mesh)
        material = mesh.visual.material
        if hasattr(material, "to_simple"):
            material = material.to_simple()
        image = material.image
        if image is None:
            mesh = vertex_color_texture(mesh)
            image = mesh.visual.material.image
        image = image.convert("RGB")
        image.thumbnail((1024, 1024), getattr(Image, "Resampling", Image).LANCZOS)
        image.save(png_path)
        obj_path.write_text(trimesh.exchange.obj.export_obj(mesh, write_texture=False).rstrip()+"\n")
    label = path.stem.replace("_metric_fp", "").removeprefix("ycb_").replace("_", " ").title()
    labels = {"Ritz Crackers": "Ritz Cracker Box", "Can": "Potted Meat Can", "Foam": "Foam Brick",
              "Mustard": "Mustard Bottle", "Vase": "Blue Vase"}
    return GalleryItem(path, labels.get(label, label),
                       "SCANNED MESH" if path.parent.name == "household" else "COMPACT MESH",
                       obj_path, png_path, obj.extents, .235/float(obj.extents.max()))


def grid_layout(count, columns=8, rows=5):
    if count > columns*rows:
        raise ValueError("The grid is too small for the object inventory")
    return np.array([[(i % columns-(columns-1)/2)*.38, 0.,
                      (rows-1-i//columns)*.34] for i in range(count)])


def scene_xml(items, cache, grid=False):
    root = ET.Element("mujoco", model="Grasp Sampler / Object Turntable")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "statistic", center="0 0 1.15" if grid else "0 0 .15",
                  extent="5" if grid else ".65")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="2560", offheight="1600")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="2")
    ET.SubElement(visual, "headlight", ambient=".3 .3 .3", diffuse=".55 .55 .55")
    asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1=".035 .04 .05", rgb2=".035 .04 .05", width="256", height="1536")
    ET.SubElement(world, "light", pos="-.5 .4 1.2", dir=".3 -.2 -1", diffuse=".8 .8 .8", directional="true", castshadow="false" if grid else "true")
    ET.SubElement(world, "light", pos=".5 -.4 .8", dir="-.5 .3 -1", diffuse=".5 .5 .5", directional="true", castshadow="false")
    for i, item in enumerate(items):
        scale = " ".join([str(item.scale)]*3)
        ET.SubElement(asset, "mesh", name=f"mesh_{i}", file=str(item.mesh_path), scale=scale)
        ET.SubElement(asset, "texture", name=f"texture_{i}", type="2d", file=str(item.texture_path))
        ET.SubElement(asset, "material", name=f"material_{i}", texture=f"texture_{i}", rgba="1 1 1 1",
                      specular=".12", shininess=".12")
        body = ET.SubElement(world, "body", name=f"object_{i}", mocap="true", pos="0 0 -20")
        ET.SubElement(body, "geom", name=f"visual_{i}", type="mesh", mesh=f"mesh_{i}",
                      material=f"material_{i}", rgba="1 1 1 0", contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")
