"""Import original household scans and precompute concave contact geometry.

Run with the original workspace at --sources (defaults to ~/Documents/Coding).
Requires trimesh, Pillow, scipy and coacd. Runtime uses only the saved assets.
YCB textures/UVs are retained; Ritz colors are transferred from the original
dense vertex-colored reconstruction onto the existing decimated geometry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import coacd
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def prepare_classic():
    """Retain the classic contact geometry; transfer original scan colors."""
    output = ROOT / "assets/household"
    for name, old_name in (("mustard", "mustard_bottle"), ("can", "potted_meat_can"),
                           ("foam", "foam_brick")):
        low = trimesh.load(ROOT/f"assets/meshes/ycb_{old_name}_metric_fp.glb", force="mesh")
        low.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        low.apply_translation(-low.bounds.mean(axis=0))
        dense = trimesh.load(output/f"{name}.obj", force="mesh", process=False)
        dense.vertices *= low.extents/dense.extents
        dense.visual = dense.visual.to_color()
        mesh = ritz_texture(low, dense)
        key = f"classic_{name}"
        mesh.visual.material.name = key
        text = trimesh.exchange.obj.export_obj(mesh, write_texture=False, mtl_name=f"{key}.mtl")
        (output/f"{key}.obj").write_text(text.rstrip()+"\n")
        (output/f"{key}.mtl").write_text(f"newmtl {key}\nKd 1 1 1\nmap_Kd {key}.png\n")
        mesh.visual.material.image.convert("RGB").save(output/f"{key}.png")


def ritz_texture(mesh, dense):
    """Bake source vertex colors into independent UV triangle tiles."""
    tile = 24
    side = int(np.ceil(np.sqrt(len(mesh.faces))))
    atlas = np.zeros((side*tile, side*tile, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    tree = cKDTree(dense.vertices)
    uv = []
    a, b = np.meshgrid(np.arange(tile)/(tile-1), np.arange(tile)/(tile-1))
    # Extend the triangle into its tile's unused half to avoid edge bleeding.
    weights = np.stack([1-a-b, a, b], axis=-1)
    weights = np.maximum(weights, 0)
    weights /= weights.sum(axis=-1, keepdims=True)
    for i, face in enumerate(mesh.faces):
        x, y = i % side, i // side
        points = weights @ mesh.vertices[face]
        nearest = tree.query(points.reshape(-1, 3))[1]
        colors = dense.visual.vertex_colors[nearest].reshape(tile, tile, 4)
        atlas[y*tile:(y+1)*tile, x*tile:(x+1)*tile] = colors
        uv.extend([[(x*tile+.5)/(side*tile), 1-(y*tile+.5)/(side*tile)],
                   [((x+1)*tile-.5)/(side*tile), 1-(y*tile+.5)/(side*tile)],
                   [(x*tile+.5)/(side*tile), 1-((y+1)*tile-.5)/(side*tile)]])
    out = trimesh.Trimesh(vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
                          faces=np.arange(len(mesh.faces)*3).reshape(-1, 3), process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=np.array(uv), image=Image.fromarray(atlas))
    return out


def prepare(sources, names):
    output = ROOT / "assets/household"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "bowl": sources / "bowl/meshes/bowl_textured.obj",
        "mug": sources / "025_mug/meshes/025_mug_textured.obj",
        "cup": sources / "065-f_cups/meshes/065-f_cups_textured.obj",
        "plate": sources / "029_plate/meshes/029_plate_textured.obj",
        "mustard": sources / "006_mustard_bottle/meshes/006_mustard_bottle_textured.obj",
        "foam": sources / "061_foam_brick/meshes/061_foam_brick_textured.obj",
        "can": (sources / "research/ntu/research_workspace/proposal/smcr/ycb_assets"
                / "010_potted_meat_can/010_potted_meat_can/google_16k/textured.obj"),
    }
    for name in names:
        print(f"Preparing {name}: original appearance", flush=True)
        if name == "ritz":
            low = trimesh.load(ROOT / "assets/meshes/ritz_crackers_metric_fp.glb",
                               force="mesh", process=False)
            source = (sources / "research/ntu/research_workspace/proposal/imagination/meshes"
                      / "ritz_cracker_box/ritz_cracker_box.mesh.metric-fp.glb")
            dense = trimesh.load(source, force="mesh", process=False)
            mesh = ritz_texture(low, dense)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        else:
            mesh = trimesh.load(paths[name], force="mesh", process=False)
        if name == "bowl":
            # Local bowl reconstruction is Y-up with unit diameter.
            mesh.apply_scale(.16 / max(mesh.extents))
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        # Native OBJ retains per-corner UV coordinates. Explicit MuJoCo
        # materials point at the image; MuJoCo does not import OBJ MTLs.
        mesh.visual.material.name = name
        # Source MTLs omit Kd. Use neutral white so the original image isn't
        # darkened by trimesh's fallback diffuse factor during GLB export.
        mesh.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
        text = trimesh.exchange.obj.export_obj(mesh, write_texture=False, mtl_name=f"{name}.mtl")
        (output/f"{name}.obj").write_text(text.rstrip()+"\n")
        (output/f"{name}.mtl").write_text(f"newmtl {name}\nKd 1 1 1\nmap_Kd {name}.png\n")
        mesh.visual.material.image.convert("RGB").save(output / f"{name}.png")
        y_up = mesh.copy()
        y_up.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
        y_up.export(output / f"{name}.glb")
        if name in ("mug", "cup", "plate", "bowl"):
            print(f"  decomposing {name} for cavity/handle contacts", flush=True)
            geometry = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
            coacd.set_log_level("warn")
            parts = coacd.run_coacd(coacd.Mesh(geometry.vertices, geometry.faces.astype(np.int32)),
                threshold=.0007 if name == "plate" else .003, real_metric=True,
                max_convex_hull=128 if name == "plate" else 40,
                preprocess_resolution=200 if name == "plate" else 60,
                resolution=1000, mcts_nodes=10, mcts_iterations=60,
                decimate=True, max_ch_vertex=64, seed=0)
        else:
            hull = mesh.convex_hull
            parts = [(hull.vertices, hull.faces)]
        arrays = {}
        for i, (vertices, faces) in enumerate(parts):
            arrays[f"vertices_{i}"] = vertices
            arrays[f"faces_{i}"] = faces
        np.savez_compressed(output / f"{name}_collision.npz", **arrays)
        print(f"  saved {len(parts)} convex parts; {len(mesh.faces)} visual faces", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path.home()/"Documents/Coding")
    parser.add_argument("--objects", nargs="+",
                        default=["ritz", "mug", "cup", "plate", "mustard", "can", "foam", "bowl"])
    args = parser.parse_args()
    prepare(args.sources, args.objects)
    if all((ROOT/f"assets/household/{name}.obj").exists() for name in ("mustard", "can", "foam")):
        prepare_classic()
