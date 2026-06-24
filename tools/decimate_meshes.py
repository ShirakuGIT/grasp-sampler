"""Decimate a folder of meshes into small, geometry-only assets.

Foundation-scan GLBs carry dense triangles and baked textures, so they run to
many megabytes each. For grasp sampling only the geometry matters, so we drop
the texture/material and collapse the mesh to a target face count. Output GLBs
are typically a few tens of KB -- small enough to commit.

    python tools/decimate_meshes.py <src_dir> <out_dir> [--faces 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

_EXTS = (".glb", ".gltf", ".obj", ".stl", ".ply")


def decimate(path: Path, faces: int) -> trimesh.Trimesh:
    """Load any mesh, merge to a single geometry, strip visuals, simplify."""
    raw = trimesh.load(str(path), force="scene")
    mesh = (trimesh.util.concatenate(list(raw.geometry.values()))
            if isinstance(raw, trimesh.Scene) else raw)
    # Geometry only -- discard textures, UVs, vertex colors.
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, float),
                           faces=np.asarray(mesh.faces), process=True)
    if len(mesh.faces) > faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=faces)
        except Exception as exc:                       # pragma: no cover
            print(f"    (simplify failed, keeping full res: {exc})")
    return mesh


def main() -> None:
    ap = argparse.ArgumentParser(description="decimate meshes for grasp sampling")
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--faces", type=int, default=2000, help="target face count")
    args = ap.parse_args()

    src, out = Path(args.src_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in _EXTS)
    if not files:
        raise SystemExit(f"no meshes found in {src}")

    for f in files:
        mesh = decimate(f, args.faces)
        dst = out / (f.stem + ".glb")
        mesh.export(dst)
        kb_in, kb_out = f.stat().st_size / 1024, dst.stat().st_size / 1024
        print(f"{f.name:40s} {len(mesh.faces):5d}f  "
              f"{kb_in:8.0f}KB -> {kb_out:6.0f}KB")


if __name__ == "__main__":
    main()
