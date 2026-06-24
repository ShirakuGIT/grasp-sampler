"""Mesh loading and coarse geometry classification."""
from __future__ import annotations

import numpy as np
import trimesh

from .types import ObjMesh

# Rotate 90 deg about X so a Y-up mesh stands upright (Y -> Z). Most GLB exports
# are Y-up; adjust or pass an already-upright mesh if yours differs.
_R_X90 = np.array([[1, 0, 0, 0],
                   [0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]], float)


def load_mesh(path, *, upright_transform: np.ndarray | None = _R_X90) -> ObjMesh:
    """Load a mesh file, stand it upright, and center it at its OBB centroid.

    Parameters
    ----------
    path : str or Path
        Any format trimesh can read (GLB, OBJ, STL, PLY, ...).
    upright_transform : (4, 4) array or None
        Applied before centering. Defaults to a Y-up -> Z-up rotation. Pass
        ``None`` if the mesh is already oriented as you want it.
    """
    raw = trimesh.load(str(path), force="scene")
    mesh = (trimesh.util.concatenate(list(raw.geometry.values()))
            if isinstance(raw, trimesh.Scene) else raw)
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, float),
                           faces=np.asarray(mesh.faces), process=False)
    if upright_transform is not None:
        mesh.apply_transform(upright_transform)
    mesh.apply_translation(-mesh.bounding_box.centroid)

    extents = np.asarray(mesh.bounding_box.extents, float)
    try:
        obb_transform = mesh.bounding_box_oriented.primitive.transform.copy()
        obb_extents = np.asarray(mesh.bounding_box_oriented.primitive.extents, float)
    except Exception:
        obb_transform = np.eye(4)
        obb_extents = extents.copy()

    return ObjMesh(mesh, extents, obb_transform, obb_extents, _classify(extents, mesh))


def _classify(extents: np.ndarray, mesh: trimesh.Trimesh) -> str:
    """Coarse shape class from sphericity, footprint aspect, and footprint fill.

    The XY footprint fill (hull area / bbox area) separates a square box (fill ~1)
    from a cylinder (square bbox, circular footprint, fill ~0.78) -- extents alone
    cannot tell them apart.
    """
    span = extents.max()
    sphericity = (span - extents.min()) / span
    if sphericity < 0.12:
        return "sphere"

    fx, fy = extents[0], extents[1]
    foot_ratio = min(fx, fy) / max(fx, fy)
    try:
        from scipy.spatial import ConvexHull
        pts, _ = trimesh.sample.sample_surface(mesh, 2500)
        fill = float(ConvexHull(pts[:, :2]).volume) / (fx * fy)   # 2D hull -> area
    except Exception:
        fill = 1.0

    if fill >= 0.90:
        return "box"
    if foot_ratio < 0.72:
        return "irregular"
    if foot_ratio >= 0.9 and 0.70 <= fill < 0.90:
        return "cylinder"
    if foot_ratio >= 0.9:
        return "box"
    return "irregular"
