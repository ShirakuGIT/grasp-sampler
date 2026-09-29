"""Mesh loading and coarse geometry classification."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from .types import GeomClass, ObjMesh

# Rotate 90 deg about X so a Y-up mesh stands upright (Y -> Z). Most GLB exports
# are Y-up; adjust or pass an already-upright mesh if yours differs.
_R_X90 = np.array([[1, 0, 0, 0],
                   [0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]], float)


def load_mesh(path, *, upright_transform: np.ndarray | None = _R_X90,
              center: bool = True, process: bool = True) -> ObjMesh:
    """Load geometry, optionally orient it and center its axis-aligned bounds.

    Use ``center=False, upright_transform=None`` to preserve the file frame.
    Scene node transforms are baked into scene coordinates. ``source_to_mesh``
    maps those coordinates into the returned mesh frame; compose a perception
    pose as ``T_W_mesh = T_W_source @ inv(obj.source_to_mesh)``.

    Parameters
    ----------
    path : str or Path
        Any format trimesh can read (GLB, OBJ, STL, PLY, ...).
    upright_transform : (4, 4) array or None
        Applied before centering. Defaults to a Y-up -> Z-up rotation. Pass
        ``None`` if the mesh is already oriented as you want it.
    center : bool
        Translate the axis-aligned bounds centroid to zero (default True).
    process : bool
        Retain trimesh's default vertex merging/cleanup. Set False to preserve
        raw vertices/topology too; this is independent of frame preservation.
    """
    raw = trimesh.load(str(path), force="scene", process=process)
    if isinstance(raw, trimesh.Scene):
        mesh = (raw.to_geometry() if hasattr(raw, "to_geometry")
                else raw.dump(concatenate=True))
    else:
        mesh = raw
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, float),
                           faces=np.asarray(mesh.faces), process=False)
    source_to_mesh = np.eye(4)
    if upright_transform is not None:
        mesh.apply_transform(upright_transform)
        source_to_mesh = np.asarray(upright_transform, float).copy()
    if center:
        shift = -mesh.bounding_box.centroid
        mesh.apply_translation(shift)
        translation = np.eye(4)
        translation[:3, 3] = shift
        source_to_mesh = translation @ source_to_mesh

    extents = np.asarray(mesh.bounding_box.extents, float)
    try:
        obb_transform = mesh.bounding_box_oriented.primitive.transform.copy()
        obb_extents = np.asarray(mesh.bounding_box_oriented.primitive.extents, float)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):   # degenerate OBB
        obb_transform = np.eye(4)
        obb_extents = extents.copy()

    return ObjMesh(mesh, extents, obb_transform, obb_extents,
                   _classify(extents, mesh), source_to_mesh)


def _classify(extents: np.ndarray, mesh: trimesh.Trimesh) -> GeomClass:
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
        pts, _ = trimesh.sample.sample_surface(mesh, 2500)
        fill = float(ConvexHull(pts[:, :2]).volume) / (fx * fy)   # 2D hull -> area
    except (RuntimeError, ValueError):   # degenerate footprint hull (QhullError)
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
