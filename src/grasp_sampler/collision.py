"""Shared contact and gripper-body geometry checks.

Used by the antipodal sampler (nominal generation) and the robust evaluator
(fixed-command validation), so both apply the same definition of "valid".

Gripper-body clearance uses Open3D signed-distance queries (Embree BVH, float32)
when the mesh is watertight, and falls back to trimesh otherwise.
"""
from __future__ import annotations

import weakref

import numpy as np
import trimesh

from .types import GraspConfig

try:  # exact BVH signed-distance queries, ~100x faster than trimesh
    import open3d as _o3d
except ImportError:  # pragma: no cover
    _o3d = None

_SCENES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def ray_cast_proxy(mesh: trimesh.Trimesh, cfg: GraspConfig) -> trimesh.Trimesh:
    """Return the solid used for contact ray casts, per ``cfg.contact_geometry_policy``.

    ``"auto"`` substitutes the convex hull for hollow scans.
    """
    policy = cfg.contact_geometry_policy
    if policy == "mesh":
        return mesh
    if policy == "convex_hull":
        return mesh.convex_hull
    if policy != "auto":
        raise ValueError(f"unknown contact_geometry_policy: {policy!r}")
    try:
        hull = mesh.convex_hull
        if hull.volume > 0 and mesh.volume / hull.volume < cfg.hull_volume_ratio:
            return hull
    except (RuntimeError, ValueError):   # degenerate hull (scipy QhullError is a RuntimeError)
        pass
    return mesh


def friction_cones_valid(n1, n2, closing, cone: float) -> bool:
    """Outward normals: -n1 and +n2 must align with p1 -> p2 (+TCP X)."""
    return bool(np.dot(-np.asarray(n1), closing) >= cone and np.dot(n2, closing) >= cone)


def gripper_body_points(center, closing, approach, half: float, cfg: GraspConfig) -> np.ndarray:
    """Sample the palm bar and wrist stem behind the fingertips as an (N, 3) array."""
    finger_len = cfg.finger_len
    base_left = center + closing * half - approach * finger_len
    base_right = center - closing * half - approach * finger_len
    palm_center = center - approach * finger_len
    wrist = palm_center - approach * (cfg.wrist_stem - finger_len)

    t = np.linspace(0.0, 1.0, 5)[:, None]
    palm = base_left[None, :] * (1.0 - t) + base_right[None, :] * t
    s = np.linspace(0.0, 1.0, 4)[:, None]
    stem = palm_center[None, :] * (1.0 - s) + wrist[None, :] * s
    return np.vstack([palm, stem])


def bodies_clear(mesh: trimesh.Trimesh, poses, closing, approach, cfg: GraspConfig) -> bool:
    """True if the gripper body clears ``mesh`` at every pose.

    ``poses`` is a list of ``(center, half_opening)`` pairs sharing one closing and
    approach direction. Only the body behind the fingertips is checked; the
    fingertips straddle the object by design. All poses share one batched query.
    """
    pts = np.vstack([gripper_body_points(c, closing, approach, h, cfg) for c, h in poses])
    scene = _signed_distance_scene(mesh)
    if scene is not None:
        # Negative inside the mesh. float32 limits precision to ~1e-7 m.
        sd = scene.compute_signed_distance(_o3d.core.Tensor(pts.astype(np.float32))).numpy()
        return bool((sd >= 0).all() and np.abs(sd).min() >= cfg.spine_clearance)
    if mesh.contains(pts).any():
        return False
    _, dist, _ = trimesh.proximity.ProximityQuery(mesh).on_surface(pts)
    return bool(np.min(dist) >= cfg.spine_clearance)


def _signed_distance_scene(mesh: trimesh.Trimesh):
    """Cached Open3D raycasting scene for a watertight mesh, else None."""
    if _o3d is None or not mesh.is_watertight:
        return None
    scene = _SCENES.get(mesh)
    if scene is None:
        scene = _o3d.t.geometry.RaycastingScene()
        scene.add_triangles(_o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                            _o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        _SCENES[mesh] = scene
    return scene
