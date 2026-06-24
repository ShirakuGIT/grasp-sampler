"""Oriented-bounding-box face grasps.

For each pair of opposing OBB faces whose separation fits the gripper, emit a
grasp centered on that axis, at a few offsets along the third axis. This gives
clean axis-aligned grasps for box-like and tilted objects.
"""
from __future__ import annotations

import numpy as np

from .types import Grasp, GraspConfig, ObjMesh

# Fingertip-to-palm graspable depth (m). The TCP dips at most this far past the
# approach face so the palm stays outside the object.
_FINGER_REACH = 0.035


def obb_face_grasps(obj: ObjMesh, cfg: GraspConfig) -> list[Grasp]:
    R_obb, extents, center = _obb_frame(obj)
    grasps: list[Grasp] = []

    for axis in range(3):
        width = extents[axis]
        if not (cfg.gripper_min_width <= width <= cfg.gripper_max_width):
            continue
        closing = R_obb[:, axis]
        for approach_axis in (i for i in range(3) if i != axis):
            half_depth = extents[approach_axis] / 2.0
            for sign in (+1.0, -1.0):
                z = -sign * R_obb[:, approach_axis]       # approach into the object
                y = np.cross(z, closing)
                if np.linalg.norm(y) < 1e-6:
                    continue
                y /= np.linalg.norm(y)
                z = np.cross(closing, y)
                R = np.column_stack([closing, y, z])

                dip = min(_FINGER_REACH, half_depth)
                face_center = center + sign * half_depth * R_obb[:, approach_axis]
                tcp = face_center + z * dip
                third = next(k for k in range(3) if k not in (axis, approach_axis))
                for frac in (-0.3, 0.0, 0.3):
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = tcp + frac * extents[third] * R_obb[:, third]
                    grasps.append(Grasp(pose, "obb_face", float(width),
                                        meta={"axis": axis, "approach": approach_axis,
                                              "offset": frac}))
    return grasps


def _obb_frame(obj: ObjMesh):
    """Pick a trustworthy oriented box. Falls back to the axis-aligned box when
    trimesh's OBB is degenerate (its volume disagrees with the AABB)."""
    aabb_vol = float(np.prod(obj.extents))
    obb_vol = float(np.prod(obj.obb_extents))
    if not (0.85 * aabb_vol <= obb_vol <= 1.15 * aabb_vol):
        return np.eye(3), obj.extents.copy(), np.zeros(3)
    return obj.obb_transform[:3, :3], obj.obb_extents, obj.obb_transform[:3, 3]
