"""Bounding-box primitive grasps: top-down and side approaches.

Adapted from the ``get_top_grasps`` / ``get_side_grasps`` primitives in
ss-pybullet (pybullet-planning) by Caelan Reed Garrett, re-expressed with NumPy.

    Repo:    https://github.com/caelan/pybullet-planning
    License: MIT (Copyright (c) Caelan Reed Garrett)

Each function returns a list of ``(pose, meta)`` tuples, where ``pose`` is a
(4, 4) TCP pose in the object frame (+Z approach, +X closing, +Y completion).
"""
from __future__ import annotations

import numpy as np


def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def top_grasps(extents, *, max_width: float = 0.08, yaw_steps: int = 4,
               grasp_length: float = 0.0, under: bool = True):
    """Top-down grasps on a box: the hand descends along -Z onto the top face.

    A yaw whose finger span exceeds ``max_width`` is dropped (object too wide).
    ``under`` also emits each grasp flipped 180 deg about its approach axis.
    """
    ex, ey, ez = extents
    R_down = _Rx(np.pi)                      # TCP +Z -> object -Z (approach down)
    grasps = []
    for k in range(yaw_steps):
        yaw = np.pi * k / max(1, yaw_steps)
        R = R_down @ _Rz(yaw)
        close_dir = R[:, 0]
        span = abs(close_dir[0]) * ex + abs(close_dir[1]) * ey
        if span > max_width:
            continue
        t = np.array([0.0, 0.0, ez / 2.0 - grasp_length])
        meta = {"type": "top", "yaw": float(yaw), "span": float(span)}
        grasps.append((_pose(R, t), meta))
        if under:
            grasps.append((_pose(R @ _Rz(np.pi), t), {**meta, "yaw": float(yaw + np.pi)}))
    return grasps


# Per-face: (approach direction into the object, hand-offset direction, half-thickness)
_FACE_DEFS = {
    0: (np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]), 0),   # +X face
    1: (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), 0),   # -X face
    2: (np.array([0, -1.0, 0]), np.array([0, 1.0, 0]), 1),   # +Y face
    3: (np.array([0, 1.0, 0]), np.array([0, -1.0, 0]), 1),   # -Y face
}


def side_grasps(extents, *, max_width: float = 0.08, roll_steps: int = 2,
                grasp_length: float = 0.0, under: bool = True, faces=(0, 1, 2, 3)):
    """Lateral grasps into the four vertical side faces of a box.

    ``roll=0`` makes the fingers span the object height; ``roll=pi/2`` spans the
    cross dimension. Faces whose straddled dimension exceeds ``max_width`` drop.
    """
    ex, ey, ez = extents
    half = {0: ex / 2.0, 1: ex / 2.0, 2: ey / 2.0, 3: ey / 2.0}
    grasps = []
    for f in faces:
        approach, offdir, _ = _FACE_DEFS[f]
        z_axis = approach / np.linalg.norm(approach)
        x_axis = np.array([0.0, 0.0, 1.0])            # closing axis starts vertical
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        R0 = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
        for r in range(roll_steps):
            roll = np.pi * r / max(1, roll_steps)
            R = R0 @ _Rz(roll)
            close_dir = R[:, 0]
            span = (abs(close_dir[0]) * ex + abs(close_dir[1]) * ey
                    + abs(close_dir[2]) * ez)
            if span > max_width:
                continue
            t = offdir * (half[f] - grasp_length)
            meta = {"type": "side", "face": int(f), "roll": float(roll), "span": float(span)}
            grasps.append((_pose(R, t), meta))
            if under:
                grasps.append((_pose(R @ _Rz(np.pi), t), {**meta, "roll": float(roll + np.pi)}))
    return grasps
