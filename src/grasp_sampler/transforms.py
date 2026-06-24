"""Frame conversions: object-local TCP grasps -> world / robot poses.

The sampler emits grasps as TCP poses in the object's local frame. Downstream
motion planners usually want them placed in the world and expressed in the
robot's end-effector (flange) frame. These helpers do both, as batched arrays.
"""
from __future__ import annotations

import numpy as np

from .types import Grasp

# Map a TCP frame (+X closing, +Z approach) to the Franka end-effector / flange
# frame (+Y closing, +Z approach). Columns are the EE axes in TCP coordinates.
_TCP_TO_PANDA_EE = np.array([[0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]], float)


def stack_poses(grasps: list[Grasp]) -> np.ndarray:
    """Collect ``Grasp.pose`` matrices into a single (N, 4, 4) array."""
    if not grasps:
        return np.zeros((0, 4, 4), float)
    return np.stack([g.pose for g in grasps]).astype(float)


def to_world(poses: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
    """Place object-local TCP poses into the world: ``object_pose @ pose``.

    ``object_pose`` is the (4, 4) world pose of the object whose local frame the
    grasps were sampled in.
    """
    poses = np.asarray(poses, float).reshape(-1, 4, 4)
    return np.asarray(object_pose, float)[None] @ poses


def tcp_to_ee(poses: np.ndarray, *, standoff: float = 0.105,
              remap: np.ndarray = _TCP_TO_PANDA_EE) -> np.ndarray:
    """Convert TCP poses to end-effector (flange) poses.

    The flange sits ``standoff`` metres behind the TCP along the approach axis
    (0.105 m for a Franka Panda). ``remap`` rotates the closing axis from the TCP
    convention (+X) to the gripper convention (+Y by default).
    """
    poses = np.asarray(poses, float).reshape(-1, 4, 4)
    out = poses.copy()
    out[:, :3, :3] = poses[:, :3, :3] @ remap
    out[:, :3, 3] = poses[:, :3, 3] - standoff * out[:, :3, 2]   # back off along approach
    return out
