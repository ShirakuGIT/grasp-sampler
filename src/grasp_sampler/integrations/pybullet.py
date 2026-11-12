"""PyBullet / pybullet-planning targets; no simulator import needed for export."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..planning import PlanningGrasps, Stage


def to_pybullet(targets: PlanningGrasps, *, stage: Stage = "grasp"):
    """Return [((x,y,z), (qx,qy,qz,qw)), ...] tool-link targets.

    Supply simulation-world transforms when building ``targets``. These can
    be passed to calculateInverseKinematics or pybullet-planning's IK helpers.
    They are link-frame poses, not inertial/centre-of-mass poses.
    """
    p, q = targets.components(stage, quaternion_order="xyzw")
    return [
        (tuple(map(float, position)), tuple(map(float, quaternion))) for position, quaternion in zip(p, q)
    ]


def draw_grasps(
    grasps,
    object_world_pose,
    *,
    selected_index=None,
    rejected_indices=(),
    max_grasps=24,
    scale=1.0,
    physics_client_id=0,
):
    """Render real gripper glyphs: cyan candidates, coral rejects, gold selection.

    Returns visual-only body IDs; remove them with removeBody when advancing
    the visualization. Glyphs use the sampler's TCP axes (not a planner tool's
    axes), and have no collision geometry. ``object_world_pose`` can also place
    a scaled inspection inset, provided local translations scale with it.
    """
    import pybullet as p

    from ..uncertainty import validate_pose

    world = validate_pose(object_world_pose)
    if max_grasps < 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("max_grasps and scale must be positive")
    indices = np.linspace(0, len(grasps) - 1, min(len(grasps), max_grasps), dtype=int).tolist()
    if selected_index is not None:
        if not 0 <= selected_index < len(grasps):
            raise IndexError("selected_index out of range")
        indices = list(dict.fromkeys([*indices, selected_index]))
    rejected = set(rejected_indices)
    bodies = []
    for index in indices:
        g = grasps[index]
        local = validate_pose(g.pose)
        local[:3, 3] *= scale
        pose = world @ local
        selected = index == selected_index
        color = (
            [1.0, 0.74, 0.15, 1.0]
            if selected
            else [0.98, 0.29, 0.30, 0.35]
            if index in rejected
            else [0.15, 0.85, 0.94, 0.18]
        )
        width = g.width * scale
        depth, stem = 0.04 * scale, 0.035 * scale
        radius = (0.0018 if selected else 0.00065) * scale
        centers = [
            [width / 2, 0, -depth / 2],
            [-width / 2, 0, -depth / 2],
            [0, 0, -depth],
            [0, 0, -depth - stem / 2],
        ]
        extents = [[radius, radius, depth / 2]] * 2 + [
            [width / 2, radius, radius],
            [radius, radius, stem / 2],
        ]
        visual = p.createVisualShapeArray(
            [p.GEOM_BOX] * 4,
            halfExtents=extents,
            visualFramePositions=centers,
            rgbaColors=[color] * 4,
            physicsClientId=physics_client_id,
        )
        bodies.append(
            p.createMultiBody(
                0,
                -1,
                visual,
                pose[:3, 3].tolist(),
                Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
                physicsClientId=physics_client_id,
            )
        )
    return bodies
