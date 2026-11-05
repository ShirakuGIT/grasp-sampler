"""PyBullet / pybullet-planning targets; no simulator import needed for export."""
from __future__ import annotations

from ..planning import PlanningGrasps, Stage


def to_pybullet(targets: PlanningGrasps, *, stage: Stage = "grasp"):
    """Return [((x,y,z), (qx,qy,qz,qw)), ...] tool-link targets.

    Supply simulation-world transforms when building ``targets``. These can
    be passed to calculateInverseKinematics or pybullet-planning's IK helpers.
    They are link-frame poses, not inertial/centre-of-mass poses.
    """
    p, q = targets.components(stage, quaternion_order="xyzw")
    return [(tuple(map(float, position)), tuple(map(float, quaternion)))
            for position, quaternion in zip(p, q)]
