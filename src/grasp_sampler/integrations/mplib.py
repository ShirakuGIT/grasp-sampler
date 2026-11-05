"""MPlib Pose goals (including MPlib/SAPIEN workflows)."""
from __future__ import annotations

from ..planning import PlanningGrasps, Stage


def to_mplib(targets: PlanningGrasps, *, stage: Stage = "grasp"):
    """Return mplib.Pose targets in the frame expected by Planner.plan_pose."""
    try:
        from mplib import Pose
    except ImportError as exc:
        raise ImportError("MPlib export requires mplib; install it in your planning environment") from exc
    p, q = targets.components(stage, quaternion_order="wxyz")
    return [Pose(p=position, q=quaternion) for position, quaternion in zip(p, q)]
