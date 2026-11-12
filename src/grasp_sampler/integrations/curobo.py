"""cuRobo current GoalToolPose and legacy 0.7 Pose exports (wxyz quaternions)."""
from __future__ import annotations

import numpy as np

from ..planning import PlanningGrasps, Stage


def goal_arrays(targets: PlanningGrasps, *, stage: Stage = "grasp", api="current", layout="goalset"):
    """Return NumPy position/quaternion arrays with cuRobo's required axes.

    Current: [batch,horizon,tool,goalset,xyz/quat]. Legacy: [batch,goalset,xyz/quat]
    for a goalset or [batch,xyz/quat] for batch planning. A goalset lets one
    start choose among N targets; batch mode pairs N starts with N targets.
    """
    if api not in ("current", "legacy") or layout not in ("goalset", "batch"):
        raise ValueError("api must be current/legacy and layout must be goalset/batch")
    if not len(targets):
        raise ValueError("cuRobo requires at least one goal")
    p, q = targets.components(stage, quaternion_order="wxyz")
    if api == "current":
        shape = (1, 1, 1, len(targets)) if layout == "goalset" else (len(targets), 1, 1, 1)
        return p.reshape(*shape, 3), q.reshape(*shape, 4)
    if layout == "goalset":
        return p[None], q[None]
    return p, q


def to_curobo(targets: PlanningGrasps, *, stage: Stage = "grasp", api="current", layout="goalset",
              device="cuda:0", dtype=None):
    """Create current ``GoalToolPose`` or legacy ``curobo.types.math.Pose``.

    Install cuRobo and PyTorch using NVIDIA's instructions for your CUDA stack.
    Pass the same device/dtype as your planner. Current export represents one
    tool link; multi-tool tasks must add their other link goals explicitly.
    This converts goals only and never starts planning or execution.
    """
    p, q = goal_arrays(targets, stage=stage, api=api, layout=layout)
    try:
        import torch
    except ImportError as exc:
        raise ImportError("cuRobo export requires PyTorch and a compatible cuRobo installation") from exc
    kwargs = {"device": device, "dtype": torch.float32 if dtype is None else dtype}
    p, q = (torch.as_tensor(np.ascontiguousarray(a), **kwargs) for a in (p, q))
    try:
        if api == "legacy":
            from curobo.types.math import Pose
            return Pose(position=p, quaternion=q, name=targets.tool_link)
        from curobo.types import GoalToolPose
        return GoalToolPose(tool_frames=[targets.tool_link], position=p, quaternion=q)
    except ImportError as exc:
        raise ImportError(
            f"Install cuRobo's {api} API; see docs/planning.md for version-specific setup"
        ) from exc
