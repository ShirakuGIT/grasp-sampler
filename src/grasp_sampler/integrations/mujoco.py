"""MuJoCo targets and mocap/site helpers using the official mujoco package."""
from __future__ import annotations

import numpy as np

from ..planning import PlanningGrasps, Stage


def to_mujoco(targets: PlanningGrasps, *, stage: Stage = "grasp"):
    """Return (position[N,3], quaternion[N,4]) arrays; quaternion order is wxyz.

    Targets must be in the simulation world. Apply tool calibration if the
    controlled site/mocap body differs from the sampler TCP.
    """
    return targets.components(stage, quaternion_order="wxyz")


def set_mocap_target(model, data, body_name: str, targets: PlanningGrasps, *,
                     index: int = 0, stage: Stage = "grasp"):
    """Set one named mocap body's world target and refresh forward kinematics.

    Mutates only that body's mocap position/orientation, then calls mj_forward.
    The XML must supply the mocap body and any controller/weld to the robot.
    This does not solve IK or write robot joints/controls.
    """
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"unknown MuJoCo body: {body_name!r}")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"body {body_name!r} is not a mocap body")
    if not isinstance(index, (int, np.integer)) or not 0 <= index < len(targets):
        raise IndexError("target index out of range")
    p, q = to_mujoco(targets, stage=stage)
    data.mocap_pos[mocap_id] = p[index]
    data.mocap_quat[mocap_id] = q[index]
    mujoco.mj_forward(model, data)


def site_world_pose(model, data, site_name: str) -> np.ndarray:
    """Read a site's 4x4 world transform after refreshing forward kinematics."""
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"unknown MuJoCo site: {site_name!r}")
    mujoco.mj_forward(model, data)
    pose = np.eye(4)
    pose[:3, 3] = data.site_xpos[site_id]
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return pose
