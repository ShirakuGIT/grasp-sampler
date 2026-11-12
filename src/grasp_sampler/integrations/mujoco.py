"""MuJoCo targets and mocap/site helpers using the official mujoco package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..planning import PlanningGrasps, Stage
from ..uncertainty import validate_pose


@dataclass
class SiteIKResult:
    """Joint-order-explicit IK result; no trajectory or collision guarantee."""

    joint_names: tuple
    positions: np.ndarray
    success: bool
    position_error: float
    orientation_error: float
    iterations: int


def solve_site_ik(
    model,
    data,
    site_name,
    target_pose,
    joint_names,
    *,
    seed=None,
    max_iterations=100,
    position_tolerance=0.001,
    orientation_tolerance=0.02,
    damping=0.03,
) -> SiteIKResult:
    """Damped site IK on named hinge/slide joints, without mutating ``data``.

    Targets are world SE(3) matrices. Uses MuJoCo's analytic Jacobian, respects
    joint limits, and returns explicit residuals. Collision/path checks remain
    separate. Free/ball joints must not appear in ``joint_names``.
    """
    import mujoco

    target = validate_pose(target_pose)
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    if (
        max_iterations < 1
        or damping <= 0
        or not np.isfinite(damping)
        or not np.isfinite([position_tolerance, orientation_tolerance]).all()
        or position_tolerance <= 0
        or orientation_tolerance <= 0
    ):
        raise ValueError("IK iteration count, damping and tolerances must be positive")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if min(ids) < 0 or site < 0:
        raise ValueError("unknown joint or site name")
    if any(
        int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        for j in ids
    ):
        raise ValueError("IK supports only named hinge/slide joints")
    q_indices, dofs = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    if seed is not None:
        values = np.asarray(seed, float)
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise ValueError("seed must have one finite value per named joint")
        scratch.qpos[q_indices] = values
    lower = np.where(model.jnt_limited[ids], model.jnt_range[ids, 0], -np.inf)
    upper = np.where(model.jnt_limited[ids], model.jnt_range[ids, 1], np.inf)
    scratch.qpos[q_indices] = np.clip(scratch.qpos[q_indices], lower, upper)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for iteration in range(max_iterations + 1):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        delta = target[:3, 3] - scratch.site_xpos[site]
        rotation = Rotation.from_matrix(target[:3, :3] @ scratch.site_xmat[site].reshape(3, 3).T).as_rotvec()
        pe, re = float(np.linalg.norm(delta)), float(np.linalg.norm(rotation))
        success = pe <= position_tolerance and re <= orientation_tolerance
        if success or iteration == max_iterations:
            return SiteIKResult(names, scratch.qpos[q_indices].copy(), success, pe, re, iteration)
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site)
        jac = np.vstack([jacp[:, dofs], 0.3 * jacr[:, dofs]])
        error = np.r_[delta, 0.3 * rotation]
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)
        step *= min(1.0, 0.15 / max(np.linalg.norm(step), 1e-12))
        scratch.qpos[q_indices] = np.clip(scratch.qpos[q_indices] + step, lower, upper)


def to_mujoco(targets: PlanningGrasps, *, stage: Stage = "grasp"):
    """Return (position[N,3], quaternion[N,4]) arrays; quaternion order is wxyz.

    Targets must be in the simulation world. Apply tool calibration if the
    controlled site/mocap body differs from the sampler TCP.
    """
    return targets.components(stage, quaternion_order="wxyz")


def set_mocap_target(
    model, data, body_name: str, targets: PlanningGrasps, *, index: int = 0, stage: Stage = "grasp"
):
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
