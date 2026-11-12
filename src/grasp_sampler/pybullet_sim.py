"""Optional PyBullet integration: collision-check and visualize grasps.

Spawns the object mesh and a simple parallel-jaw gripper proxy (two finger boxes
plus a palm bar -- no robot URDF needed) to:

  * `collision_filter` -- reject grasps whose gripper body penetrates the object;
  * `show_grasps` -- view the object with gripper proxies at each grasp in a GUI.

Requires ``pybullet`` (``pip install pybullet``). Grasp poses are TCP poses in
the object frame: +X closing, +Z approach, origin between the fingertips.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .mesh import load_mesh
from .types import Grasp

# Proxy finger/palm box thickness (m). Thin enough not to dominate collisions.
_BAR = 0.012


def _quat_from_matrix(R: np.ndarray):
    """Rotation matrix -> xyzw quaternion (PyBullet order)."""
    return Rotation.from_matrix(R).as_quat().tolist()


def _load_object(p, mesh_path: str):
    """Load in the sampler frame, then release the temporary OBJ after import."""
    mesh = load_mesh(mesh_path).mesh
    with tempfile.TemporaryDirectory(prefix="grasp-pybullet-") as directory:
        obj_file = str(Path(directory) / "object.obj")
        mesh.export(obj_file)
        col = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_file, rgbaColor=[0.8, 0.8, 0.85, 1.0])
    return p.createMultiBody(0, col, vis)


def _spawn_gripper(p, width: float, finger_len: float, *, body_only: bool = False):
    """Create one compound proxy with every component offset relative to TCP.

    Offsets belong to the collision shape, so resetting the body's TCP pose
    cannot move the base component to the fingertip center. Both collision
    entry points use this same proxy. Fingers are optional contact surfaces.
    """
    half = width / 2.0
    b = _BAR / 2.0
    # Keep the existing narrow body proxy behind the contacts, rather than
    # letting a wide palm bar clip the object's side faces.
    frames = [
        [0.0, 0.0, -finger_len - finger_len / 2.0],  # block behind fingertips
        [0.0, 0.0, -finger_len - 3.0 * finger_len / 2.0],  # stem toward the wrist
    ]
    if not body_only:
        frames += [[+half, 0.0, -finger_len / 2.0], [-half, 0.0, -finger_len / 2.0]]
    shape = p.createCollisionShapeArray(
        [p.GEOM_BOX] * len(frames),
        halfExtents=[[b, b, finger_len / 2]] * len(frames),
        collisionFramePositions=frames,
    )
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape)


def collision_filter(
    mesh_path: str,
    grasps: list[Grasp],
    *,
    clearance: float = 0.0,
    finger_len: float = 0.04,
    gui: bool = False,
) -> np.ndarray:
    """Boolean mask of grasps whose gripper body clears the object.

    A grasp is rejected if any gripper link comes within ``clearance`` of the
    object mesh (``clearance=0`` rejects only actual penetration).
    Every file format uses the same upright, centered frame as ``load_mesh``.
    The function owns an isolated client and closes it even when loading fails.
    """
    import pybullet
    from pybullet_utils.bullet_client import BulletClient

    if not np.isfinite(clearance) or clearance < 0 or not np.isfinite(finger_len) or finger_len <= 0:
        raise ValueError("clearance must be finite and nonnegative; finger_len must be finite and positive")
    p = BulletClient(connection_mode=pybullet.GUI if gui else pybullet.DIRECT)
    try:
        obj = _load_object(p, mesh_path)
        keep = np.zeros(len(grasps), bool)
        for i, g in enumerate(grasps):
            grip = _spawn_gripper(p, g.width, finger_len, body_only=True)
            pos = g.pose[:3, 3]
            quat = _quat_from_matrix(g.pose[:3, :3])
            p.resetBasePositionAndOrientation(grip, pos.tolist(), quat)
            p.performCollisionDetection()
            pts = p.getClosestPoints(obj, grip, distance=max(clearance, 1e-4))
            worst = min((c[8] for c in pts), default=1.0)  # signed distance
            keep[i] = worst >= -1e-6 if clearance == 0 else worst >= clearance
            p.removeBody(grip)
        return keep
    finally:
        p.disconnect()


def fixed_pose_collision_check(mesh, poses, opening_width, config):
    """Optional robust-backend callback: body clearance at object-local poses.

    Uses the supplied hypothesis vertices exactly (no file loading, centering,
    or convexification). This is the existing palm/stem proxy, NOT fingers,
    robot links, continuous collision detection, or dynamic closure/lift.
    Each client is isolated from other running PyBullet sessions.
    """
    import pybullet
    from pybullet_utils.bullet_client import BulletClient

    p = BulletClient(connection_mode=pybullet.DIRECT)
    try:
        shape = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=np.asarray(mesh.vertices).tolist(),
            indices=np.asarray(mesh.faces).ravel().tolist(),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        obj = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape)
        body = _spawn_gripper(p, opening_width, config.finger_len, body_only=True)
        for pose in poses:
            p.resetBasePositionAndOrientation(body, pose[:3, 3].tolist(), _quat_from_matrix(pose[:3, :3]))
            pts = p.getClosestPoints(obj, body, distance=max(config.spine_clearance, 1e-4))
            if any(c[8] < config.spine_clearance for c in pts):
                return False
        return True
    finally:
        p.disconnect()


def show_grasps(
    mesh_path: str,
    grasps: list[Grasp],
    *,
    max_grasps: int | None = None,
    finger_len: float = 0.04,
    hold: bool = True,
):
    """Open an isolated PyBullet GUI with the object and gripper proxies.

    A blocking viewer closes its client on exit. With ``hold=False``, return
    the live ``BulletClient``; the caller must call its ``disconnect()`` method.
    Loading failures always close the client.
    """
    import pybullet
    from pybullet_utils.bullet_client import BulletClient

    if max_grasps and len(grasps) > max_grasps:
        idx = np.linspace(0, len(grasps) - 1, max_grasps).astype(int)
        grasps = [grasps[i] for i in idx]

    p = BulletClient(connection_mode=pybullet.GUI)
    caller_owns_client = False
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        _load_object(p, mesh_path)
        for g in grasps:
            grip = _spawn_gripper(p, g.width, finger_len)
            p.resetBasePositionAndOrientation(grip, g.pose[:3, 3].tolist(), _quat_from_matrix(g.pose[:3, :3]))
        if not hold:
            caller_owns_client = True
            return p
        import time

        print("PyBullet GUI open; Ctrl-C to quit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if not caller_owns_client and p.isConnected():
            p.disconnect()
