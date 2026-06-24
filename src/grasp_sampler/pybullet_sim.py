"""Optional PyBullet integration: collision-check and visualize grasps.

Spawns the object mesh and a simple parallel-jaw gripper proxy (two finger boxes
plus a palm bar -- no robot URDF needed) to:

  * `collision_filter` -- reject grasps whose gripper body penetrates the object;
  * `show_grasps` -- view the object with gripper proxies at each grasp in a GUI.

Requires ``pybullet`` (``pip install pybullet``). Grasp poses are TCP poses in
the object frame: +X closing, +Z approach, origin between the fingertips.
"""
from __future__ import annotations

import numpy as np

from .types import Grasp

# Proxy finger/palm box thickness (m). Thin enough not to dominate collisions.
_BAR = 0.012


def _quat_from_matrix(R: np.ndarray):
    """Rotation matrix -> xyzw quaternion (PyBullet order)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _as_obj(mesh_path: str) -> str:
    """PyBullet's GEOM_MESH reads OBJ/STL, not GLB. Convert (and center, matching
    the sampler) to a temp OBJ when needed; pass OBJ/STL straight through."""
    path = str(mesh_path)
    if path.lower().endswith((".obj", ".stl")):
        return path
    import tempfile
    from .mesh import load_mesh
    obj = load_mesh(path)                       # upright + centered, like sampling
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    obj.mesh.export(tmp.name)
    return tmp.name


def _load_object(p, mesh_path: str):
    """Spawn the mesh as a static (concave) collision body at the origin."""
    obj_file = _as_obj(mesh_path)
    col = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file,
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_file,
                              rgbaColor=[0.8, 0.8, 0.85, 1.0])
    return p.createMultiBody(0, col, vis)


def _gripper_links(p, width: float, finger_len: float, *, body_only: bool):
    """Collision shapes for a parallel-jaw proxy in the TCP frame.

    Returns (shape_ids, local_frames): a palm bar, a stem, and (unless
    ``body_only``) two fingers. The fingers straddle the object and are meant to
    touch it; the palm + stem are the body that must stay clear. ``body_only``
    drops the fingers so a collision check tests only that clearance.
    """
    half = width / 2.0
    b = _BAR / 2.0
    # Palm + stem modeled as a single block behind the fingertips: a wide bar would
    # clip the object's side faces (the TCP sits between the two contacts), so we
    # probe only the volume directly behind the grasp, where the wrist actually is.
    palm = p.createCollisionShape(p.GEOM_BOX, halfExtents=[b, b, finger_len / 2.0])
    stem = p.createCollisionShape(p.GEOM_BOX, halfExtents=[b, b, finger_len / 2.0])
    shapes = [palm, stem]
    frames = [
        [0.0, 0.0, -finger_len - finger_len / 2.0],          # block behind fingertips
        [0.0, 0.0, -finger_len - 3.0 * finger_len / 2.0],    # stem toward the wrist
    ]
    if not body_only:
        finger = p.createCollisionShape(p.GEOM_BOX, halfExtents=[b, b, finger_len / 2.0])
        shapes += [finger, finger]
        frames += [[+half, 0.0, -finger_len / 2.0], [-half, 0.0, -finger_len / 2.0]]
    return shapes, frames


def _spawn_gripper(p, width: float, finger_len: float, *, body_only: bool = False):
    shapes, frames = _gripper_links(p, width, finger_len, body_only=body_only)
    n = len(shapes) - 1                                  # links beyond the base
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=shapes[0],
        basePosition=frames[0],
        linkMasses=[0] * n,
        linkCollisionShapeIndices=shapes[1:],
        linkVisualShapeIndices=[-1] * n,
        linkPositions=frames[1:],
        linkOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=[0] * n,
        linkJointTypes=[p.JOINT_FIXED] * n,
        linkJointAxis=[[0, 0, 1]] * n,
        linkInertialFramePositions=[[0, 0, 0]] * n,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
    )


def collision_filter(mesh_path: str, grasps: list[Grasp], *, clearance: float = 0.0,
                     finger_len: float = 0.04, gui: bool = False) -> np.ndarray:
    """Boolean mask of grasps whose gripper body clears the object.

    A grasp is rejected if any gripper link comes within ``clearance`` of the
    object mesh (``clearance=0`` rejects only actual penetration).
    """
    import pybullet as p

    cid = p.connect(p.GUI if gui else p.DIRECT)
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
            worst = min((c[8] for c in pts), default=1.0)    # signed distance
            keep[i] = worst >= -1e-6 if clearance == 0 else worst >= clearance
            p.removeBody(grip)
        return keep
    finally:
        p.disconnect(cid)


def show_grasps(mesh_path: str, grasps: list[Grasp], *, max_grasps: int | None = None,
                finger_len: float = 0.04, hold: bool = True) -> None:
    """Open a PyBullet GUI with the object and a gripper proxy at each grasp."""
    import pybullet as p

    if max_grasps and len(grasps) > max_grasps:
        idx = np.linspace(0, len(grasps) - 1, max_grasps).astype(int)
        grasps = [grasps[i] for i in idx]

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    _load_object(p, mesh_path)
    for g in grasps:
        grip = _spawn_gripper(p, g.width, finger_len)
        p.resetBasePositionAndOrientation(
            grip, g.pose[:3, 3].tolist(), _quat_from_matrix(g.pose[:3, :3]))
    if hold:
        import time
        print("PyBullet GUI open; Ctrl-C to quit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
