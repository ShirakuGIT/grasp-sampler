"""Visualize generated grasps as gripper wireframes over the object mesh.

Uses trimesh's built-in viewer (pyglet). No simulator or robot model required.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .types import Grasp, ObjMesh

# Color per generator, so you can see at a glance which method produced what.
_KIND_COLORS = {
    "top": (60, 160, 230),
    "side": (60, 200, 140),
    "obb_face": (235, 175, 40),
    "antipodal": (225, 70, 70),
}
_DEFAULT_COLOR = (180, 180, 180)


def gripper_glyph(pose: np.ndarray, width: float, *, finger_len: float = 0.04,
                  stem: float = 0.06, color=_DEFAULT_COLOR) -> trimesh.path.Path3D:
    """Build a parallel-jaw gripper wireframe at a TCP ``pose``.

    The TCP frame is +X closing, +Z approach, origin between the fingertips.
    """
    half = width / 2.0
    tip_l = np.array([half, 0, 0.0])             # fingertips straddle the object
    tip_r = np.array([-half, 0, 0.0])
    base_l = tip_l - np.array([0, 0, finger_len])  # fingers run back along -Z
    base_r = tip_r - np.array([0, 0, finger_len])
    palm = 0.5 * (base_l + base_r)
    wrist = palm - np.array([0, 0, stem])

    verts = np.array([tip_l, base_l, tip_r, base_r, palm, wrist])
    edges = [[0, 1], [2, 3], [1, 3], [4, 5]]      # 2 fingers, palm bar, stem
    verts = trimesh.transform_points(verts, pose)

    path = trimesh.load_path(verts[np.asarray(edges)])
    path.colors = np.tile(np.array([*color, 255], np.uint8), (len(path.entities), 1))
    return path


def show_grasps(obj: ObjMesh, grasps: list[Grasp], *, max_grasps: int | None = None,
                finger_len: float = 0.04):
    """Open an interactive window with the mesh and gripper glyphs.

    Pass ``max_grasps`` to subsample evenly when there are many grasps.
    """
    scene = trimesh.Scene(build_scene(obj, grasps, max_grasps=max_grasps,
                                      finger_len=finger_len))
    scene.show()


def build_scene(obj: ObjMesh, grasps: list[Grasp], *, max_grasps: int | None = None,
                finger_len: float = 0.04) -> list:
    """Return the mesh + glyph geometries (handy for headless rendering)."""
    mesh = obj.mesh.copy()
    mesh.visual.face_colors = (200, 200, 210, 255)

    if max_grasps and len(grasps) > max_grasps:
        idx = np.linspace(0, len(grasps) - 1, max_grasps).astype(int)
        grasps = [grasps[i] for i in idx]

    geoms = [mesh]
    for g in grasps:
        color = _KIND_COLORS.get(g.kind, _DEFAULT_COLOR)
        geoms.append(gripper_glyph(g.pose, g.width, finger_len=finger_len, color=color))
    return geoms
