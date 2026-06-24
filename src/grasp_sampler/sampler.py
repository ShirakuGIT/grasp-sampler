"""The high-level ``GraspSampler`` facade."""
from __future__ import annotations

import numpy as np

from .antipodal import antipodal_grasps
from .mesh import load_mesh
from .obb import obb_face_grasps
from .primitives import side_grasps, top_grasps
from .types import Grasp, GraspConfig, ObjMesh

_METHODS = ("primitives", "obb_face", "antipodal")


class GraspSampler:
    """Generate parallel-jaw grasps for a mesh.

    Example
    -------
    >>> sampler = GraspSampler()
    >>> grasps = sampler.sample("mug.glb")
    >>> len(grasps)
    87
    """

    def __init__(self, config: GraspConfig | None = None):
        self.config = config or GraspConfig()

    def load(self, path, **kwargs) -> ObjMesh:
        """Load and normalize a mesh file (see :func:`grasp_sampler.mesh.load_mesh`)."""
        return load_mesh(path, **kwargs)

    def generate(self, obj: ObjMesh, *, methods=_METHODS, seed: int = 0) -> list[Grasp]:
        """Run the requested generators on an already-loaded mesh."""
        cfg = self.config
        grasps: list[Grasp] = []
        if "primitives" in methods:
            ex = obj.extents
            for pose, meta in top_grasps(ex, max_width=cfg.gripper_max_width,
                                         yaw_steps=cfg.primitive_yaw_steps):
                grasps.append(Grasp(pose, meta["type"], meta["span"], meta))
            for pose, meta in side_grasps(ex, max_width=cfg.gripper_max_width,
                                          roll_steps=cfg.primitive_roll_steps):
                grasps.append(Grasp(pose, meta["type"], meta["span"], meta))
        if "obb_face" in methods:
            grasps += obb_face_grasps(obj, cfg)
        if "antipodal" in methods:
            grasps += antipodal_grasps(obj, cfg, np.random.default_rng(seed))
        return grasps

    def sample(self, path, *, methods=_METHODS, seed: int = 0,
               load_kwargs: dict | None = None) -> list[Grasp]:
        """Load a mesh and generate grasps in one call."""
        obj = self.load(path, **(load_kwargs or {}))
        return self.generate(obj, methods=methods, seed=seed)
