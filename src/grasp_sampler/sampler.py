"""The high-level ``GraspSampler`` facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from .antipodal import antipodal_grasps
from .mesh import load_mesh
from .obb import obb_face_grasps
from .primitives import side_grasps, top_grasps
from .types import Grasp, GraspConfig, ObjMesh


class GraspGenerator(Protocol):
    """A sampling strategy operating in the loaded mesh's local TCP frame.

    Implementations must not mutate ``obj`` or ``config``. Use the supplied
    random generator for reproducible draws; return a list of ``Grasp`` objects.
    """

    def __call__(self, obj: ObjMesh, config: GraspConfig, rng: np.random.Generator, /) -> list[Grasp]: ...


def _primitive_grasps(obj: ObjMesh, config: GraspConfig, _rng: np.random.Generator) -> list[Grasp]:
    top = top_grasps(obj.extents, max_width=config.gripper_max_width, yaw_steps=config.primitive_yaw_steps)
    side = side_grasps(
        obj.extents, max_width=config.gripper_max_width, roll_steps=config.primitive_roll_steps
    )
    center = obj.mesh.bounding_box.centroid
    grasps = []
    for pose, meta in [*top, *side]:
        pose[:3, 3] += center
        grasps.append(Grasp(pose, meta["type"], meta["span"], meta))
    return grasps


def _obb_grasps(obj: ObjMesh, config: GraspConfig, _rng: np.random.Generator) -> list[Grasp]:
    return obb_face_grasps(obj, config)


_GENERATORS: Mapping[str, GraspGenerator] = {
    "primitives": _primitive_grasps,
    "obb_face": _obb_grasps,
    "antipodal": antipodal_grasps,
}


class GraspSampler:
    """Generate parallel-jaw grasps for a mesh.

    ``generators`` adds or replaces named strategies for this instance only.
    All registered generators run by default, in registration order. ``methods``
    selects a subset in that same order, preserving built-in seeded behavior.

    Example
    -------
    >>> sampler = GraspSampler()
    >>> grasps = sampler.sample("mug.glb")
    """

    def __init__(
        self, config: GraspConfig | None = None, *, generators: Mapping[str, GraspGenerator] | None = None
    ):
        self.config = GraspConfig() if config is None else config
        self._generators = dict(_GENERATORS)
        if generators is not None:
            for name, generator in generators.items():
                if not isinstance(name, str) or not name.strip() or not callable(generator):
                    raise ValueError("generators must map nonempty names to callable strategies")
                self._generators[name] = generator

    @property
    def available_methods(self) -> tuple[str, ...]:
        """Registered generator names, in their deterministic execution order."""
        return tuple(self._generators)

    def load(self, path, **kwargs) -> ObjMesh:
        """Load and normalize a mesh file (see :func:`grasp_sampler.mesh.load_mesh`)."""
        return load_mesh(path, **kwargs)

    def generate(self, obj: ObjMesh, *, methods=None, seed: int = 0) -> list[Grasp]:
        """Run registered strategies on an already-loaded mesh.

        ``methods`` is a name or iterable of names. An empty iterable returns no
        candidates. Unknown names raise ``ValueError`` instead of silently
        dropping a requested method. No input mesh or configuration is modified.
        """
        requested = set(
            self.available_methods if methods is None else (methods,) if isinstance(methods, str) else methods
        )
        unknown = requested.difference(self._generators)
        if unknown:
            raise ValueError(
                f"unknown grasp methods: {sorted(unknown, key=str)!r}; "
                f"available methods: {self.available_methods!r}"
            )
        rng = np.random.default_rng(seed)
        grasps: list[Grasp] = []
        for name, generator in self._generators.items():
            if name in requested:
                grasps.extend(generator(obj, self.config, rng))
        return grasps

    def sample(self, path, *, methods=None, seed: int = 0, load_kwargs: dict | None = None) -> list[Grasp]:
        """Load a mesh and generate grasps in one call."""
        obj = self.load(path, **(load_kwargs or {}))
        return self.generate(obj, methods=methods, seed=seed)
