"""Object-state hypotheses only: metres, radians, and explicit local geometry.

No robot execution uncertainty is represented. Empirical/correlated perception
runs can supply ObjectHypothesis directly, bypassing the independent sampler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def validate_pose(value):
    pose = np.asarray(value, float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1)):
        raise ValueError("pose must be a finite 4x4 SE(3) transform")
    return pose.copy()


def _vector(value, name):
    a = np.asarray(value, float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite XYZ vector")
    return a


def _draw(rng, std, bounds, baseline, name):
    if std is not None and bounds is not None:
        raise ValueError(f"choose std or bounds for {name}, not both")
    if std is not None:
        std = _vector(std, name)
        if (std < 0).any():
            raise ValueError(f"{name} std must be nonnegative")
        return rng.normal(baseline, std)
    if bounds is not None:
        bounds = np.asarray(bounds, float)
        if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
                or (bounds[:, 0] > bounds[:, 1]).any()):
            raise ValueError(f"{name} bounds must be three [lower, upper] pairs")
        return rng.uniform(bounds[:, 0], bounds[:, 1])
    return np.full(3, baseline, float)


@dataclass
class PoseUncertainty:
    """Independent Gaussian or bounded translation and RPY deltas.

    Bounds have shape (3, 2). RPY means D = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    world: R_i = D @ R_est, t_i = t_est + dt (rotate about object origin).
    object: R_i = R_est @ D, t_i = t_est + R_est @ dt.
    Thus world rotations never orbit the object around the world origin.
    """
    translation_std_xyz: object = None
    rotation_std_rpy: object = None
    translation_bounds_xyz: object = None
    rotation_bounds_rpy: object = None
    frame: str = "world"

    def sample(self, estimated_pose, rng):
        pose = validate_pose(estimated_pose)
        dt = _draw(rng, self.translation_std_xyz, self.translation_bounds_xyz, 0, "translation")
        rpy = _draw(rng, self.rotation_std_rpy, self.rotation_bounds_rpy, 0, "rotation")
        delta = Rotation.from_euler("xyz", rpy).as_matrix()
        if self.frame == "world":
            pose[:3, :3] = delta @ pose[:3, :3]
            pose[:3, 3] += dt
        elif self.frame == "object":
            pose[:3, 3] += pose[:3, :3] @ dt
            pose[:3, :3] = pose[:3, :3] @ delta
        else:
            raise ValueError("frame must be 'world' or 'object'")
        return pose


@dataclass
class GeometryUncertainty:
    """Anisotropic scale about an explicit mesh-local pivot, default origin.

    Scale bounds are three [lower, upper] pairs (absolute scale factors).
    Gaussian scales are centered at one; nonpositive draws raise an error.
    No recentering or change to the mesh coordinate frame is performed.
    """
    scale_bounds_xyz: object = None
    scale_std_xyz: object = None
    pivot: object = (0.0, 0.0, 0.0)

    def sample(self, mesh, rng):
        scale = _draw(rng, self.scale_std_xyz, self.scale_bounds_xyz, 1, "scale")
        if (scale <= 0).any():
            raise ValueError("scale factors must be positive")
        pivot = _vector(self.pivot, "pivot")
        result = mesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = np.diag(scale)
        transform[:3, 3] = pivot - scale * pivot
        result.apply_transform(transform)
        return result, {"scale_xyz": scale.tolist(), "scale_pivot": pivot.tolist()}


@dataclass
class ObjectHypothesis:
    mesh: trimesh.Trimesh
    world_pose: np.ndarray
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.world_pose = validate_pose(self.world_pose)
        if not isinstance(self.mesh, trimesh.Trimesh) or self.mesh.is_empty:
            raise ValueError("hypothesis needs a nonempty Trimesh")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and nonnegative")


def sample_hypotheses(nominal_mesh, nominal_object_pose, count, *,
                      pose_uncertainty=None, geometry_uncertainty=None,
                      seed=None, rng=None):
    """Sample independent mesh/pose hypotheses with equal weights.

    Supply either an explicit Generator or a seed. Explicit hypotheses can have
    correlated geometry/poses and arbitrary nonnegative weights instead.
    """
    if not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("count must be a positive integer")
    if rng is not None and seed is not None:
        raise ValueError("supply rng or seed, not both")
    rng = rng if rng is not None else np.random.default_rng(seed)
    pose_spec = pose_uncertainty or PoseUncertainty()
    geometry_spec = geometry_uncertainty or GeometryUncertainty()
    hypotheses = []
    for i in range(count):
        pose = pose_spec.sample(nominal_object_pose, rng)
        mesh, metadata = geometry_spec.sample(nominal_mesh, rng)
        hypotheses.append(ObjectHypothesis(mesh, pose, metadata={
            **metadata, "sample_index": i, "pose_frame": pose_spec.frame}))
    return hypotheses
