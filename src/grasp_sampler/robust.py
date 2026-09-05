"""Evaluate fixed robot commands against uncertain object states.

The command is computed ONCE: T_W_G_cmd = T_W_O_est @ T_O_G.
Every hypothesis uses inv(T_W_O_i) @ T_W_G_cmd. Recomputing T_W_O_i @ T_O_G
would make the robot follow the unknown true object and invalidate the model.

V1 uses point contacts, outward mesh normals, symmetric closing along TCP X,
and sampled palm/stem clearance along -TCP Z to pregrasp. This is a geometric
screen, not a force/dynamic lift test or a continuous swept-volume guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol
import numpy as np
import trimesh

from .antipodal import _body_clears, _ray_cast_proxy, friction_cones_valid
from .types import GraspConfig
from .uncertainty import validate_pose


@dataclass
class FixedGraspResult:
    valid: bool
    reason: str
    contact_width: float | None = None
    contacts: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class ValidityBackend(Protocol):
    """Future dynamic backends receive the same immutable-in-meaning command.

    A backend must never relocate the commanded TCP to fit the hypothesis.
    """
    def evaluate(self, mesh, *, object_world_pose, world_gripper_command,
                 opening_width) -> FixedGraspResult: ...


class GeometricGraspBackend:
    def __init__(self, config=None, *, geometry_policy="mesh",
                 contact_tolerance=0.002, approach_distance=0.05,
                 approach_steps=6, check_body=True, collision_checker=None):
        self.config = replace(config or GraspConfig(),
                              contact_geometry_policy=geometry_policy)
        if geometry_policy not in ("mesh", "convex_hull", "existing_auto_policy"):
            raise ValueError("unknown geometry policy")
        if (not np.isfinite(contact_tolerance) or contact_tolerance < 0
                or not np.isfinite(approach_distance) or approach_distance < 0
                or not isinstance(approach_steps, int) or approach_steps < 2):
            raise ValueError("invalid contact tolerance or approach sampling")
        self.contact_tolerance = contact_tolerance
        self.approach_distance = approach_distance
        self.approach_steps = approach_steps
        self.check_body = check_body
        # Optional callable(mesh, object-local poses, opening_width, config).
        self.collision_checker = collision_checker

    def evaluate(self, mesh, *, object_world_pose, world_gripper_command, opening_width):
        pose = np.linalg.inv(validate_pose(object_world_pose)) @ validate_pose(world_gripper_command)
        cfg = self.config
        proxy = _ray_cast_proxy(mesh, cfg)
        metadata = {"contact_geometry_policy": cfg.contact_geometry_policy,
                    "contact_geometry": "mesh" if proxy is mesh else "convex_hull",
                    "collision_geometry": "mesh", "mesh_watertight": bool(mesh.is_watertight),
                    "body_checked": self.check_body, "dynamic_validation": False,
                    "opening_width": opening_width,
                    "approach_steps": self.approach_steps}
        def result(valid, reason, width=None, contacts=None):
            return FixedGraspResult(valid, reason, width, contacts, metadata.copy())
        if not np.isfinite(opening_width) or not cfg.gripper_min_width <= opening_width <= cfg.gripper_max_width:
            return result(False, "opening_width")
        center, closing, approach = pose[:3, 3], pose[:3, 0], pose[:3, 2]
        half = opening_width / 2
        # Cast inward from both OPEN finger positions. Take the first hit each
        # finger would encounter; never skip a wall or sample replacement poses.
        origins = np.array([center - half * closing, center + half * closing])
        directions = np.array([closing, -closing])
        locations, ray_ids, faces = proxy.ray.intersects_location(origins, directions, multiple_hits=True)
        contacts, normals = [], []
        for i in range(2):
            idx = np.flatnonzero(ray_ids == i)
            if not len(idx):
                return result(False, "missing_contacts")
            distances = (locations[idx] - origins[i]) @ directions[i]
            idx = idx[(distances >= -1e-9) & (distances <= opening_width + 1e-9)]
            if not len(idx):
                return result(False, "contacts_outside_opening")
            j = idx[np.argmin((locations[idx] - origins[i]) @ directions[i])]
            contacts.append(locations[j])
            normals.append(proxy.face_normals[faces[j]])
        contacts = np.asarray(contacts)
        offsets = (contacts - center) @ closing
        width = float(offsets[1] - offsets[0])
        metadata["contact_offsets_x"] = offsets.tolist()
        if (width < cfg.gripper_min_width or width > opening_width + 1e-9
                or offsets[0] > 0 or offsets[1] < 0):
            return result(False, "contact_width", width, contacts)
        if abs(offsets.sum()) > self.contact_tolerance:
            return result(False, "asymmetric_contacts", width, contacts)
        if not friction_cones_valid(*normals, closing, np.cos(np.arctan(cfg.friction_coef))):
            return result(False, "friction_cones", width, contacts)
        poses = np.repeat(pose[None], self.approach_steps, axis=0)
        poses[:, :3, 3] -= np.linspace(0, self.approach_distance, self.approach_steps)[:, None] * approach
        if self.check_body:
            pq = trimesh.proximity.ProximityQuery(mesh)
            # Check both fixed open width during approach and final contact width.
            if (not all(_body_clears(mesh, pq, p[:3, 3], closing, approach, half, cfg) for p in poses)
                    or not _body_clears(mesh, pq, center, closing, approach, width / 2, cfg)):
                return result(False, "body_or_approach_collision", width, contacts)
        if self.collision_checker is not None:
            metadata["additional_collision_checker"] = True
            if not self.collision_checker(mesh, poses, opening_width, cfg):
                return result(False, "backend_collision", width, contacts)
        return result(True, "valid", width, contacts)


@dataclass
class HypothesisGraspResult:
    success: bool
    weight: float
    world_gripper_command: np.ndarray
    object_gripper_command: np.ndarray
    validity: FixedGraspResult
    metadata: dict = field(default_factory=dict)


@dataclass
class RobustGraspResult:
    success_probability: float
    weighted_success_probability: float
    worst_case_valid: bool
    num_successes: int
    num_hypotheses: int
    nominal_valid: bool
    per_hypothesis_results: list[HypothesisGraspResult]
    nominal_result: FixedGraspResult
    world_gripper_command: np.ndarray
    metadata: dict = field(default_factory=dict)


class RobustGraspEvaluator:
    """Nominal mesh must share the frame used by Grasp.pose.

    Default deterministic command: open to configured maximum, approach along
    TCP +Z, close symmetrically. Grasp.width is the nominal required span, not
    an execution opening command. Override opening_width to model a narrower
    fixed opening. Contact tolerance approximates finite pad compliance only.
    """
    def __init__(self, nominal_mesh, config=None, *, backend=None, opening_width=None, **backend_options):
        self.nominal_mesh = nominal_mesh
        self.config = config or GraspConfig()
        self.backend = backend or GeometricGraspBackend(self.config, **backend_options)
        self.opening_width = self.config.gripper_max_width if opening_width is None else opening_width

    def evaluate(self, grasp, *, nominal_object_pose, hypotheses):
        nominal_pose = validate_pose(nominal_object_pose)
        command = nominal_pose @ validate_pose(grasp.pose)  # ONCE, outside loop
        hypotheses = list(hypotheses)
        weights = np.array([h.weight for h in hypotheses], float)
        if (not len(weights) or not np.isfinite(weights).all() or (weights < 0).any()
                or not np.isfinite(weights.sum()) or weights.sum() <= 0):
            raise ValueError("hypotheses need finite nonnegative weights with positive total")
        def check(mesh, pose):
            return self.backend.evaluate(mesh, object_world_pose=pose.copy(),
                world_gripper_command=command.copy(), opening_width=self.opening_width)
        nominal = check(self.nominal_mesh, nominal_pose)
        outcomes = []
        for h in hypotheses:
            pose = validate_pose(h.world_pose)
            relative = np.linalg.inv(pose) @ command
            validity = check(h.mesh, pose)
            outcomes.append(HypothesisGraspResult(validity.valid, h.weight,
                command.copy(), relative, validity, dict(h.metadata)))
        successes = np.array([r.success for r in outcomes], bool)
        return RobustGraspResult(float(successes.mean()),
            float(np.dot(weights / weights.sum(), successes)), bool(successes.all()),
            int(successes.sum()), len(outcomes), nominal.valid, outcomes, nominal,
            command.copy(), {"fixed_world_command": True, "opening_width": self.opening_width})


@dataclass
class RankedGrasp:
    grasp_id: int
    nominal_score: float | None
    result: RobustGraspResult


def rank_grasps(grasps, evaluator, *, nominal_object_pose, hypotheses,
                nominal_scores=None, criterion="expected_success"):
    """Preserve supplied nominal scores; repository generators have no score.

    Expected success normalizes weights; worst-case includes zero-weight
    hypotheses as well. Ties preserve candidate order.
    """
    if criterion not in ("expected_success", "worst_case_valid"):
        raise ValueError("unknown ranking criterion")
    grasps, hypotheses = list(grasps), list(hypotheses)
    scores = list(nominal_scores) if nominal_scores is not None else [None] * len(grasps)
    if len(scores) != len(grasps):
        raise ValueError("nominal_scores length must match grasps")
    rows = [RankedGrasp(i, scores[i], evaluator.evaluate(g,
        nominal_object_pose=nominal_object_pose, hypotheses=hypotheses)) for i, g in enumerate(grasps)]
    return sorted(rows, key=lambda r: (r.result.weighted_success_probability
        if criterion == "expected_success" else r.result.worst_case_valid), reverse=True)
