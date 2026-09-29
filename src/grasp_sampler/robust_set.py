"""Nominal-first robust grasp-set generation.

Candidates are generated on the nominal reconstruction first.  Search-bank
failures then guide a bounded second proposal phase on the hypotheses that
contribute most failure mass.  Every candidate is converted to one immutable
world command before cross-evaluation.  Final acceptance uses only the
independent validation-bank weighted success probability ``R``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import math
import numpy as np

from .antipodal import antipodal_grasps
from .robust import GeometricGraspBackend, RobustGraspEvaluator
from .types import Grasp, GraspConfig, ObjMesh
from .uncertainty import ObjectHypothesis, sample_hypotheses, validate_pose


@dataclass
class RobustSetConfig:
    candidate_budget: int = 1000
    nominal_fraction: float = .7
    augmentation_rounds: int = 2
    guided_hypotheses_per_round: int = 2
    max_validation_candidates: int | None = None
    dedup_translation: float = .001
    dedup_angle: float = np.deg2rad(3)
    jaw_symmetry: bool = True
    diversity_translation: float = .01
    diversity_angle: float = np.deg2rad(20)
    strategy: str = "failure_guided"  # failure_guided|nominal|representative|blind_pooling
    require_nominal_valid: bool = False

    def __post_init__(self):
        if not isinstance(self.candidate_budget, int) or self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if not 0 <= self.nominal_fraction <= 1:
            raise ValueError("nominal_fraction must lie in [0,1]")
        if self.strategy not in ("failure_guided", "nominal", "representative", "blind_pooling"):
            raise ValueError("unknown strategy")
        if self.augmentation_rounds < 0 or self.guided_hypotheses_per_round < 1:
            raise ValueError("invalid augmentation settings")


@dataclass
class RobustCandidate:
    grasp: Grasp
    world_command: np.ndarray
    sources: list[str] = field(default_factory=list)
    duplicate_count: int = 0


@dataclass
class RobustCandidateRecord:
    candidate: RobustCandidate
    search_R: float
    validation_R: float | None
    search_successes: int
    validation_successes: int | None
    search_failures: dict
    validation_failures: dict
    accepted: bool
    nominal_valid: bool


@dataclass
class RobustGraspSetResult:
    selected: list[RobustCandidateRecord]
    records: list[RobustCandidateRecord]
    diagnostics: dict

    @property
    def grasps(self):
        return [r.candidate.grasp for r in self.selected]

    @property
    def world_commands(self):
        return (np.stack([r.candidate.world_command for r in self.selected])
                if self.selected else np.empty((0, 4, 4)))


def _bank(values):
    hs = [v if isinstance(v, ObjectHypothesis) else ObjectHypothesis(*v) for v in values]
    if not hs:
        raise ValueError("hypothesis bank cannot be empty")
    w = np.asarray([h.weight for h in hs], float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("hypothesis weights must be finite, nonnegative and nonzero")
    return hs, w / w.sum()


def _obj(mesh):
    b = mesh.bounding_box
    return ObjMesh(mesh, b.extents, b.primitive.transform, b.extents, "irregular")


def _source_candidate(g, source, nominal_pose, source_id):
    command = validate_pose(source.world_pose) @ validate_pose(g.pose)
    local = np.linalg.inv(validate_pose(nominal_pose)) @ command
    return RobustCandidate(Grasp(local, g.kind, g.width, dict(g.meta)), command, [source_id])


def _rot_distance(a, b):
    rel = a[:3, :3].T @ b[:3, :3]
    return float(np.arccos(np.clip((np.trace(rel)-1)/2, -1, 1)))


def _dedup(candidates, cfg):
    out = []
    for c in candidates:
        found = None
        for i, old in enumerate(out):
            if (np.linalg.norm(c.world_command[:3,3]-old.world_command[:3,3]) <= cfg.dedup_translation
                    and min(_rot_distance(c.world_command, old.world_command),
                            _rot_distance(c.world_command @ np.diag([-1,-1,1,1]), old.world_command)) <= cfg.dedup_angle):
                found = i; break
        if found is None: out.append(c)
        else:
            out[found].duplicate_count += 1 + c.duplicate_count
            out[found].sources = list(dict.fromkeys(out[found].sources + c.sources))
    return out


def _generate(source, nominal_pose, cap, samples, seed, source_id, grasp_config):
    if cap <= 0: return []
    cfg = replace(grasp_config, antipodal_max_grasps=cap, antipodal_samples=max(1, samples))
    return [_source_candidate(g, source, nominal_pose, source_id)
            for g in antipodal_grasps(_obj(source.mesh), cfg, np.random.default_rng(seed))]


def _evaluate(candidate, mesh, pose, bank, weights, backend, opening):
    evaluator = RobustGraspEvaluator(mesh, backend=backend, opening_width=opening)
    # evaluator's result is deliberately used only for binary valid outcomes.
    result = evaluator.evaluate(candidate.grasp, nominal_object_pose=pose, hypotheses=bank)
    failures = Counter(r.validity.reason for r in result.per_hypothesis_results if not r.success)
    return result.weighted_success_probability, result.num_successes, dict(failures), result.nominal_valid


def generate_robust_grasp_set(mesh, estimated_pose, *, requested_count=20,
        robustness_threshold=.95, search_hypotheses=None, validation_hypotheses=None,
        hypothesis_count=100, validation_count=None, pose_uncertainty=None,
        geometry_uncertainty=None, config=None, grasp_config=None, backend=None,
        opening_width=None, seed=0):
    """Generate up to ``requested_count`` validated fixed-command grasps.

    If banks are sampled internally, independent random streams create search
    and validation banks. Explicit banks preserve correlated ``(mesh, pose)``
    pairs and weights. ``R`` is always weighted binary closure success.
    """
    cfg = config or RobustSetConfig()
    gcfg = grasp_config or GraspConfig(contact_geometry_policy="mesh")
    if requested_count < 1 or not 0 <= robustness_threshold <= 1:
        raise ValueError("invalid requested_count or threshold")
    nominal_pose = validate_pose(estimated_pose)
    if search_hypotheses is None:
        ss = np.random.SeedSequence(seed).spawn(2)
        search_hypotheses = sample_hypotheses(mesh, nominal_pose, hypothesis_count,
            pose_uncertainty=pose_uncertainty, geometry_uncertainty=geometry_uncertainty,
            rng=np.random.default_rng(ss[0]))
        validation_hypotheses = sample_hypotheses(mesh, nominal_pose,
            validation_count or hypothesis_count,
            pose_uncertainty=pose_uncertainty, geometry_uncertainty=geometry_uncertainty,
            rng=np.random.default_rng(ss[1]))
    elif validation_hypotheses is None:
        raise ValueError("validation_hypotheses is required when search_hypotheses is supplied")
    search, sw = _bank(search_hypotheses)
    validation, vw = _bank(validation_hypotheses)
    backend = backend or GeometricGraspBackend(gcfg)
    if hasattr(backend, "set_nominal_pose"):
        backend.set_nominal_pose(nominal_pose)
    opening = gcfg.gripper_max_width if opening_width is None else opening_width
    nominal = ObjectHypothesis(mesh, nominal_pose)
    raw = []
    ncap = int(round(cfg.candidate_budget * cfg.nominal_fraction))
    raw += _generate(nominal, nominal_pose, ncap, max(1, ncap * 8), seed, "nominal", gcfg)
    source_usage = Counter(["nominal"] if ncap else [])
    search_cache = {}
    failures_by_h = np.zeros(len(search), float)

    def assess(c, hs, ws):
        # Explicitly use the fixed world command stored on the candidate.
        ev = RobustGraspEvaluator(mesh, backend=backend, opening_width=opening)
        outcomes = []
        for h in hs:
            v = backend.evaluate(h.mesh, object_world_pose=h.world_pose,
                world_gripper_command=c.world_command, opening_width=opening)
            outcomes.append(v)
        ok = np.asarray([v.valid for v in outcomes], bool)
        return float(np.dot(ws, ok)), int(ok.sum()), dict(Counter(v.reason for v in outcomes if not v.valid)), ok

    def update(c):
        key = id(c)
        if key in search_cache: return search_cache[key]
        r = assess(c, search, sw)
        search_cache[key] = r
        failures_by_h[:] += sw * (~r[3])
        return r

    # The initial nominal proposals must be scored before choosing guided
    # sources; otherwise augmentation would be blind pooling in disguise.
    for c in list(raw):
        update(c)

    # Failure-guided rounds use only the currently observed search failures.
    # Start at one: nominal proposals were already generated and scored above.
    if cfg.strategy in ("representative", "blind_pooling") and not raw:
        order = np.arange(len(search)) if cfg.strategy == "blind_pooling" else np.argsort(-sw)[:cfg.guided_hypotheses_per_round]
        remaining = cfg.candidate_budget
        cap_each = max(1, remaining // max(1, len(order)))
        for j in order:
            raw += _generate(search[int(j)], nominal_pose, cap_each, cap_each * 8,
                seed + int(j), f"source_hypothesis:{int(j)}", gcfg)
    for round_id in range(1, cfg.augmentation_rounds + 1):
        # debug removed after verification
        if round_id > 0 and cfg.strategy == "failure_guided":
            order = np.argsort(-failures_by_h, kind="stable")[:cfg.guided_hypotheses_per_round]
            remaining = cfg.candidate_budget - len(raw)
            if remaining <= 0: break
            cap_each = max(1, remaining // max(1, len(order)))
            for j in order:
                raw += _generate(search[j], nominal_pose, cap_each, cap_each * 8,
                    seed + 1009 * round_id + int(j), f"failure_hypothesis:{int(j)}", gcfg)
                source_usage[f"failure_hypothesis:{int(j)}"] += 1
        elif round_id > 0 and cfg.strategy in ("blind_pooling", "representative"):
            # Blind modes are deliberately exposed only for benchmark comparison.
            order = np.arange(len(search)) if cfg.strategy == "blind_pooling" else np.argsort(-sw)[:cfg.guided_hypotheses_per_round]
            remaining = cfg.candidate_budget - len(raw)
            if remaining <= 0: break
            cap_each = max(1, remaining // max(1, len(order)))
            for j in order:
                raw += _generate(search[int(j)], nominal_pose, cap_each, cap_each * 8,
                    seed + 2003 * round_id + int(j), f"source_hypothesis:{int(j)}", gcfg)
        else:
            break
        # Force progress even when a sampler returns fewer than requested.
        if len(raw) >= cfg.candidate_budget: break
    raw = raw[:cfg.candidate_budget]
    unique = _dedup(raw, cfg)
    search_records = []
    for c in unique:
        r, n, reasons, ok = update(c)
        nominal_ok = backend.evaluate(mesh, object_world_pose=nominal_pose,
            world_gripper_command=c.world_command, opening_width=opening).valid
        search_records.append((c, r, n, reasons, nominal_ok))
    search_records.sort(key=lambda x: x[1], reverse=True)
    # Validate search survivors in independent bank; no candidate is accepted on search R.
    if cfg.require_nominal_valid:
        search_records = [x for x in search_records if x[4]]
    if cfg.max_validation_candidates is not None:
        search_records = search_records[:cfg.max_validation_candidates]
    records = []
    for c, sr, sn, sf, nv in search_records:
        vr, vn, vf, _ = assess(c, validation, vw)
        records.append(RobustCandidateRecord(c, sr, vr, sn, vn, sf, vf,
            vr >= robustness_threshold, nv))
    accepted = [r for r in records if r.accepted]
    # Diversity is applied after, never before, the R acceptance constraint.
    selected = []
    for r in accepted:
        if len(selected) >= requested_count: break
        if all(np.linalg.norm(r.candidate.world_command[:3,3]-q.candidate.world_command[:3,3]) > cfg.diversity_translation
               or _rot_distance(r.candidate.world_command,q.candidate.world_command) > cfg.diversity_angle for q in selected):
            selected.append(r)
    if len(selected) < requested_count:
        for r in accepted:
            if all(r is not q for q in selected):
                selected.append(r)
                if len(selected) == requested_count: break
    diagnostics = {"strategy": cfg.strategy, "requested_count": requested_count,
        "returned_count": len(selected), "robustness_threshold": robustness_threshold,
        "raw_generated": len(raw), "unique_candidates": len(unique),
        "search_candidates": len(search_records), "search_hypotheses": len(search),
        "validation_hypotheses": len(validation), "search_evaluations": len(unique)*len(search),
        "validation_evaluations": len(records)*len(validation),
        "robust_candidates": len(accepted), "source_usage": dict(source_usage),
        "independent_validation_bank": True,
        "failure_guided": cfg.strategy == "failure_guided"}
    return RobustGraspSetResult(selected, records, diagnostics)
