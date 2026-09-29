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
from typing import Literal, NamedTuple, get_args

import numpy as np

from .antipodal import antipodal_grasps
from .robust import GeometricGraspBackend
from .types import Grasp, GraspConfig, ObjMesh
from .uncertainty import ObjectHypothesis, sample_hypotheses, validate_pose

Strategy = Literal["failure_guided", "nominal", "representative", "blind_pooling"]
_STRATEGIES = get_args(Strategy)
# Seed offset per proposal round, so each (round, source) pair draws its own samples.
_ROUND_SEED_STRIDE = {"failure_guided": 1009, "representative": 2003, "blind_pooling": 2003}


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
    strategy: Strategy = "failure_guided"   # the others exist for benchmarking
    require_nominal_valid: bool = False

    def __post_init__(self):
        if not isinstance(self.candidate_budget, int) or self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if not 0 <= self.nominal_fraction <= 1:
            raise ValueError("nominal_fraction must lie in [0,1]")
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"unknown strategy: {self.strategy!r}")
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


class _Bank(NamedTuple):
    hypotheses: list
    weights: np.ndarray            # normalized to sum to one


class _Score(NamedTuple):
    weighted: float                # weighted success probability R
    successes: int
    failures: dict                 # failure reason -> count
    passed: np.ndarray             # per-hypothesis bool


class _Ranked(NamedTuple):
    candidate: RobustCandidate
    score: _Score                  # on the search bank
    nominal_valid: bool


def _make_bank(values) -> _Bank:
    hypotheses = [v if isinstance(v, ObjectHypothesis) else ObjectHypothesis(*v) for v in values]
    if not hypotheses:
        raise ValueError("hypothesis bank cannot be empty")
    weights = np.asarray([h.weight for h in hypotheses], float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("hypothesis weights must be finite, nonnegative and nonzero")
    return _Bank(hypotheses, weights / weights.sum())


def _as_obj_mesh(mesh):
    bounds = mesh.bounding_box
    return ObjMesh(mesh, bounds.extents, bounds.primitive.transform, bounds.extents, "irregular")


def _source_candidate(grasp, source, nominal_pose, source_id):
    command = validate_pose(source.world_pose) @ validate_pose(grasp.pose)
    local = np.linalg.inv(validate_pose(nominal_pose)) @ command
    return RobustCandidate(Grasp(local, grasp.kind, grasp.width, dict(grasp.meta)), command, [source_id])


def _rot_distance(a, b):
    rel = a[:3, :3].T @ b[:3, :3]
    return float(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))


def _jaw_flip_distance(a, b):
    """Rotation distance between two commands, allowing the 180 deg jaw swap."""
    return min(_rot_distance(a, b), _rot_distance(a @ np.diag([-1, -1, 1, 1]), b))


def _dedup(candidates, cfg):
    """Merge candidates whose commands agree within the dedup tolerances (first wins)."""
    unique = []
    for c in candidates:
        for kept in unique:
            offset = np.linalg.norm(c.world_command[:3, 3] - kept.world_command[:3, 3])
            close = offset <= cfg.dedup_translation
            if close and _jaw_flip_distance(c.world_command, kept.world_command) <= cfg.dedup_angle:
                kept.duplicate_count += 1 + c.duplicate_count
                kept.sources = list(dict.fromkeys(kept.sources + c.sources))
                break
        else:
            unique.append(c)
    return unique


def _is_diverse(record, selected, cfg):
    a = record.candidate.world_command
    return all(np.linalg.norm(a[:3, 3] - q.candidate.world_command[:3, 3]) > cfg.diversity_translation
               or _rot_distance(a, q.candidate.world_command) > cfg.diversity_angle
               for q in selected)


def _select_diverse(accepted, requested_count, cfg):
    """Pick a spread-out subset of ``accepted``; pad with the rest if it is too small.

    Runs after, never before, the R acceptance constraint.
    """
    selected = []
    for record in accepted:
        if len(selected) >= requested_count:
            break
        if _is_diverse(record, selected, cfg):
            selected.append(record)
    for record in accepted:
        if len(selected) >= requested_count:
            break
        if all(record is not q for q in selected):
            selected.append(record)
    return selected


def _generate(source, nominal_pose, cap, samples, seed, source_id, grasp_config):
    if cap <= 0:
        return []
    cfg = replace(grasp_config, antipodal_max_grasps=cap, antipodal_samples=max(1, samples))
    return [_source_candidate(g, source, nominal_pose, source_id)
            for g in antipodal_grasps(_as_obj_mesh(source.mesh), cfg, np.random.default_rng(seed))]


def _score_candidate(candidate, bank, backend, opening, nominal_pose) -> _Score:
    """Evaluate the candidate's fixed world command on every hypothesis in ``bank``."""
    outcomes = [backend.evaluate(h.mesh, object_world_pose=h.world_pose,
                                 world_gripper_command=candidate.world_command,
                                 opening_width=opening, nominal_object_pose=nominal_pose)
                for h in bank.hypotheses]
    passed = np.asarray([o.valid for o in outcomes], bool)
    failures = dict(Counter(o.reason for o in outcomes if not o.valid))
    return _Score(float(np.dot(bank.weights, passed)), int(passed.sum()), failures, passed)


class _SearchScorer:
    """Scores candidates on the search bank once, tracking failure mass per hypothesis."""

    def __init__(self, bank, backend, opening, nominal_pose):
        self.bank, self.backend, self.opening, self.nominal_pose = bank, backend, opening, nominal_pose
        self.failure_mass = np.zeros(len(bank.hypotheses), float)
        # Values keep the candidate alive so its id() cannot be reused by another object.
        self._cache: dict[int, tuple[RobustCandidate, _Score]] = {}

    def score(self, candidate) -> _Score:
        hit = self._cache.get(id(candidate))
        if hit is None:
            score = _score_candidate(candidate, self.bank, self.backend, self.opening, self.nominal_pose)
            self.failure_mass += self.bank.weights * ~score.passed
            hit = self._cache[id(candidate)] = (candidate, score)
        return hit[1]

    @property
    def evaluations(self) -> int:
        return len(self._cache) * len(self.bank.hypotheses)


def _propose(cfg, round_id, scorer, remaining, nominal_pose, seed, grasp_config, usage):
    """Generate candidates from search-bank hypotheses chosen by ``cfg.strategy``."""
    bank = scorer.bank
    if cfg.strategy == "failure_guided":
        order = np.argsort(-scorer.failure_mass, kind="stable")[:cfg.guided_hypotheses_per_round]
        label = "failure_hypothesis"
    elif cfg.strategy == "blind_pooling":
        order, label = np.arange(len(bank.hypotheses)), "source_hypothesis"
    else:  # representative: the highest-weight hypotheses
        order, label = np.argsort(-bank.weights)[:cfg.guided_hypotheses_per_round], "source_hypothesis"
    cap_each = max(1, remaining // max(1, len(order)))
    stride = _ROUND_SEED_STRIDE[cfg.strategy]
    proposals = []
    for j in map(int, order):
        source_id = f"{label}:{j}"
        proposals += _generate(bank.hypotheses[j], nominal_pose, cap_each, cap_each * 8,
                               seed + stride * round_id + j, source_id, grasp_config)
        if cfg.strategy == "failure_guided":
            usage[source_id] += 1
    return proposals


def _collect_raw_candidates(mesh, nominal_pose, scorer, cfg, grasp_config, seed):
    """Nominal proposals first, then strategy-driven augmentation rounds."""
    nominal_cap = int(round(cfg.candidate_budget * cfg.nominal_fraction))
    nominal = ObjectHypothesis(mesh, nominal_pose)
    raw = _generate(nominal, nominal_pose, nominal_cap, max(1, nominal_cap * 8), seed, "nominal",
                    grasp_config)
    usage = Counter(["nominal"] if nominal_cap else [])
    # Score nominal proposals before choosing guided sources; otherwise
    # augmentation would be blind pooling in disguise.
    for candidate in raw:
        scorer.score(candidate)

    def propose(round_id, remaining):
        return _propose(cfg, round_id, scorer, remaining, nominal_pose, seed, grasp_config, usage)

    if cfg.strategy in ("representative", "blind_pooling") and not raw:
        raw += propose(0, cfg.candidate_budget)       # no nominal proposals: seed from the bank
    if cfg.strategy != "nominal":
        for round_id in range(1, cfg.augmentation_rounds + 1):
            remaining = cfg.candidate_budget - len(raw)
            if remaining <= 0:
                break
            raw += propose(round_id, remaining)
    return raw[:cfg.candidate_budget], usage


def _rank_on_search_bank(unique, scorer, mesh, cfg):
    ranked = [_Ranked(c, scorer.score(c), scorer.backend.evaluate(
                  mesh, object_world_pose=scorer.nominal_pose, world_gripper_command=c.world_command,
                  opening_width=scorer.opening, nominal_object_pose=scorer.nominal_pose).valid)
              for c in unique]
    ranked.sort(key=lambda r: r.score.weighted, reverse=True)
    if cfg.require_nominal_valid:
        ranked = [r for r in ranked if r.nominal_valid]
    if cfg.max_validation_candidates is not None:
        ranked = ranked[:cfg.max_validation_candidates]
    return ranked


def _resolve_banks(mesh, nominal_pose, search_hypotheses, validation_hypotheses, hypothesis_count,
                   validation_count, pose_uncertainty, geometry_uncertainty, seed):
    if search_hypotheses is None:
        streams = np.random.SeedSequence(seed).spawn(2)
        def sample(count, stream):
            return sample_hypotheses(mesh, nominal_pose, count, pose_uncertainty=pose_uncertainty,
                                     geometry_uncertainty=geometry_uncertainty,
                                     rng=np.random.default_rng(stream))
        search_hypotheses = sample(hypothesis_count, streams[0])
        validation_hypotheses = sample(validation_count or hypothesis_count, streams[1])
    elif validation_hypotheses is None:
        raise ValueError("validation_hypotheses is required when search_hypotheses is supplied")
    return _make_bank(search_hypotheses), _make_bank(validation_hypotheses)


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
    search, validation = _resolve_banks(mesh, nominal_pose, search_hypotheses, validation_hypotheses,
        hypothesis_count, validation_count, pose_uncertainty, geometry_uncertainty, seed)
    backend = backend or GeometricGraspBackend(gcfg)
    opening = gcfg.gripper_max_width if opening_width is None else opening_width
    scorer = _SearchScorer(search, backend, opening, nominal_pose)

    raw, usage = _collect_raw_candidates(mesh, nominal_pose, scorer, cfg, gcfg, seed)
    unique = _dedup(raw, cfg)
    ranked = _rank_on_search_bank(unique, scorer, mesh, cfg)

    # No candidate is accepted on search R; only the independent validation bank decides.
    records = []
    for r in ranked:
        v = _score_candidate(r.candidate, validation, backend, opening, nominal_pose)
        records.append(RobustCandidateRecord(r.candidate, r.score.weighted, v.weighted,
            r.score.successes, v.successes, r.score.failures, v.failures,
            v.weighted >= robustness_threshold, r.nominal_valid))
    accepted = [r for r in records if r.accepted]
    selected = _select_diverse(accepted, requested_count, cfg)

    diagnostics = {"strategy": cfg.strategy, "requested_count": requested_count,
        "returned_count": len(selected), "robustness_threshold": robustness_threshold,
        "raw_generated": len(raw), "unique_candidates": len(unique),
        "search_candidates": len(ranked), "search_hypotheses": len(search.hypotheses),
        "validation_hypotheses": len(validation.hypotheses),
        "search_evaluations": scorer.evaluations,
        "validation_evaluations": len(records) * len(validation.hypotheses),
        "robust_candidates": len(accepted), "source_usage": dict(usage),
        "independent_validation_bank": True,
        "failure_guided": cfg.strategy == "failure_guided"}
    return RobustGraspSetResult(selected, records, diagnostics)
