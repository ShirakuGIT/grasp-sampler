import numpy as np
import pytest
import trimesh

from grasp_sampler import (
    FixedGraspResult,
    Grasp,
    GraspConfig,
    ObjectHypothesis,
    RobustCandidate,
    RobustCandidateRecord,
    RobustSetConfig,
    generate_robust_grasp_set,
    sample_hypotheses,
)
from grasp_sampler import robust_set as robust_set_module


def _run(seed=5, **kwargs):
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    pose = np.eye(4)
    search = sample_hypotheses(mesh, pose, 12, seed=seed)
    validation = sample_hypotheses(mesh, pose, 16, seed=seed + 1)
    return generate_robust_grasp_set(
        mesh,
        pose,
        search_hypotheses=search,
        validation_hypotheses=validation,
        requested_count=3,
        config=RobustSetConfig(candidate_budget=30, **kwargs),
        grasp_config=GraspConfig(
            directed=False, antipodal_samples=100, antipodal_max_grasps=30, contact_geometry_policy="mesh"
        ),
        seed=seed,
    )


def test_only_validation_R_accepts_and_is_independent():
    result = _run()
    assert result.diagnostics["independent_validation_bank"]
    assert all(r.accepted and r.validation_R >= 0.95 for r in result.selected)
    assert all(r.search_R >= 0 for r in result.records)


def test_seed_reproducibility():
    a, b = _run(17), _run(17)
    np.testing.assert_allclose(a.world_commands, b.world_commands)
    assert [(r.search_R, r.validation_R) for r in a.records] == [
        (r.search_R, r.validation_R) for r in b.records
    ]


def test_explicit_bank_requires_validation_bank():
    mesh, pose = trimesh.creation.box(), np.eye(4)
    hs = [ObjectHypothesis(mesh, pose)]
    try:
        generate_robust_grasp_set(mesh, pose, search_hypotheses=hs)
    except ValueError as exc:
        assert "validation_hypotheses" in str(exc)
    else:
        raise AssertionError("independent validation bank was not required")


def test_threshold_is_sole_acceptance_metric():
    result = _run(nominal_fraction=1.0, augmentation_rounds=0)
    assert all(r.accepted == (r.validation_R >= 0.95) for r in result.records)


def test_strategies_are_exposed_for_benchmarking():
    for strategy, fraction in (("nominal", 1.0), ("representative", 0.0), ("blind_pooling", 0.0)):
        result = _run(strategy=strategy, nominal_fraction=fraction, augmentation_rounds=0)
        assert result.diagnostics["strategy"] == strategy


def test_failure_guided_calls_failing_hypothesis_source(monkeypatch):
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    nominal = np.eye(4)
    shifted = np.eye(4)
    shifted[0, 3] = 0.01
    banks = [ObjectHypothesis(mesh, nominal), ObjectHypothesis(mesh, shifted)]
    calls = []

    def fake_generate(source, nominal_pose, cap, samples, seed, source_id, grasp_config):
        calls.append(source_id)
        p = np.eye(4)
        p[:3, 3] = source.world_pose[:3, 3]
        return [robust_set_module.RobustCandidate(Grasp(p, "test", 0.04), source.world_pose @ p, [source_id])]

    monkeypatch.setattr(robust_set_module, "_generate", fake_generate)

    class PoseGate:
        def evaluate(
            self, mesh, *, object_world_pose, world_gripper_command, opening_width, nominal_object_pose=None
        ):
            ok = abs(object_world_pose[0, 3]) < 0.001
            return FixedGraspResult(ok, "valid" if ok else "pose_failure")

    result = robust_set_module.generate_robust_grasp_set(
        mesh,
        nominal,
        requested_count=1,
        robustness_threshold=0.5,
        search_hypotheses=banks,
        validation_hypotheses=banks,
        config=RobustSetConfig(
            candidate_budget=3, nominal_fraction=0.34, augmentation_rounds=1, guided_hypotheses_per_round=1
        ),
        grasp_config=GraspConfig(),
        backend=PoseGate(),
        seed=3,
    )
    assert calls[0] == "nominal"
    assert any(c.startswith("failure_hypothesis:") for c in calls[1:])
    assert result.diagnostics["failure_guided"]


def _candidate(x, yaw=0.0, sources=("nominal",)):
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    pose[:3, 3] = [x, 0, 0]
    return RobustCandidate(Grasp(pose, "test", 0.04), pose, list(sources))


def _record(candidate):
    return RobustCandidateRecord(candidate, 1.0, 1.0, 1, 1, {}, {}, True, True)


def test_dedup_merges_close_and_jaw_flipped_commands():
    a, close, flipped, far = (
        _candidate(0, sources=["a"]),
        _candidate(0.0005, sources=["b"]),
        _candidate(0, np.pi, ["c"]),
        _candidate(0.05),
    )
    unique = robust_set_module._dedup([a, close, flipped, far], RobustSetConfig())
    assert len(unique) == 2 and unique[0] is a and unique[1] is far
    assert a.duplicate_count == 2 and a.sources == ["a", "b", "c"]


def test_select_diverse_prefers_spread_then_pads_with_accepted():
    cfg = RobustSetConfig()  # 1 cm / 20 deg diversity
    near_dup = _record(_candidate(0.002))
    accepted = [_record(_candidate(0)), near_dup, _record(_candidate(0.05))]

    def pick(n):
        return robust_set_module._select_diverse(accepted, n, cfg)

    assert pick(1) == accepted[:1]
    assert pick(2) == [accepted[0], accepted[2]]  # skips the near-duplicate
    assert pick(3) == [accepted[0], accepted[2], near_dup]  # padded when too few are diverse


@pytest.mark.parametrize("symmetry, expected", [(True, 1), (False, 2)])
def test_jaw_symmetry_controls_dedup_and_diversity(symmetry, expected):
    a, flipped = _candidate(0), _candidate(0, np.pi)
    cfg = RobustSetConfig(jaw_symmetry=symmetry)
    assert len(robust_set_module._dedup([a, flipped], cfg)) == expected
    assert robust_set_module._is_diverse(_record(flipped), [_record(a)], cfg) is (not symmetry)


def test_internal_search_preserves_explicit_validation_bank():
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    supplied = ObjectHypothesis(mesh, np.eye(4))
    search, validation = robust_set_module._resolve_banks(
        mesh, np.eye(4), None, [supplied], 3, None, None, None, 1
    )
    assert len(search.hypotheses) == 3
    assert validation.hypotheses == [supplied]


def test_zero_validation_count_is_rejected_instead_of_using_search_count():
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    with pytest.raises(ValueError, match="positive integer"):
        robust_set_module._resolve_banks(mesh, np.eye(4), None, None, 3, 0, None, None, 1)


def test_each_guided_round_uses_failures_from_previous_round(monkeypatch):
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    shifted = np.eye(4)
    shifted[0, 3] = 1
    hypotheses = [ObjectHypothesis(mesh, np.eye(4), 0.25), ObjectHypothesis(mesh, shifted, 0.75)]
    calls = []

    def generate(source, nominal_pose, cap, samples, seed, source_id, grasp_config):
        candidate = _candidate(float(len(calls)), sources=[source_id])
        calls.append(source_id)
        return [candidate]

    class AlternatingGate:
        def evaluate(self, mesh, *, object_world_pose, world_gripper_command, **kwargs):
            valid = (world_gripper_command[0, 3] < 0.5) == (object_world_pose[0, 3] > 0.5)
            return FixedGraspResult(valid, "valid" if valid else "pose_failure")

    monkeypatch.setattr(robust_set_module, "_generate", generate)
    generate_robust_grasp_set(
        mesh,
        np.eye(4),
        search_hypotheses=hypotheses,
        validation_hypotheses=hypotheses,
        backend=AlternatingGate(),
        config=RobustSetConfig(
            candidate_budget=3, nominal_fraction=1 / 3, augmentation_rounds=2, guided_hypotheses_per_round=1
        ),
    )
    assert calls == ["nominal", "failure_hypothesis:0", "failure_hypothesis:1"]


def test_custom_backend_cannot_mutate_stored_commands_or_hypothesis_poses(monkeypatch):
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    hypothesis, candidate = ObjectHypothesis(mesh, np.eye(4)), _candidate(0)

    class MutatingGate:
        def __bool__(self):
            return False  # A valid strategy must not be replaced because it is falsey.

        def evaluate(self, mesh, *, object_world_pose, world_gripper_command, nominal_object_pose, **kwargs):
            object_world_pose[:] = 9
            world_gripper_command[:] = 9
            nominal_object_pose[:] = 9
            return FixedGraspResult(True, "valid")

    monkeypatch.setattr(robust_set_module, "_generate", lambda *args: [candidate])
    result = generate_robust_grasp_set(
        mesh,
        np.eye(4),
        search_hypotheses=[hypothesis],
        validation_hypotheses=[hypothesis],
        backend=MutatingGate(),
        config=RobustSetConfig(candidate_budget=1, nominal_fraction=1, augmentation_rounds=0),
    )
    assert len(result.selected) == 1
    np.testing.assert_array_equal(result.world_commands[0], np.eye(4))
    np.testing.assert_array_equal(hypothesis.world_pose, np.eye(4))


def test_hypothesis_bank_rejects_weight_total_overflow():
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    bank = [ObjectHypothesis(mesh, np.eye(4), 1e308) for _ in range(2)]
    with pytest.raises(ValueError, match="finite total"):
        robust_set_module._make_bank(bank)


@pytest.mark.parametrize(
    "options",
    [
        {"augmentation_rounds": 1.5},
        {"max_validation_candidates": -1},
        {"dedup_translation": np.nan},
        {"diversity_angle": -1},
        {"candidate_budget": True},
    ],
)
def test_invalid_robust_set_configuration_fails_early(options):
    with pytest.raises(ValueError):
        RobustSetConfig(**options)
