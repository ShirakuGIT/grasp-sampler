import numpy as np
import trimesh
from grasp_sampler import robust_set as robust_set_module
from grasp_sampler import FixedGraspResult

from grasp_sampler import (Grasp, GraspConfig, ObjectHypothesis, RobustSetConfig,
    generate_robust_grasp_set, sample_hypotheses)


def _run(seed=5, **kwargs):
    mesh = trimesh.creation.box(extents=[.04]*3)
    pose = np.eye(4)
    search = sample_hypotheses(mesh, pose, 12, seed=seed)
    validation = sample_hypotheses(mesh, pose, 16, seed=seed+1)
    return generate_robust_grasp_set(mesh, pose, search_hypotheses=search,
        validation_hypotheses=validation, requested_count=3,
        config=RobustSetConfig(candidate_budget=30, **kwargs),
        grasp_config=GraspConfig(directed=False, antipodal_samples=100,
            antipodal_max_grasps=30, contact_geometry_policy="mesh"), seed=seed)


def test_only_validation_R_accepts_and_is_independent():
    result = _run()
    assert result.diagnostics["independent_validation_bank"]
    assert all(r.accepted and r.validation_R >= .95 for r in result.selected)
    assert all(r.search_R >= 0 for r in result.records)


def test_seed_reproducibility():
    a, b = _run(17), _run(17)
    np.testing.assert_allclose(a.world_commands, b.world_commands)
    assert [(r.search_R, r.validation_R) for r in a.records] == [(r.search_R, r.validation_R) for r in b.records]


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
    assert all(r.accepted == (r.validation_R >= .95) for r in result.records)


def test_strategies_are_exposed_for_benchmarking():
    for strategy, fraction in (("nominal", 1.), ("representative", 0.), ("blind_pooling", 0.)):
        result = _run(strategy=strategy, nominal_fraction=fraction, augmentation_rounds=0)
        assert result.diagnostics["strategy"] == strategy


def test_failure_guided_calls_failing_hypothesis_source(monkeypatch):
    mesh = trimesh.creation.box(extents=[.04]*3)
    nominal = np.eye(4)
    shifted = np.eye(4); shifted[0, 3] = .01
    banks = [ObjectHypothesis(mesh, nominal), ObjectHypothesis(mesh, shifted)]
    calls = []
    def fake_generate(source, nominal_pose, cap, samples, seed, source_id, grasp_config):
        calls.append(source_id)
        p = np.eye(4); p[:3, 3] = source.world_pose[:3, 3]
        return [robust_set_module.RobustCandidate(
            Grasp(p, "test", .04), source.world_pose @ p, [source_id])]
    monkeypatch.setattr(robust_set_module, "_generate", fake_generate)
    class PoseGate:
        def evaluate(self, mesh, *, object_world_pose, world_gripper_command, opening_width):
            ok = abs(object_world_pose[0, 3]) < .001
            return FixedGraspResult(ok, "valid" if ok else "pose_failure")
    result = robust_set_module.generate_robust_grasp_set(mesh, nominal,
        requested_count=1, robustness_threshold=.5, search_hypotheses=banks,
        validation_hypotheses=banks, config=RobustSetConfig(candidate_budget=3,
            nominal_fraction=.34, augmentation_rounds=1,
            guided_hypotheses_per_round=1), grasp_config=GraspConfig(), backend=PoseGate(), seed=3)
    print('CALLS', calls, result.diagnostics)
    assert calls[0] == "nominal"
    assert any(c.startswith("failure_hypothesis:") for c in calls[1:])
    assert result.diagnostics["failure_guided"]
