"""Sampler extension, selection and reproducibility contracts."""

from dataclasses import asdict

import numpy as np
import pytest
import trimesh

from grasp_sampler import Grasp, GraspConfig, GraspSampler, load_mesh


@pytest.fixture
def obj(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=[0.04, 0.05, 0.06]).export(path)
    return load_mesh(path, upright_transform=None)


def test_unknown_method_fails_before_running_any_strategy(obj):
    def must_not_run(obj, config, rng):
        pytest.fail("Validate method names before starting expensive generators")

    sampler = GraspSampler(generators={"antipodal": must_not_run})
    with pytest.raises(ValueError, match="antipodla"):
        sampler.generate(obj, methods=["antipodal", "antipodla"])
    assert sampler.generate(obj, methods=[]) == []


def test_custom_strategies_are_instance_local_and_seeded(obj):
    def custom(obj, config, rng):
        pose = np.eye(4)
        pose[:3, 3] = rng.uniform(-0.01, 0.01, 3)
        return [Grasp(pose, "antipodal", config.gripper_max_width)]

    registry = {"custom": custom}
    sampler = GraspSampler(generators=registry)
    registry.clear()  # The caller's dictionary cannot change an existing sampler.
    assert sampler.available_methods == ("primitives", "obb_face", "antipodal", "custom")
    assert "custom" not in GraspSampler().available_methods
    a = sampler.generate(obj, methods="custom", seed=4)
    b = sampler.generate(obj, methods=iter(["custom"]), seed=4)
    c = sampler.generate(obj, methods="custom", seed=5)
    np.testing.assert_array_equal(a[0].pose, b[0].pose)
    assert not np.array_equal(a[0].pose, c[0].pose)


def test_builtin_selection_preserves_order_inputs_and_seeded_output(obj):
    cfg = GraspConfig(antipodal_samples=30, antipodal_max_grasps=5)
    sampler = GraspSampler(cfg)
    vertices, config = obj.mesh.vertices.copy(), asdict(cfg)
    # Selection order is intentionally independent of execution order.
    a = sampler.generate(obj, methods=iter(["antipodal", "primitives"]), seed=2)
    b = sampler.generate(obj, methods=["primitives", "antipodal"], seed=2)
    assert a and a[0].kind == "top" and a[-1].kind == "antipodal"
    np.testing.assert_array_equal([g.pose for g in a], [g.pose for g in b])
    np.testing.assert_array_equal(obj.mesh.vertices, vertices)
    assert asdict(cfg) == config


@pytest.mark.parametrize("registry", [{"": lambda *args: []}, {"strategy": None}])
def test_invalid_strategy_registration(registry):
    with pytest.raises(ValueError, match="generators"):
        GraspSampler(generators=registry)
