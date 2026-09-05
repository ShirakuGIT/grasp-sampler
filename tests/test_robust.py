import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from grasp_sampler import (Grasp, GraspConfig, GraspSampler, load_mesh,
    PoseUncertainty, GeometryUncertainty, ObjectHypothesis, sample_hypotheses,
    RobustGraspEvaluator, GeometricGraspBackend, rank_grasps, to_world, stack_poses)
from grasp_sampler.antipodal import _ray_cast_proxy, friction_cones_valid


@pytest.fixture
def box():
    return trimesh.creation.box(extents=[0.04, 0.04, 0.04])


@pytest.fixture
def grasp():
    # TCP X joins contacts, TCP Z is approach; palm at z=-0.04.
    return Grasp(np.eye(4), "antipodal", 0.04)


def test_fixed_command_invariant(box, grasp):
    estimated = np.eye(4)
    shifted = np.eye(4)
    shifted[0, 3] = 0.01
    result = RobustGraspEvaluator(box).evaluate(grasp, nominal_object_pose=estimated,
        hypotheses=[ObjectHypothesis(box, estimated), ObjectHypothesis(box, shifted)])
    first, second = result.per_hypothesis_results
    np.testing.assert_array_equal(first.world_gripper_command, second.world_gripper_command)
    np.testing.assert_allclose(second.object_gripper_command[:3, 3], [-0.01, 0, 0])
    np.testing.assert_allclose(shifted @ second.object_gripper_command, estimated @ grasp.pose)
    np.testing.assert_allclose(to_world(stack_poses([grasp]), estimated)[0], result.world_gripper_command)
    assert not second.success  # no following or recentering on the true object


def test_translation_tolerance(box, grasp):
    hypotheses = []
    for dy in [0, 0.001, 0.03]:
        pose = np.eye(4)
        pose[1, 3] = dy
        hypotheses.append(ObjectHypothesis(box, pose))
    r = RobustGraspEvaluator(box).evaluate(grasp, nominal_object_pose=np.eye(4), hypotheses=hypotheses)
    assert [x.success for x in r.per_hypothesis_results] == [True, True, False]
    assert r.nominal_valid
    assert r.num_successes == 2


def test_geometry_width_changes_without_moving_command(box, grasp):
    original = box.vertices.copy()
    hypotheses = []
    for scale in [0.97, 1, 1.03, 2.1]:
        mesh, _ = GeometryUncertainty(scale_bounds_xyz=[[scale, scale], [1, 1], [1, 1]]).sample(box, np.random.default_rng(0))
        hypotheses.append(ObjectHypothesis(mesh, np.eye(4)))
    r = RobustGraspEvaluator(box).evaluate(grasp, nominal_object_pose=np.eye(4), hypotheses=hypotheses)
    np.testing.assert_allclose([x.validity.contact_width for x in r.per_hypothesis_results[:3]], [0.0388, 0.04, 0.0412])
    assert [x.success for x in r.per_hypothesis_results] == [True, True, True, False]
    np.testing.assert_array_equal(box.vertices, original)
    for outcome in r.per_hypothesis_results:
        np.testing.assert_array_equal(outcome.world_gripper_command, grasp.pose)


def test_deterministic_sampling_and_pivot(box):
    box.apply_translation([0.2, 0.1, 0])
    kwargs = dict(pose_uncertainty=PoseUncertainty(translation_std_xyz=[0.01]*3,
        rotation_bounds_rpy=[[-0.1, 0.1]]*3), geometry_uncertainty=GeometryUncertainty(
        scale_bounds_xyz=[[0.97, 1.03]]*3, pivot=[0.1, 0, 0]))
    a = sample_hypotheses(box, np.eye(4), 4, seed=12, **kwargs)
    b = sample_hypotheses(box, np.eye(4), 4, rng=np.random.default_rng(12), **kwargs)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.world_pose, y.world_pose)
        np.testing.assert_array_equal(x.mesh.vertices, y.mesh.vertices)
        scale = np.array(x.metadata["scale_xyz"])
        np.testing.assert_allclose(x.mesh.vertices, (box.vertices-[0.1,0,0])*scale+[0.1,0,0])


@pytest.mark.parametrize("frame", ["world", "object"])
def test_pose_composition(frame):
    nominal = np.eye(4)
    nominal[:3, :3] = Rotation.from_euler("z", np.pi/2).as_matrix()
    nominal[:3, 3] = [1, 2, 3]
    delta = Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_matrix()
    pose = PoseUncertainty(translation_bounds_xyz=[[0.01,0.01],[0,0],[0,0]],
        rotation_bounds_rpy=[[0.2,0.2],[0.3,0.3],[0.4,0.4]], frame=frame).sample(nominal, np.random.default_rng(0))
    R = nominal[:3, :3]
    np.testing.assert_allclose(pose[:3,:3], delta @ R if frame == "world" else R @ delta)
    np.testing.assert_allclose(pose[:3,3], nominal[:3,3] + (np.array([0.01,0,0]) if frame == "world" else R @ [0.01,0,0]))


def test_mesh_preservation_and_default_transform(tmp_path, box):
    box.apply_translation([0.1, 0.2, 0.3])
    path = tmp_path / "box.obj"
    box.export(path)
    preserved = load_mesh(path, center=False, upright_transform=None, process=False)
    np.testing.assert_allclose(preserved.mesh.vertices, box.vertices)
    np.testing.assert_array_equal(preserved.source_to_mesh, np.eye(4))
    normal = load_mesh(path)
    np.testing.assert_allclose(normal.mesh.bounding_box.centroid, 0, atol=1e-8)
    np.testing.assert_allclose(trimesh.transform_points(box.vertices, normal.source_to_mesh), normal.mesh.vertices)
    estimated = np.eye(4)
    estimated[:3,3] = [1,2,3]
    normalized_pose = estimated @ np.linalg.inv(normal.source_to_mesh)
    np.testing.assert_allclose(trimesh.transform_points(normal.mesh.vertices, normalized_pose),
                               trimesh.transform_points(box.vertices, estimated))
    primitive = GraspSampler().generate(preserved, methods=("primitives",))[0]
    np.testing.assert_allclose(primitive.pose[:3,3], [0.1,0.2,0.32])


def test_scene_transforms(tmp_path, box):
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3,3] = [0.1,0.2,0.3]
    scene.add_geometry(box, transform=transform)
    path = tmp_path / "scene.glb"
    scene.export(path)
    obj = load_mesh(path, center=False, upright_transform=None)
    np.testing.assert_allclose(obj.mesh.bounds, box.bounds + transform[:3,3], atol=1e-7)


def test_weights_and_ranking(box, grasp):
    shifted = np.eye(4)
    shifted[1,3] = 0.1
    hs = [ObjectHypothesis(box,np.eye(4),3), ObjectHypothesis(box,shifted,1)]
    evaluator = RobustGraspEvaluator(box)
    rows = rank_grasps([grasp, Grasp(shifted, "antipodal",0.04)], evaluator,
        nominal_object_pose=np.eye(4), hypotheses=iter(hs), nominal_scores=[0.2,0.9])
    assert rows[0].grasp_id == 0 and rows[0].nominal_score == 0.2
    assert rows[0].result.success_probability == 0.5
    assert rows[0].result.weighted_success_probability == 0.75
    assert not rows[0].result.worst_case_valid
    hs[1].weight = 0
    r = evaluator.evaluate(grasp,nominal_object_pose=np.eye(4),hypotheses=hs)
    assert r.weighted_success_probability == 1 and not r.worst_case_valid


def test_proxy_policies_and_friction(box, grasp):
    assert _ray_cast_proxy(box, GraspConfig(contact_geometry_policy="mesh")) is box
    assert _ray_cast_proxy(box, GraspConfig()) is box
    assert _ray_cast_proxy(box, GraspConfig(contact_geometry_policy="convex_hull")) is not box
    assert friction_cones_valid([-1,0,0], [1,0,0], [1,0,0], 0.9)
    assert not friction_cones_valid([1,0,0], [-1,0,0], [1,0,0], 0.9)
    r = RobustGraspEvaluator(box, geometry_policy="convex_hull").evaluate(grasp,
        nominal_object_pose=np.eye(4),hypotheses=[ObjectHypothesis(box,np.eye(4))])
    assert r.nominal_result.metadata["contact_geometry"] == "convex_hull"
    assert r.nominal_result.metadata["collision_geometry"] == "mesh"


def test_approach_collision(box, grasp):
    obstacle = trimesh.creation.box(extents=[0.02]*3)
    obstacle.apply_translation([0,0,-0.13])
    mesh = trimesh.util.concatenate([box, obstacle])
    backend = GeometricGraspBackend(approach_distance=0)
    args = dict(object_world_pose=np.eye(4),world_gripper_command=grasp.pose,opening_width=0.08)
    assert backend.evaluate(mesh, **args).valid
    r = GeometricGraspBackend(approach_distance=0.05).evaluate(mesh, **args)
    assert not r.valid and r.reason == "body_or_approach_collision"


def test_hull_can_invent_contacts(grasp):
    strips = []
    for y in [-0.03, 0.03]:
        strip = trimesh.creation.box(extents=[0.04,0.005,0.04])
        strip.apply_translation([0,y,0])
        strips.append(strip)
    mesh = trimesh.util.concatenate(strips)
    # TCP closing line lies entirely in the gap; a hull invents two surfaces.
    args = dict(nominal_object_pose=np.eye(4), hypotheses=[ObjectHypothesis(mesh,np.eye(4))])
    true_mesh = RobustGraspEvaluator(mesh).evaluate(grasp, **args)
    hull = RobustGraspEvaluator(mesh,geometry_policy="convex_hull").evaluate(grasp, **args)
    auto = RobustGraspEvaluator(mesh,geometry_policy="existing_auto_policy").evaluate(grasp, **args)
    assert not true_mesh.nominal_valid
    assert hull.nominal_valid and auto.nominal_valid
    assert auto.nominal_result.metadata["contact_geometry"] == "convex_hull"


def test_rotated_world_command_and_backend(box, grasp):
    from grasp_sampler import FixedGraspResult
    seen = []
    class Recorder:
        def evaluate(self, mesh, *, object_world_pose, world_gripper_command, opening_width):
            seen.append(world_gripper_command.copy())
            # Backend receives copies: accidental command edits must not leak.
            world_gripper_command[:] = 0
            return FixedGraspResult(True,"recorded")
    nominal = np.eye(4)
    nominal[:3,:3] = Rotation.from_euler("xyz", [0.2,0.4,0.7]).as_matrix()
    nominal[:3,3] = [0.4,-0.3,0.2]
    hs = sample_hypotheses(box, nominal, 3, seed=3,
        pose_uncertainty=PoseUncertainty(translation_std_xyz=[0.01]*3,rotation_std_rpy=[0.1]*3))
    r = RobustGraspEvaluator(box,backend=Recorder()).evaluate(grasp,nominal_object_pose=nominal,hypotheses=hs)
    for command in seen:
        np.testing.assert_array_equal(command, nominal @ grasp.pose)
    for h, outcome in zip(hs,r.per_hypothesis_results):
        np.testing.assert_allclose(h.world_pose @ outcome.object_gripper_command, nominal @ grasp.pose,atol=1e-12)


def test_pybullet_callback(box, grasp):
    pytest.importorskip("pybullet")
    from grasp_sampler.pybullet_sim import fixed_pose_collision_check
    cfg = GraspConfig()
    assert fixed_pose_collision_check(box, grasp.pose[None], 0.08, cfg)
    collision = grasp.pose.copy()
    collision[2,3] = 0.06
    assert not fixed_pose_collision_check(box, collision[None], 0.08, cfg)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weight(box, weight):
    with pytest.raises(ValueError):
        ObjectHypothesis(box,np.eye(4),weight)


def test_invalid_inputs(box, grasp):
    with pytest.raises(ValueError):
        ObjectHypothesis(box,np.zeros((4,4)))
    for hs in [[], [ObjectHypothesis(box,np.eye(4),0)]]:
        with pytest.raises(ValueError):
            RobustGraspEvaluator(box).evaluate(grasp,nominal_object_pose=np.eye(4),hypotheses=hs)
    for spec in [PoseUncertainty(frame="bad"), PoseUncertainty(translation_std_xyz=[-1]*3),
                 PoseUncertainty(rotation_bounds_rpy=[1,2,3]),
                 PoseUncertainty(translation_std_xyz=[1]*3,translation_bounds_xyz=[[0,1]]*3)]:
        with pytest.raises(ValueError):
            spec.sample(np.eye(4),np.random.default_rng(0))
    with pytest.raises(ValueError):
        GeometryUncertainty(scale_bounds_xyz=[[-1,-1]]*3).sample(box,np.random.default_rng(0))


def test_nominal_generators_smoke(tmp_path, box):
    path = tmp_path / "box.obj"
    box.export(path)
    sampler = GraspSampler(GraspConfig(antipodal_samples=30,antipodal_max_grasps=5))
    for methods in [("primitives",), ("obb_face",), ("antipodal",)]:
        gs = sampler.sample(path, methods=methods)
        assert gs
        for g in gs:
            np.testing.assert_allclose(g.pose[:3,:3].T @ g.pose[:3,:3], np.eye(3),atol=1e-7)
            assert np.linalg.det(g.pose[:3,:3]) == pytest.approx(1)
