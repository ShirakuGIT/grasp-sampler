"""Frame, command-preservation and planner-boundary regression tests."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from grasp_sampler import Grasp, PlanningGrasps, mesh_world_pose, pose_components, tcp_to_ee
from grasp_sampler.integrations.curobo import goal_arrays, to_curobo
from grasp_sampler.integrations.mplib import to_mplib
from grasp_sampler.integrations.mujoco import set_mocap_target, site_world_pose, solve_site_ik, to_mujoco
from grasp_sampler.integrations.pybullet import draw_grasps, to_pybullet
from grasp_sampler.mesh import load_mesh
from grasp_sampler.robust_set import RobustCandidate, RobustCandidateRecord, RobustGraspSetResult


def transform(xyz=(0, 0, 0), angles=(0, 0, 0)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    out[:3, 3] = xyz
    return out


def targets(n=2, **kwargs):
    commands = np.stack([transform((.1 * i, .2, .3), (0, .3, .7 * i)) for i in range(n)]) if n else np.empty((0, 4, 4))
    return PlanningGrasps.from_world_commands(commands, [.04] * n, frame_id="base", tool_link="tool", **kwargs)


def test_arbitrary_calibration_and_base_frame_do_not_rotate_the_approach_axis():
    local = transform((.01, .02, .03), (.3, .5, -.2))
    world = transform((.4, -.2, .6), (1., .2, -.5))
    calibration = transform((.015, -.008, -.11), (.7, -.4, 1.1))
    base = transform((-.3, .1, .2), (.2, .8, -.4))
    grasp = Grasp(local, "antipodal", .035)
    before = local.copy()
    result = PlanningGrasps.from_grasps(
        [grasp], world, tcp_to_tool=calibration, world_to_frame=base,
        frame_id="base", tool_link="flange", approach_distance=.12, lift_offset=(.02, .01, .2),
    )
    command = world @ local
    np.testing.assert_allclose(result.grasp[0], base @ command @ calibration)
    # Recover the TCP from tool targets, then measure the displacement in world.
    def recover(stage):
        return np.linalg.inv(base) @ result.poses(stage)[0] @ np.linalg.inv(calibration)
    np.testing.assert_allclose(recover("pregrasp")[:3, 3] - command[:3, 3], -.12 * command[:3, 2])
    np.testing.assert_allclose(recover("lift")[:3, 3] - command[:3, 3], [.02, .01, .2])
    np.testing.assert_array_equal(grasp.pose, before)


def test_panda_calibration_matches_existing_helper():
    commands = targets().grasp
    calibration = tcp_to_ee(np.eye(4)[None])[0]
    result = PlanningGrasps.from_world_commands(commands, [.04, .03], frame_id="world",
                                               tool_link="panda_link8", tcp_to_tool=calibration)
    np.testing.assert_allclose(result.grasp, tcp_to_ee(commands))


def test_source_frame_matches_loaded_mesh_vertices(tmp_path):
    path = tmp_path / "offset.ply"
    mesh = trimesh.creation.box(extents=[.03, .04, .05])
    mesh.apply_translation([.2, -.1, .3])
    mesh.export(path)
    obj = load_mesh(path)
    source_pose = transform((.4, .2, .6), (.3, .5, .7))
    mesh_pose = mesh_world_pose(obj, source_pose)
    np.testing.assert_allclose(mesh_pose @ obj.source_to_mesh, source_pose, atol=1e-12)


def test_robust_selected_commands_are_not_recomputed_from_local_grasps():
    records = []
    for i in range(3):
        candidate = RobustCandidate(Grasp(np.eye(4), "antipodal", .03 + .005 * i),
                                    transform((i * .1, .2, .3), (.1, .2, .3)))
        records.append(RobustCandidateRecord(candidate, .99, .97 + i * .01, 99, 97 + i,
                                            {}, {}, True, True))
    robust = RobustGraspSetResult([records[2], records[0]], records, {})
    result = PlanningGrasps.from_robust_result(robust, frame_id="world", tool_link="tcp")
    np.testing.assert_allclose(result.grasp, robust.world_commands)
    np.testing.assert_array_equal(result.candidate_ids, [2, 0])
    np.testing.assert_allclose(result.widths, [.04, .03])
    np.testing.assert_allclose(result.scores, [.99, .97])


def test_filter_and_serialization_preserve_alignment_and_empty_batches():
    batch = targets(3, scores=[.96, .97, .99], candidate_ids=[12, 4, 8])
    selected = batch.select([True, False, True]).select([1, 0])
    np.testing.assert_array_equal(selected.candidate_ids, [8, 12])
    np.testing.assert_allclose(selected.scores, [.99, .96])
    for value in (batch, selected, batch.select([]), targets(0)):
        restored = PlanningGrasps.from_dict(json.loads(json.dumps(value.to_dict())))
        np.testing.assert_array_equal(restored.grasp, value.grasp)
        np.testing.assert_array_equal(restored.candidate_ids, value.candidate_ids)
        assert restored.components()[1].shape == (len(value), 4)
    with pytest.raises(ValueError, match="unique"):
        batch.select([0, 0])
    with pytest.raises(ValueError, match="shape"):
        batch.select([True])
    with pytest.raises(ValueError, match="shape"):
        batch.select(np.array([], dtype=bool))
    with pytest.raises(ValueError):
        batch.grasp[0, 0, 0] = 2


@pytest.mark.parametrize("change", [
    {"approach_distance": -1}, {"approach_distance": np.nan}, {"lift_offset": [1, 2]},
    {"world_to_frame": np.diag([2, 1, 1, 1])}, {"tcp_to_tool": np.diag([-1, 1, 1, 1])},
    {"candidate_ids": [1.1, 2.2]}, {"scores": [1.1, .9]},
])
def test_invalid_planning_inputs_are_rejected(change):
    with pytest.raises(ValueError):
        targets(**change)


def test_quaternion_orders_and_external_pose_round_trips():
    batch = targets()
    p, xyzw = batch.components()
    _, wxyz = to_mujoco(batch)
    np.testing.assert_allclose(wxyz, xyzw[:, [3, 0, 1, 2]])
    for i, (position, quaternion) in enumerate(to_pybullet(batch)):
        np.testing.assert_allclose(position, p[i])
        np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(), batch.grasp[i, :3, :3])
    with pytest.raises(ValueError):
        pose_components(batch.grasp, quaternion_order="wzyx")
    with pytest.raises(ValueError):
        batch.poses("approach")


@pytest.mark.parametrize("api,layout,shape", [
    ("current", "goalset", (1, 1, 1, 2, 3)), ("current", "batch", (2, 1, 1, 1, 3)),
    ("legacy", "goalset", (1, 2, 3)), ("legacy", "batch", (2, 3)),
])
def test_curobo_goal_axes_and_order(api, layout, shape):
    p, q = goal_arrays(targets(), api=api, layout=layout)
    assert p.shape == shape
    assert q.shape == (*shape[:-1], 4)
    expected_p, expected_q = targets().components(quaternion_order="wxyz")
    np.testing.assert_allclose(p.reshape(-1, 3), expected_p)
    np.testing.assert_allclose(q.reshape(-1, 4), expected_q)


def test_curobo_requires_nonempty_goals():
    with pytest.raises(ValueError, match="at least one"):
        goal_arrays(targets(0))


def test_curobo_constructor_contract_with_real_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    # No CUDA/cuRobo is required in default CI. Real tensor conversion is tested
    # against the documented public constructors; GPU planning is a separate check.
    curobo = ModuleType("curobo")
    types = ModuleType("curobo.types")
    math = ModuleType("curobo.types.math")
    types.GoalToolPose = lambda **kw: SimpleNamespace(**kw)
    math.Pose = lambda **kw: SimpleNamespace(**kw)
    for name, module in (("curobo", curobo), ("curobo.types", types), ("curobo.types.math", math)):
        monkeypatch.setitem(sys.modules, name, module)
    for api in ("current", "legacy"):
        goal = to_curobo(targets(), api=api, device="cpu", dtype=torch.float64)
        assert goal.position.dtype == torch.float64
        assert goal.position.device.type == "cpu"
        if api == "current":
            assert goal.tool_frames == ["tool"]
        else:
            assert goal.name == "tool"
        np.testing.assert_allclose(goal.quaternion.numpy().reshape(-1, 4),
                                   targets().components(quaternion_order="wxyz")[1])


def test_moveit_message_contract(monkeypatch):
    from grasp_sampler.integrations.moveit import to_pose_stamped

    def message():
        return SimpleNamespace(header=SimpleNamespace(frame_id="", stamp=None),
                               pose=SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace()))
    module = ModuleType("geometry_msgs.msg")
    module.PoseStamped = message
    monkeypatch.setitem(sys.modules, "geometry_msgs", ModuleType("geometry_msgs"))
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", module)
    stamp = object()
    batch = targets()
    result = to_pose_stamped(batch, stage="pregrasp", stamp=stamp)
    assert len(result) == 2
    assert result[0].header.frame_id == "base"
    assert result[0].header.stamp is stamp
    pos, quat = batch.components("pregrasp")
    np.testing.assert_allclose([result[1].pose.position.x, result[1].pose.position.y,
                               result[1].pose.position.z], pos[1])
    np.testing.assert_allclose([result[1].pose.orientation.x, result[1].pose.orientation.y,
                               result[1].pose.orientation.z, result[1].pose.orientation.w], quat[1])


def test_mplib_pose_contract(monkeypatch):
    module = ModuleType("mplib")
    module.Pose = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "mplib", module)
    result = to_mplib(targets())
    p, q = targets().components(quaternion_order="wxyz")
    np.testing.assert_allclose([r.p for r in result], p)
    np.testing.assert_allclose([r.q for r in result], q)


def test_pybullet_actual_link_frame_ik():
    p = pytest.importorskip("pybullet")
    data = pytest.importorskip("pybullet_data")
    client = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(data.getDataPath(), physicsClientId=client)
        robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=client)
        for i, q in enumerate([0, -.4, 0, -2.2, 0, 1.8, .8]):
            p.resetJointState(robot, i, q, physicsClientId=client)
        state = p.getLinkState(robot, 11, computeForwardKinematics=True, physicsClientId=client)
        command = transform(state[4])
        command[:3, :3] = Rotation.from_quat(state[5]).as_matrix()
        batch = PlanningGrasps.from_world_commands(command[None], [.04], frame_id="world", tool_link="panda_grasptarget")
        position, quaternion = to_pybullet(batch)[0]
        solution = p.calculateInverseKinematics(robot, 11, position, quaternion, maxNumIterations=200,
                                               residualThreshold=1e-8, physicsClientId=client)
        for i, value in enumerate(solution[:7]):
            p.resetJointState(robot, i, value, physicsClientId=client)
        actual = p.getLinkState(robot, 11, computeForwardKinematics=True, physicsClientId=client)
        np.testing.assert_allclose(actual[4], position, atol=1e-5)
    finally:
        p.disconnect(client)


def test_mujoco_actual_mocap_and_site():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="target" mocap="true"><site name="tcp"/></body>
      <body name="other" mocap="true" pos="1 0 0"/>
      <body name="fixed"><site name="fixed_site"/></body>
    </worldbody></mujoco>''')
    data = mj.MjData(model)
    other = data.mocap_pos[1].copy()
    batch = targets()
    set_mocap_target(model, data, "target", batch, index=1, stage="pregrasp")
    np.testing.assert_allclose(site_world_pose(model, data, "tcp"), batch.pregrasp[1], atol=1e-12)
    np.testing.assert_array_equal(data.mocap_pos[1], other)
    for name in ("missing", "fixed"):
        with pytest.raises(ValueError):
            set_mocap_target(model, data, name, batch)
    with pytest.raises(IndexError):
        set_mocap_target(model, data, "target", batch, index=9)


def test_pybullet_grasp_visuals_have_no_collision_geometry():
    p = pytest.importorskip("pybullet")
    client = p.connect(p.DIRECT)
    try:
        grasps = [Grasp(transform((.1 * i, 0, .2)), "antipodal", .04) for i in range(4)]
        bodies = draw_grasps(grasps, np.eye(4), selected_index=3, rejected_indices=[0],
                            max_grasps=2, physics_client_id=client)
        assert len(bodies) == 2  # subsampling includes selected candidate 3
        assert all(not p.getCollisionShapeData(body, -1, physicsClientId=client) for body in bodies)
        np.testing.assert_allclose(p.getBasePositionAndOrientation(bodies[-1], physicsClientId=client)[0], [.3, 0, .2])
        with pytest.raises(IndexError):
            draw_grasps(grasps, np.eye(4), selected_index=8, physics_client_id=client)
    finally:
        p.disconnect(client)


def test_mujoco_site_ik_preserves_live_state_and_reports_unreachable_goals():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size=".02" mass="1"/>
      <site name="tip" pos="1 0 0"/>
    </body></worldbody></mujoco>'''.replace('<mujoco>', '<mujoco><compiler angle="radian"/>'))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    before = data.qpos.copy()
    goal = transform((np.cos(.6), np.sin(.6), 0), (0, 0, .6))
    result = solve_site_ik(model, data, "tip", goal, ["shoulder"])
    assert result.success
    np.testing.assert_allclose(result.positions, [.6], atol=.002)
    np.testing.assert_array_equal(data.qpos, before)
    unreachable = goal.copy(); unreachable[0, 3] = 3
    failed = solve_site_ik(model, data, "tip", unreachable, ["shoulder"], max_iterations=20)
    assert not failed.success
    assert failed.position_error > 1
    assert abs(failed.positions[0]) <= 1.5
    with pytest.raises(ValueError, match="unknown"):
        solve_site_ik(model, data, "missing", goal, ["shoulder"])


def test_mujoco_placement_requires_the_correct_support_and_allows_settled_rotation(monkeypatch):
    mj = pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    monkeypatch.setenv("GRASP_SAMPLER_DISABLE_OPEN3D", "1")
    from pick_place_mujoco import Workstation

    model = mj.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
      <geom name="table" type="box" pos="0 0 .71" size="1 1 .01"/>
      <geom name="upper_tier" type="box" pos="0 0 1.11" size="1 1 .01"/>
      <geom name="upper_tier_edge" type="box" pos="1 0 1.10" size=".01 1 .01"/>
      <body name="gripper_hand" pos="5 0 0"><geom size=".01" contype="0" conaffinity="0"/></body>
      <body name="item" pos="0 0 1.14" quat=".70710678 0 .70710678 0">
        <freejoint/><geom type="box" size=".02 .03 .07" mass=".06"/>
      </body></worldbody></mujoco>''')
    data = mj.MjData(model)
    demo = object.__new__(Workstation)
    demo.model, demo.data = model, data
    demo.body_ids, demo.finger_ids = [model.body("item").id], set()
    demo.objects = [(trimesh.creation.box(extents=[.04, .06, .14]), "box", np.zeros(3))]
    mj.mj_step(model, data, nstep=400)
    passed, metrics = demo.placement_status(0, np.array([0., 0., 1.19]))
    assert passed and metrics["supported"] and metrics["released"]
    assert metrics["support_normal_force"] > .1
    # Resting on the tabletop at the same XY is not an upper-tier success.
    data.qpos[2] = .74
    data.qvel[:] = 0
    mj.mj_step(model, data, nstep=400)
    passed, metrics = demo.placement_status(0, np.array([0., 0., 1.19]))
    assert not passed
    assert not metrics["supported"]


def test_mujoco_joint_transit_routes_around_an_obstacle_without_moving_live_state(monkeypatch):
    mj = pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    monkeypatch.setenv("GRASP_SAMPLER_DISABLE_OPEN3D", "1")
    from pick_place_mujoco import Workstation

    model = mj.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/><worldbody>
      <geom name="wall" type="box" pos="0 0 .1" size=".05 .3 .1"/>
      <body name="robot" pos="0 0 .1">
        <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
        <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
        <geom type="sphere" size=".05" mass="1"/>
        <body name="gripper_hand"/>
      </body></worldbody></mujoco>''')
    demo = object.__new__(Workstation)
    demo.model, demo.data = model, mj.MjData(model)
    demo.q_indices, demo.joint_ids = np.array([0, 1]), [0, 1]
    demo.robot_bodies = {model.body("robot").id, model.body("gripper_hand").id}
    demo.held_index, demo.finger_ids, demo.planning_calls = None, set(), 0
    demo.phase = "obstacle regression"
    start, goal = np.array([-.6, 0.]), np.array([.6, 0.])
    demo.data.qpos[:] = start
    path = demo.joint_path(start, goal)
    np.testing.assert_array_equal(demo.data.qpos, start)
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    assert len(path) > 2
    assert max(abs(q[1]) for q in path) >= .34
    scratch = mj.MjData(model)
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0, 1, 40):
            scratch.qpos[:] = (1-t)*a+t*b
            mj.mj_forward(model, scratch)
            assert all(c.dist >= -.0011 for c in scratch.contact[:scratch.ncon])
