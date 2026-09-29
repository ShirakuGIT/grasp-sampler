import numpy as np
import trimesh

from grasp_sampler import (
    FinitePadFixedObjectBackend,
    FinitePadQuasistaticBackend,
    GraspConfig,
    SymmetricPointBackend,
)


def test_backends_are_explicit_and_fixed_command_is_preserved():
    mesh = trimesh.creation.box(extents=[.04]*3)
    pose = np.eye(4)
    command = np.eye(4)
    shifted = pose.copy(); shifted[0, 3] = .002
    cfg = GraspConfig(contact_geometry_policy="mesh")
    point = SymmetricPointBackend(cfg, check_body=False)
    pad = FinitePadFixedObjectBackend(cfg, check_body=False)
    quasi = FinitePadQuasistaticBackend(cfg, check_body=False)
    rp = point.evaluate(mesh, object_world_pose=shifted, world_gripper_command=command, opening_width=.08)
    rf = pad.evaluate(mesh, object_world_pose=shifted, world_gripper_command=command, opening_width=.08)
    rq = quasi.evaluate(mesh, object_world_pose=shifted, world_gripper_command=command, opening_width=.08,
                        nominal_object_pose=pose)
    assert not rp.valid
    assert rf.valid
    assert rq.valid
    assert rf.metadata["closure_backend"] == "finite_pad_fixed_object"
    assert rq.metadata["closure_backend"] == "finite_pad_quasistatic"
    np.testing.assert_array_equal(command, np.eye(4))


def test_quasistatic_motion_is_bounded_and_reported():
    mesh = trimesh.creation.box(extents=[.04]*3)
    nominal = np.eye(4)
    true = nominal.copy(); true[0, 3] = .020
    backend = FinitePadQuasistaticBackend(GraspConfig(contact_geometry_policy="mesh"),
        check_body=False, max_object_translation=.004)
    result = backend.evaluate(mesh, object_world_pose=true,
        world_gripper_command=np.eye(4), opening_width=.08,
        nominal_object_pose=nominal)
    assert result.metadata["object_motion"]
    assert np.linalg.norm(result.metadata["object_translation_correction"]) <= .004 + 1e-12

