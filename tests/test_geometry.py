import numpy as np
import pytest
import trimesh

from grasp_sampler import (
    GraspConfig,
    load_mesh,
    obb_face_grasps,
    side_grasps,
    tcp_to_ee,
    to_world,
    top_grasps,
)


def assert_rotation(R):
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_to_world_composes_object_pose():
    pose = np.eye(4)
    pose[:3, 3] = [.1, .2, .3]
    obj = np.eye(4)
    obj[:3, 3] = [1., 0., 0.]
    np.testing.assert_allclose(to_world(pose, obj)[0, :3, 3], [1.1, .2, .3])


def test_tcp_to_ee_backs_off_along_approach_and_maps_closing_to_y():
    ee = tcp_to_ee(np.eye(4), standoff=.105)[0]
    np.testing.assert_allclose(ee[:3, 3], [0, 0, -.105])
    np.testing.assert_allclose(ee[:3, 1], [1, 0, 0])      # gripper +Y is the TCP closing axis
    np.testing.assert_allclose(ee[:3, 2], [0, 0, 1])


def test_top_grasps_descend_and_respect_max_width():
    grasps = top_grasps([.04, .05, .1], max_width=.08, yaw_steps=4)
    assert len(grasps) == 8                                # each yaw plus its flipped twin
    for pose, meta in grasps:
        assert_rotation(pose[:3, :3])
        np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=1e-9)
        assert meta["span"] <= .08
    assert top_grasps([.04, .05, .1], max_width=.03) == []


def test_side_grasps_approach_horizontally_and_drop_too_wide_rolls():
    grasps = side_grasps([.04, .05, .1], max_width=.08, roll_steps=2)
    assert len(grasps) == 8                                # height (.1) never fits the jaws
    for pose, meta in grasps:
        assert_rotation(pose[:3, :3])
        assert abs(pose[2, 2]) < 1e-9
        assert meta["span"] <= .08


def test_obb_face_grasps_only_use_axes_that_fit_the_gripper(tmp_path):
    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=[.04, .06, .1]).export(path)
    obj = load_mesh(path, upright_transform=None)
    grasps = obb_face_grasps(obj, GraspConfig())
    assert len(grasps) == 24                               # 2 axes x 2 approaches x 2 signs x 3 offsets
    assert {round(g.width, 3) for g in grasps} == {.04, .06}
    for g in grasps:
        assert_rotation(g.pose[:3, :3])


def test_load_mesh_uprights_centers_and_records_source_transform(tmp_path):
    box = trimesh.creation.box(extents=[.04, .06, .1])
    box.apply_translation([1, 2, 3])
    path = tmp_path / "box.stl"
    box.export(path)
    obj = load_mesh(path)
    np.testing.assert_allclose(obj.extents, [.04, .1, .06], atol=1e-6)        # Y-up -> Z-up swap
    np.testing.assert_allclose(obj.mesh.bounding_box.centroid, 0, atol=1e-6)
    mapped = trimesh.transform_points(box.vertices, obj.source_to_mesh)
    np.testing.assert_allclose(np.ptp(mapped, axis=0), np.ptp(obj.mesh.vertices, axis=0), atol=1e-6)
    np.testing.assert_allclose((mapped.min(0) + mapped.max(0)) / 2, 0, atol=1e-6)


@pytest.mark.parametrize("mesh, expected", [
    (trimesh.creation.icosphere(radius=.04), "sphere"),
    (trimesh.creation.box(extents=[.04, .06, .1]), "box"),
    (trimesh.creation.cylinder(radius=.03, height=.1), "cylinder"),
], ids=["sphere", "box", "cylinder"])
def test_geometry_classification(mesh, expected, tmp_path):
    path = tmp_path / "shape.stl"
    mesh.export(path)
    assert load_mesh(path, upright_transform=None).geom_class == expected
