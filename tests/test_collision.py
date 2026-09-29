import numpy as np
import pytest
import trimesh

from grasp_sampler import GraspConfig, collision


def _random_bodies(rng, count=300):
    for _ in range(count):
        center = rng.uniform(-.09, .09, 3)
        closing = rng.normal(size=3)
        closing /= np.linalg.norm(closing)
        approach = np.cross(closing, rng.normal(size=3))
        approach /= np.linalg.norm(approach)
        yield center, closing, approach


@pytest.mark.parametrize("mesh", [trimesh.creation.icosphere(radius=.04),
                                  trimesh.creation.box(extents=[.04, .06, .08])],
                         ids=["sphere", "box"])
def test_open3d_and_trimesh_agree_on_body_clearance(mesh, monkeypatch):
    pytest.importorskip("open3d")
    cfg = GraspConfig()
    bodies = list(_random_bodies(np.random.default_rng(0)))

    def run():
        return [collision.bodies_clear(mesh, [(c, .04)], closing, approach, cfg)
                for c, closing, approach in bodies]

    fast = run()
    monkeypatch.setattr(collision, "_o3d", None)      # force the trimesh fallback
    slow = run()
    assert set(fast) == {True, False}, "benchmark must exercise both outcomes"
    assert fast == slow


def test_signed_distance_scene_requires_watertight_mesh():
    pytest.importorskip("open3d")
    box = trimesh.creation.box()
    assert collision._signed_distance_scene(box) is not None
    assert collision._signed_distance_scene(box) is collision._signed_distance_scene(box)  # cached
    open_mesh = trimesh.Trimesh(box.vertices, box.faces[1:], process=False)
    assert collision._signed_distance_scene(open_mesh) is None


def test_bodies_clear_rejects_body_inside_mesh():
    mesh = trimesh.creation.box(extents=[.1, .1, .1])
    z = np.array([0., 0., 1.])
    x = np.array([1., 0., 0.])
    assert not collision.bodies_clear(mesh, [(np.zeros(3), .02)], x, z, GraspConfig())
    assert collision.bodies_clear(mesh, [(np.array([0., 0., .3]), .02)], x, z, GraspConfig())


def test_ray_cast_proxy_policies():
    box = trimesh.creation.box()
    assert collision.ray_cast_proxy(box, GraspConfig(contact_geometry_policy="mesh")) is box
    assert collision.ray_cast_proxy(box, GraspConfig()) is box                      # "auto": solid stays
    assert collision.ray_cast_proxy(box, GraspConfig(contact_geometry_policy="convex_hull")) is not box
    with pytest.raises(ValueError):
        collision.ray_cast_proxy(box, GraspConfig(contact_geometry_policy="bogus"))
