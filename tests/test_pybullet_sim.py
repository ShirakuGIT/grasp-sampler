"""Physical proxy placement, client ownership and sampler-frame consistency."""

import numpy as np
import pytest
import trimesh

from grasp_sampler import Grasp, GraspConfig
from grasp_sampler.pybullet_sim import collision_filter, fixed_pose_collision_check


@pytest.mark.parametrize("suffix", [".glb", ".obj", ".stl"])
def test_filter_matches_fixed_backend_in_normalized_frame_and_preserves_other_client(tmp_path, suffix):
    pybullet = pytest.importorskip("pybullet")
    from pybullet_utils.bullet_client import BulletClient

    mesh = trimesh.creation.box(extents=[0.04] * 3)
    on_disk = mesh.copy()
    on_disk.apply_translation([1, 2, 3])
    path = tmp_path / f"box{suffix}"
    on_disk.export(path)
    clear, colliding = np.eye(4), np.eye(4)
    colliding[2, 3] = 0.04  # Behind-TCP body crosses the box surface, although TCP clears it.
    poses = [clear, colliding]
    grasps = [Grasp(p, "antipodal", 0.04) for p in poses]
    other = BulletClient(connection_mode=pybullet.DIRECT)
    try:
        sentinel = other.createMultiBody(baseMass=0)
        before = other.getBodyInfo(sentinel)
        keep = collision_filter(path, grasps)
        assert keep.tolist() == [True, False]
        assert other.isConnected() and other.getBodyInfo(sentinel) == before
        expected = [fixed_pose_collision_check(mesh, p[None], 0.08, GraspConfig()) for p in poses]
        assert keep.tolist() == expected
    finally:
        other.disconnect()


def test_object_conversion_removes_temporary_files_on_loading_failure(tmp_path):
    from grasp_sampler import pybullet_sim

    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=[0.04] * 3).export(path)
    converted = []

    class FailedClient:
        GEOM_MESH, GEOM_FORCE_CONCAVE_TRIMESH = 1, 2

        def createCollisionShape(self, *args, **kwargs):
            from pathlib import Path

            converted.append(Path(kwargs["fileName"]))
            assert converted[-1].is_file()
            raise RuntimeError("simulated loader failure")

    with pytest.raises(RuntimeError, match="loader failure"):
        pybullet_sim._load_object(FailedClient(), path)
    assert converted and not converted[0].parent.exists()
