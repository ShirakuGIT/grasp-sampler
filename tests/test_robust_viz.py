import numpy as np
import pytest
import trimesh

from grasp_sampler import Grasp, ObjectHypothesis, RobustGraspEvaluator
from grasp_sampler.robust_viz import variation_sets


def test_paired_variations():
    mesh = trimesh.creation.box(extents=[0.04]*3)
    scaled = mesh.copy()
    scaled.apply_scale([1.03, 1, 1])
    pose = np.eye(4)
    pose[0, 3] = 0.01
    modes = variation_sets(mesh, np.eye(4), [ObjectHypothesis(scaled, pose)])
    assert modes["Translation"][1].mesh is mesh
    assert modes["Shape / size"][1].mesh is scaled
    np.testing.assert_array_equal(modes["Shape / size"][1].world_pose, np.eye(4))
    np.testing.assert_array_equal(modes["Translation"][1].world_pose, pose)


def test_viewer_controls_preserve_command(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from grasp_sampler.robust_viz import RobustGraspViewer
    mesh = trimesh.creation.box(extents=[0.04]*3)
    shifted = np.eye(4)
    shifted[1, 3] = 0.1
    viewer = RobustGraspViewer(mesh, [Grasp(np.eye(4), "antipodal", 0.04)],
        np.eye(4), [ObjectHypothesis(mesh, shifted)], RobustGraspEvaluator(mesh))
    command = viewer.commands[0].copy()
    assert sum(len(line.get_xdata())-1 for line in viewer.left.lines) >= 4
    assert sum(len(line.get_xdata())-1 for line in viewer.right.lines) >= 5
    viewer._next_failure(None)
    assert viewer.sample_index == 1
    assert "FAIL" in viewer.right.get_title()
    viewer.radio.set_active(3)
    assert "PASS" in viewer.right.get_title()
    np.testing.assert_array_equal(viewer.commands[0], command)
    viewer._tick()
    assert viewer.sample_index == 0
    viewer.fig.savefig(tmp_path / "viewer.png")
    assert (tmp_path / "viewer.png").stat().st_size > 1000
    plt.close(viewer.fig)
