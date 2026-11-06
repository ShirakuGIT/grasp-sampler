"""The machined plate must have real holes and a closed surface."""
from pathlib import Path

import numpy as np
import trimesh


def test_mount_is_closed_and_bolt_holes_go_through(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"examples"))
    from mujoco_mount import plate_mesh

    vertices, faces, holes = plate_mesh()
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0
    np.testing.assert_allclose(mesh.extents, [.24, .24, .012])
    origins = np.column_stack([holes, np.full(len(holes), -.02)])
    directions = np.tile([0., 0., 1.], (len(holes), 1))
    assert not mesh.ray.intersects_any(origins, directions).any()
    assert mesh.ray.intersects_any([[0, 0, -.02]], [[0, 0, 1]])[0]
    # Counterbores leave shoulders that support the washers around the smaller
    # through holes; the hardware must not float in oversized empty holes.
    shoulders = origins.copy()
    shoulders[:, 0] += .008
    assert mesh.ray.intersects_any(shoulders, directions).all()
