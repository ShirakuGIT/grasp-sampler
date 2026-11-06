"""Appearance, library coverage, and rotating-frame gallery regressions."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh


def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"examples"))
    import grasp_gallery_assets as assets
    import grasp_gallery_sampling as sampling
    return assets, sampling


def test_inventory_includes_object_variants_not_robot_parts_or_sidecars(tmp_path, monkeypatch):
    assets, _ = modules(monkeypatch)
    for folder in ("meshes", "household", "robots"):
        directory = tmp_path/folder
        directory.mkdir()
        for name in ("object.glb", "object.obj", "material.png"):
            (directory/name).touch()
    paths = assets.inventory(tmp_path)
    assert [p.parent.name for p in paths] == ["meshes", "household"]
    assert all(p.suffix == ".glb" for p in paths)


def test_vertex_color_bake_keeps_original_corner_colors(monkeypatch):
    assets, _ = modules(monkeypatch)
    mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    colors = np.array([[230, 20, 50, 255], [10, 220, 70, 255], [40, 60, 240, 255]], np.uint8)
    mesh.visual.vertex_colors = colors
    baked = assets.vertex_color_texture(mesh)
    pixels = np.asarray(baked.visual.material.image)
    np.testing.assert_array_equal(pixels[0, 0], colors[0])
    np.testing.assert_array_equal(pixels[0, -1], colors[1])
    np.testing.assert_array_equal(pixels[-1, 0], colors[2])
    np.testing.assert_array_equal(baked.vertices, mesh.vertices)


def test_batches_are_actual_sampler_outputs_in_metric_frames(tmp_path, monkeypatch):
    _, sampling = modules(monkeypatch)
    path = tmp_path/"small_box.glb"
    trimesh.creation.box(extents=[.04, .05, .06]).export(path)
    batch = sampling.sample_batch(str(path), 1, 96, .08, 42)
    assert batch
    assert all(g["kind"] == "antipodal" and 0 < g["width"] <= .08 for g in batch)
    for grasp in batch:
        pose = np.asarray(grasp["pose"])
        np.testing.assert_allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-7)
        assert np.linalg.norm(pose[:3, 3]) < .06
    assert batch == sampling.sample_batch(str(path), 1, 96, .08, 42)


def test_grid_keeps_all_40_objects_visible_and_grasps_rotate_with_them(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    assets, _ = modules(monkeypatch)
    from grasp_gallery import GridGallery, mj

    path = tmp_path/"small_box.glb"
    trimesh.creation.box(extents=[.04, .05, .06]).export(path)
    item = assets.prepare_item(path, tmp_path/"cache")
    items = [replace(item, label=f"Object {i}") for i in range(40)]
    gallery = GridGallery(items, tmp_path/"cache", gui=False, visible=2)
    gallery.select()
    gallery.angle = np.pi/2
    gallery.update_pose()
    assert np.all(gallery.model.geom_rgba[gallery.geoms, 3] == 1)
    assert gallery.model.ngeom == 40  # Objects only: no stands, labels or HUD geometry.
    assert len(np.unique(gallery.positions[:, 0])) == 8
    assert len(np.unique(gallery.positions[:, 2])) == 5
    expected = np.tile([np.sqrt(.5), 0, 0, np.sqrt(.5)], (40, 1))
    np.testing.assert_allclose(gallery.data.mocap_quat[gallery.mocaps], expected, atol=1e-7)
    gallery.candidates[7] = [{"pose": np.eye(4).tolist(), "width": .03, "kind": "antipodal"}]
    scene = mj.MjvScene(gallery.model, maxgeom=100)
    gallery.decorate(scene)
    assert scene.ngeom == 6
    center = gallery.data.mocap_pos[gallery.mocaps[7]].copy()
    depth = min(.04, item.extents.max()*.22)*item.scale
    np.testing.assert_allclose(scene.geoms[0].pos,
                               center+[0, .03*item.scale/2, -depth/2], atol=1e-7)


def test_restored_compact_appearances_are_colored_and_keep_geometry(monkeypatch):
    from PIL import Image

    from grasp_sampler import load_mesh

    assets, _ = modules(monkeypatch)
    paths = list((assets.ROOT/"assets/meshes").glob("*.glb"))
    assert len(paths) == 31
    for path in paths:
        restored = assets.ROOT/"assets/gallery_colors"/path.stem
        assert restored.with_suffix(".obj").exists()
        pixels = np.asarray(Image.open(restored.with_suffix(".png")).convert("RGB"), dtype=int)
        assert np.ptp(pixels, axis=2).max() > 10, path.name
        original = load_mesh(path, process=False).mesh
        visual = trimesh.load(restored.with_suffix(".obj"), force="mesh", process=False)
        np.testing.assert_allclose(visual.bounds, original.bounds, atol=2e-7)
        assert len(visual.faces) == len(original.faces)
