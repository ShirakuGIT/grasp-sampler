"""Mesh normalization must keep source colors, UVs and images aligned."""
import numpy as np
import pytest
import trimesh
from PIL import Image

from grasp_sampler import load_mesh


def test_normalization_preserves_vertex_colors(tmp_path):
    mesh = trimesh.creation.box(extents=[.04, .08, .12])
    colors = np.tile([240, 35, 75, 255], (len(mesh.vertices), 1)).astype(np.uint8)
    colors[0] = [10, 180, 240, 255]
    mesh.visual.vertex_colors = colors
    path = tmp_path / "colored.glb"
    mesh.export(path)
    loaded = load_mesh(path, process=False).mesh
    assert loaded.visual.kind == "vertex"
    np.testing.assert_array_equal(loaded.visual.vertex_colors, colors)
    np.testing.assert_allclose(loaded.extents, [.04, .12, .08])


def test_normalization_preserves_uvs_and_texture(tmp_path):
    mesh = trimesh.creation.box()
    uv = np.random.default_rng(4).uniform(size=(len(mesh.vertices), 2))
    pixels = np.array([[[230, 20, 90], [15, 210, 110]],
                       [[40, 70, 240], [250, 190, 20]]], dtype=np.uint8)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(pixels))
    path = tmp_path / "textured.glb"
    mesh.export(path)
    loaded = load_mesh(path, process=False).mesh
    assert loaded.visual.kind == "texture"
    np.testing.assert_allclose(loaded.visual.uv, uv, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(loaded.visual.material.baseColorTexture), pixels)


@pytest.mark.parametrize(("name", "air_probe"), [
    ("mug", [.043, -.003, .008]),
    ("cup", [0, 0, .015]),
    ("plate", [0, 0, .005]),
])
def test_household_contact_parts_leave_handle_and_cavities_open(name, air_probe):
    from pathlib import Path

    directory = Path(__file__).resolve().parents[1]/"assets/household"
    mesh = trimesh.load(directory/f"{name}.obj", force="mesh", process=False)
    # A single convex hull would fill these specific handle/cavity probes.
    assert mesh.convex_hull.contains([air_probe])[0]
    with np.load(directory/f"{name}_collision.npz", allow_pickle=False) as parts:
        count = len(parts.files)//2
        assert count > 1
        for i in range(count):
            part = trimesh.Trimesh(parts[f"vertices_{i}"], parts[f"faces_{i}"], process=False)
            assert not part.contains([air_probe])[0]
