"""Transfer original scan colors onto compact geometry without changing sampler assets.

Requires the asset-prep extra. Appearance-only OBJ/PNG files are stored separately
in assets/gallery_colors. UV/color transfer is approximate; it uses the named
source scans, never an invented palette.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from prepare_household_assets import ritz_texture

ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTIONS = {
    "axe_body_spray": "axe_body_spray", "baking_soda_box": "baking_soda_box",
    "bandage_box": "hansaplast_bandaid", "black_filter_box": "toothpaste_box",
    "blue_cleaner_box": "dishwasher_cleaner_box", "cleaner_canister": "potato_chips_can",
    "palmolive_shampoo": "palmolive_shampoo", "pocky_box": "pocky_box", "ritz_crackers": "ritz_cracker_box",
}
YCB = {"apple": "013_apple", "banana": "011_banana", "baseball": "055_baseball",
       "chips_can": "001_chips_can", "cracker_box": "003_cracker_box", "foam_brick": "061_foam_brick",
       "gelatin_box": "009_gelatin_box", "large_marker": "040_large_marker", "lemon": "014_lemon",
       "master_chef_can": "002_master_chef_can", "mustard_bottle": "006_mustard_bottle",
       "orange": "017_orange", "peach": "015_peach", "pear": "016_pear",
       "potted_meat_can": "010_potted_meat_can", "pudding_box": "008_pudding_box",
       "rubiks_cube": "077_rubiks_cube", "strawberry": "012_strawberry", "sugar_box": "004_sugar_box",
       "tennis_ball": "056_tennis_ball", "tomato_soup_can": "005_tomato_soup_can",
       "tuna_fish_can": "007_tuna_fish_can"}


def prepare(sources, downloads):
    proposal = sources/"research/ntu/research_workspace/proposal/imagination"
    output = ROOT/"assets/gallery_colors"
    output.mkdir(parents=True, exist_ok=True)
    provenance = []
    for path in sorted((ROOT/"assets/meshes").glob("*.glb")):
        name = path.stem.replace("_metric_fp", "")
        if name.startswith("ycb_"):
            source_name = YCB[name[4:]]
            candidates = list((proposal/"ycb_assets"/source_name).glob("**/google_16k/textured.obj"))
            if not candidates:
                candidates = list((downloads/source_name).glob("**/textured.obj"))
            candidates.sort(key=lambda p: ("poisson" not in p.parts, str(p)))
            if not candidates:
                raise FileNotFoundError(f"Missing original source scan for {source_name}")
            source = candidates[0]
            dense = trimesh.load(source, force="mesh", process=False)
        else:
            source_name = RECONSTRUCTIONS[name]
            source = proposal/"meshes"/source_name/f"{source_name}.mesh.metric-fp.glb"
            dense = trimesh.load(source, force="mesh", process=False)
            dense.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        low = trimesh.load(path, force="mesh", process=False)
        low.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        low.apply_translation(-low.bounds.mean(axis=0))
        dense.apply_translation(-dense.bounds.mean(axis=0))
        source_extents = dense.extents.copy()
        dense.vertices *= low.extents/dense.extents
        source_texture = source.with_suffix(".png")
        if dense.visual.kind == "texture":
            # Berkeley OBJ files can omit usemtl despite carrying valid UVs.
            # Bind their original named image explicitly instead of accepting
            # trimesh's gray fallback material.
            if source_texture.exists():
                dense.visual.material.image = Image.open(source_texture).convert("RGB")
            if hasattr(dense.visual.material, "diffuse"):
                dense.visual.material.diffuse = np.array([255, 255, 255, 255], np.uint8)
            dense.visual = dense.visual.to_color()
        mesh = ritz_texture(low, dense)
        key = path.stem
        mesh.visual.material.name = key
        obj = trimesh.exchange.obj.export_obj(mesh, write_texture=False, mtl_name=f"{key}.mtl")
        (output/f"{key}.obj").write_text(obj.rstrip()+"\n")
        (output/f"{key}.mtl").write_text(f"newmtl {key}\nKd 1 1 1\nmap_Kd {key}.png\n")
        mesh.visual.material.image.convert("RGB").save(output/f"{key}.png")
        variant = "poisson" if "poisson" in source.parts else "google_16k"
        source_file = (str(source.relative_to(sources)) if source.is_relative_to(sources)
                       else f"YCB dataset/{source_name}/{variant}/textured.obj")
        provenance.append({"asset": str(path.relative_to(ROOT)), "source_object": source_name,
            "source_file": source_file,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_texture_sha256": (hashlib.sha256(source_texture.read_bytes()).hexdigest()
                                      if source_texture.exists() else None),
            "source_extents_m": source_extents.tolist(), "target_extents_m": low.extents.tolist(),
            "method": "bounds alignment; nearest original vertex color sampled into triangle UV atlas"})
        print(f"{key}: original {source_name} colors", flush=True)
    (output/"sources.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path.home()/"Documents/Coding")
    parser.add_argument("--downloads", type=Path, default=Path("/private/tmp/grasp-gallery-sources"))
    args = parser.parse_args()
    prepare(args.sources, args.downloads)
