"""Small genuine sampler batches, run away from the rendering process."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from grasp_sampler import GraspConfig, GraspSampler, load_mesh

FRANKA_MAX_OPENING_M = .08


@lru_cache(maxsize=40)
def sampling_mesh(path):
    obj = load_mesh(path, process=False)
    # UV seams duplicate contact vertices. The independent viewer mesh keeps
    # its UVs; only this worker's geometry copy is cleaned up for ray queries.
    obj.mesh.merge_vertices(merge_tex=True, merge_norm=True)
    obj.mesh.update_faces(obj.mesh.nondegenerate_faces())
    obj.mesh.remove_unreferenced_vertices()
    # Rendering owns a separate appearance-preserving mesh. Cache only geometry
    # in sampling workers, without retaining forty decoded texture images.
    obj.mesh.visual = trimesh.visual.ColorVisuals(mesh=obj.mesh)
    return obj


def sample_batch(path, batch, batch_size, jaw_width, seed):
    if not np.isfinite(jaw_width) or jaw_width <= 0:
        raise ValueError("Jaw width must be finite and positive")
    limit = min(float(jaw_width), FRANKA_MAX_OPENING_M)
    obj = sampling_mesh(path)
    concave = Path(path).parent.name == "household" and Path(path).stem in {"mug", "cup", "plate", "bowl", "vase"}
    config = GraspConfig(directed=False, antipodal_samples=batch_size,
                         antipodal_rolls=6, antipodal_max_grasps=24,
                         antipodal_min_width=.001 if concave else .005,
                         gripper_max_width=limit,
                         contact_geometry_policy="mesh" if concave else "auto")
    sampler = GraspSampler(config)
    grasps = sampler.generate(obj, methods=("primitives", "obb_face") if batch == 0 else ("antipodal",),
                              seed=seed+batch*7919)
    return [{"pose": g.pose.tolist(), "width": float(g.width), "kind": g.kind} for g in grasps
            if np.isfinite(g.width) and 0 < g.width <= limit]


def unique_grasps(grasps):
    seen, result = set(), []
    for g in grasps:
        pose = np.asarray(g["pose"])
        key = (tuple(np.round(pose[:3, 3], 4)), tuple(np.round(pose[:3, :3].reshape(-1), 2)),
               round(g["width"], 4), g["kind"])
        if key not in seen:
            seen.add(key); result.append(g)
    return result
