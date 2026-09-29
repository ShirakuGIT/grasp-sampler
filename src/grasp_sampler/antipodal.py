"""Antipodal (force-closure) grasp sampler.

A clean NumPy + trimesh re-implementation of the antipodal contact search from
Dex-Net (Mahler, Goldberg et al., UC Berkeley AUTOLAB), using ray casting in
place of the original meshpy/SDF stack.

    Repo:    https://github.com/BerkeleyAutomation/dex-net
    License: BSD-2-Clause (Copyright (c) UC Berkeley)

Method: for each sampled surface point, cast a ray into the object to find the
opposing surface contact. Accept the pair if the closing line lies inside both
friction cones and the width fits the gripper. Then emit one or more approach
directions whose gripper body provably clears the mesh.
"""
from __future__ import annotations

import weakref

import numpy as np
import trimesh

from .types import Grasp, GraspConfig, ObjMesh


def antipodal_grasps(obj: ObjMesh, cfg: GraspConfig,
                     rng: np.random.Generator | None = None) -> list[Grasp]:
    """Sample force-closure antipodal grasps on ``obj``."""
    rng = rng or np.random.default_rng(0)
    mesh = obj.mesh
    proxy = _ray_cast_proxy(mesh, cfg)            # solid we trace rays against
    pq = trimesh.proximity.ProximityQuery(mesh)   # body clearance vs the true mesh
    cone = np.cos(np.arctan(cfg.friction_coef))   # friction-cone gate
    facets = cfg.approach_facet_array()

    points, face_idx = trimesh.sample.sample_surface(
        proxy, cfg.antipodal_samples, seed=int(rng.integers(1 << 31)))
    points = np.asarray(points, float)
    normals = np.asarray(proxy.face_normals[face_idx], float)

    grasps: list[Grasp] = []
    for p1, n1 in zip(points, normals):
        contact = _opposing_contact(proxy, p1, n1, cone, cfg)
        if contact is None:
            continue
        p2, _, width, closing = contact
        center = 0.5 * (p1 + p2)

        for approach, facet_id in _approach_dirs(closing, facets, cfg):
            y = np.cross(approach, closing)
            ny = np.linalg.norm(y)
            if ny < 1e-9:
                continue
            y /= ny
            z = np.cross(closing, y)              # re-orthonormalize approach
            if not _body_clears(mesh, pq, center, closing, z, width / 2.0, cfg):
                continue
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack([closing, y, z])
            pose[:3, 3] = center
            grasps.append(Grasp(pose, "antipodal", width,
                                meta={"facet": int(facet_id),
                                      "contact_geometry": "mesh" if proxy is mesh else "convex_hull",
                                      "contact_geometry_policy": cfg.contact_geometry_policy}))
            if len(grasps) >= cfg.antipodal_max_grasps:
                return grasps
    return grasps


def _opposing_contact(proxy, p1, n1, cone, cfg):
    """Ray-cast from ``p1`` along ``-n1`` to the nearest valid opposing contact."""
    locs, _, tri = proxy.ray.intersects_location(
        (p1 - n1 * 1e-4)[None, :], (-n1)[None, :], multiple_hits=True)
    if len(locs) == 0:
        return None
    dists = np.linalg.norm(locs - p1, axis=1)
    for j in np.argsort(dists):
        width = float(dists[j])
        if width < cfg.antipodal_min_width:
            continue                              # too narrow: skip thin-wall hits
        if width > cfg.gripper_max_width:
            break                                 # all farther hits are wider too
        n2 = proxy.face_normals[tri[j]]
        closing = (locs[j] - p1) / width          # unit closing line p1 -> p2
        # Force closure: closing line inside both friction cones.
        if friction_cones_valid(n1, n2, closing, cone):
            return locs[j], n2, width, closing
    return None


def _approach_dirs(closing, facets, cfg):
    """Yield ``(approach_unit, facet_id)`` pairs perpendicular to the closing axis."""
    if not cfg.directed or len(facets) == 0:
        # Undirected: full ring of approaches about the closing axis.
        seed = np.array([0, 0, 1.0]) if abs(closing[2]) < 0.9 else np.array([1.0, 0, 0])
        u = seed - np.dot(seed, closing) * closing
        u /= np.linalg.norm(u)
        v = np.cross(closing, u)
        for k in range(cfg.antipodal_rolls):
            ang = 2.0 * np.pi * k / cfg.antipodal_rolls
            yield (np.cos(ang) * u + np.sin(ang) * v), 0
        return

    # Directed: for each facet, keep the in-plane approach nearest to it (plus a
    # small spread) when it falls within the facet's cone.
    cone_cos = np.cos(np.radians(cfg.cone_deg))
    for facet_id, facet in enumerate(facets):
        a = facet - np.dot(facet, closing) * closing      # project into the plane
        na = np.linalg.norm(a)
        if na < 1e-6:
            continue                                       # closing parallel to facet
        a /= na
        if np.dot(a, facet) < cone_cos:
            continue                                       # outside the facet cone
        perp = np.cross(closing, a)
        room = np.radians(cfg.cone_deg) - np.arccos(np.clip(np.dot(a, facet), -1, 1))
        room = float(min(max(room, 0.0), np.radians(30.0)))
        deltas = ([0.0] if cfg.directed_spread <= 1
                  else np.linspace(-room, room, cfg.directed_spread))
        for d in deltas:
            z = np.cos(d) * a + np.sin(d) * perp
            yield z / np.linalg.norm(z), facet_id


def _ray_cast_proxy(mesh, cfg):
    """Return a solid, ray-castable proxy. Hollow scans fall back to the hull."""
    if cfg.contact_geometry_policy == "mesh":
        return mesh
    if cfg.contact_geometry_policy == "convex_hull":
        return mesh.convex_hull
    if cfg.contact_geometry_policy != "existing_auto_policy":
        raise ValueError("unknown contact_geometry_policy")
    try:
        hull = mesh.convex_hull
        if hull.volume > 0 and mesh.volume / hull.volume < cfg.hull_volume_ratio:
            return hull
    except Exception:
        pass
    return mesh


def friction_cones_valid(n1, n2, closing, cone):
    """Outward normals: -n1 and +n2 must align with p1 -> p2 (+TCP X)."""
    return bool(np.dot(-np.asarray(n1), closing) >= cone and np.dot(n2, closing) >= cone)


try:  # optional: exact BVH signed-distance queries, ~100x faster than trimesh
    import open3d as _o3d
except ImportError:  # pragma: no cover
    _o3d = None

_SCENES = weakref.WeakKeyDictionary()


def _o3d_scene(mesh):
    """Cached Open3D raycasting scene for a watertight mesh, else None."""
    if _o3d is None or not mesh.is_watertight:
        return None
    scene = _SCENES.get(mesh)
    if scene is None:
        scene = _o3d.t.geometry.RaycastingScene()
        scene.add_triangles(_o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                            _o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        _SCENES[mesh] = scene
    return scene


def _body_points(center, closing, approach, half, cfg):
    fl = cfg.finger_len
    base_l = center + closing * half - approach * fl     # +X finger base
    base_r = center - closing * half - approach * fl     # -X finger base
    palm_c = center - approach * fl
    wrist = palm_c - approach * (cfg.wrist_stem - fl)

    ts = np.linspace(0.0, 1.0, 5)[:, None]
    palm = base_l[None, :] * (1.0 - ts) + base_r[None, :] * ts
    ss = np.linspace(0.0, 1.0, 4)[:, None]
    stem = palm_c[None, :] * (1.0 - ss) + wrist[None, :] * ss
    return np.vstack([palm, stem])


def _bodies_clear(mesh, pq, poses, closing, approach, cfg) -> bool:
    """Batched `_body_clears`: `poses` is a list of (center, half) pairs.

    One containment query and one closest-point query cover every pose.
    """
    pts = np.vstack([_body_points(c, closing, approach, h, cfg) for c, h in poses])
    scene = _o3d_scene(mesh)
    if scene is not None:
        # Signed distance: negative inside. float32 limits precision to ~1e-7 m.
        sd = scene.compute_signed_distance(_o3d.core.Tensor(pts.astype(np.float32))).numpy()
        return bool((sd >= 0).all() and np.abs(sd).min() >= cfg.spine_clearance)
    if mesh.contains(pts).any():
        return False
    _, dist, _ = pq.on_surface(pts)
    return bool(np.min(dist) >= cfg.spine_clearance)


def _body_clears(mesh, pq, center, closing, approach, half, cfg) -> bool:
    """True if the gripper body (palm bar + stem) clears the mesh.

    Only the body behind the fingertips is checked; the fingertips and shafts are
    expected to straddle the object -- that is the grasp.
    """
    return _bodies_clear(mesh, pq, [(center, half)], closing, approach, cfg)
