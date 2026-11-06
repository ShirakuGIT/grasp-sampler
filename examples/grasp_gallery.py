"""A live 8 x 5 object gallery with slow 360-degree rotation and sampler batches.

macOS: PYTHONPATH=src mjpython examples/grasp_gallery.py
Linux: PYTHONPATH=src python examples/grasp_gallery.py
Audit all assets: PYTHONPATH=src python examples/grasp_gallery.py --headless --batches 2 --report /tmp/gallery.json

SPACE pauses rotation; G toggles glyphs; R resamples; Q or Escape closes.
Mouse controls orbit and zoom. --layout turntable shows one object at a time,
with N/right and P/left navigation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("GRASP_SAMPLER_DISABLE_OPEN3D", "1")

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402
from grasp_gallery_assets import (  # noqa: E402
    ROOT,
    grid_layout,
    inventory,
    prepare_item,
    scene_xml,
)
from grasp_gallery_sampling import FRANKA_MAX_OPENING_M, sample_batch, unique_grasps  # noqa: E402


class Gallery:
    def __init__(self, items, cache, *, fps=60, seconds=12., batches=12, batch_size=96,
                 jaw_width=.08, visible=32, seed=0, gui=True, grid=False):
        self.items, self.fps, self.seconds, self.grid = items, fps, seconds, grid
        self.positions = grid_layout(len(items)) if grid else np.zeros((len(items), 3))
        self.batches, self.batch_size, self.jaw_width = batches, batch_size, min(jaw_width, FRANKA_MAX_OPENING_M)
        self.visible, self.seed = visible, seed
        self.model = mj.MjModel.from_xml_string(scene_xml(items, cache, grid=grid))
        self.data = mj.MjData(self.model)
        self.geoms = [self.model.geom(f"visual_{i}").id for i in range(len(items))]
        self.mocaps = [self.model.body_mocapid[self.model.body(f"object_{i}").id] for i in range(len(items))]
        self.index, self.angle, self.elapsed = 0, 0., 0.
        self.grasps, self.completed_batches = [], 0
        self.latest_from, self.arrival = 0, 0.
        self.paused, self.show_grasps = False, True
        self.events = queue.SimpleQueue()
        self.viewer = None
        self.records = []
        self.visibility_cache = {}
        self.frame_times = deque(maxlen=180)
        self.generation, self.resamples = 0, 0
        self.sampling_error = None
        if gui:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data,
                key_callback=self.events.put, show_left_ui=False, show_right_ui=False)
            self.configure_camera(self.viewer.cam)

    def configure_camera(self, camera):
        camera.lookat[:] = [0, 0, .84] if self.grid else [0, 0, .16]
        camera.distance = 2.60 if self.grid else .70
        camera.azimuth, camera.elevation = 90, -20 if self.grid else -22
        camera.orthographic = self.grid

    def select(self, index):
        for geom in self.geoms:
            self.model.geom_rgba[geom, 3] = 0
        self.data.mocap_pos[:] = [0, 0, -20]
        self.index = index % len(self.items)
        self.elapsed, self.angle = 0., 0.
        self.grasps, self.completed_batches, self.latest_from = [], 0, 0
        self.sampling_error = None
        self.generation += 1
        self.model.geom_rgba[self.geoms[self.index], 3] = 1
        self.update_pose()
        if self.viewer:
            self.viewer.sync()  # visibility/material changes happen once per object
        item = self.items[self.index]
        print(f"[{self.index+1:02d}/{len(self.items)}] {item.label} / {item.variant} / {item.path}", flush=True)

    def update_pose(self):
        mocap = self.mocaps[self.index]
        self.data.mocap_pos[mocap] = [0, 0, .16]
        self.data.mocap_quat[mocap] = [np.cos(self.angle/2), 0, 0, np.sin(self.angle/2)]
        mj.mj_forward(self.model, self.data)

    def add_batch(self, batch):
        self.latest_from = len(self.grasps)
        self.grasps = unique_grasps([*self.grasps, *batch])
        self.arrival = time.monotonic()
        self.completed_batches += 1
        print(f"  batch {self.completed_batches:02d}/{self.batches}: {len(self.grasps)} unique candidates", flush=True)

    def decorate(self, scene):
        self.decorate_item(scene, self.index, self.grasps, self.latest_from, self.arrival)

    def decorate_item(self, scene, index, grasps, latest_from, arrival):
        if not self.show_grasps or not grasps:
            return
        item = self.items[index]
        c, s = np.cos(self.angle), np.sin(self.angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        translation = self.data.mocap_pos[self.mocaps[index]]
        # Thin only the drawing, retaining every generated candidate. Avoid
        # stacking six roll variants at the same contact pair into a wire nest.
        cached = self.visibility_cache.get(index)
        if cached is not None and cached[0] is grasps:
            indices = cached[1]
        else:
            indices, occupied = [], set()
            for i in range(len(grasps)-1, -1, -1):
                pose = np.asarray(grasps[i]["pose"])
                key = (tuple(np.round(pose[:3, 3]*item.scale/.008)),
                       tuple(np.round(pose[:3, 0])))
                if key not in occupied:
                    occupied.add(key); indices.append(i)
                if len(indices) >= self.visible:
                    break
            indices.reverse()
            self.visibility_cache[index] = (grasps, indices)
        depth = .04*item.scale
        stem = .025*item.scale
        for i in indices:
            g = grasps[i]
            pose = np.asarray(g["pose"])
            r, center = rotation@pose[:3, :3], translation+rotation@pose[:3, 3]*item.scale
            half = g["width"]*item.scale/2
            points = np.array([[half, 0, 0], [half, 0, -depth], [-half, 0, 0],
                               [-half, 0, -depth], [0, 0, -depth], [0, 0, -depth-stem]])@r.T+center
            fresh = i >= max(latest_from, len(grasps)-3) and time.monotonic()-arrival < 1.1
            color = [1., .78, .12, 1.] if fresh else ([.10, .95, 1., 1.] if g["kind"] == "antipodal"
                                                   else [.75, .48, 1., .8])
            for a, b in ((0, 1), (2, 3), (1, 3), (5, 4)):
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).reshape(-1), np.asarray(color, np.float32))
                kind = mj.mjtGeom.mjGEOM_ARROW if a == 5 else mj.mjtGeom.mjGEOM_CAPSULE
                mj.mjv_connector(geom, kind, .0028 if fresh else .0021, points[a], points[b])
                geom.emission = .8
                geom.category = mj.mjtCatBit.mjCAT_DECOR
                scene.ngeom += 1
            for point in points[[0, 2]]:
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_SPHERE, np.array([.002, 0, 0]), point,
                                np.eye(3).reshape(-1), np.asarray(color, np.float32))
                geom.emission = 1.
                geom.category = mj.mjtCatBit.mjCAT_DECOR
                scene.ngeom += 1

    def draw(self):
        if self.viewer:
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
                self.decorate(self.viewer.user_scn)
            self.viewer.sync(state_only=True)

    def snapshot(self, path):
        from PIL import Image
        camera = mj.MjvCamera()
        self.configure_camera(camera)
        with mj.Renderer(self.model, height=1600 if self.grid else 1000, width=2560 if self.grid else 1600) as renderer:
            renderer.update_scene(self.data, camera=camera)
            self.decorate(renderer.scene)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(path)

    def record(self, error=None):
        item = self.items[self.index]
        return {"asset": str(item.path.relative_to(ROOT)), "label": item.label,
                "source_extents_m": item.extents.tolist(), "display_scale": item.scale,
                "unique_candidates": len(self.grasps), "batches": self.completed_batches,
                "max_opening_m": max((g["width"] for g in self.grasps), default=None),
                "rotation_degrees": float(np.degrees(self.angle)), "error": error}

    def run(self, start=0, once=False):
        self.select(start)
        visited, future, submitted_generation = 0, None, None
        last, next_frame = time.monotonic(), time.monotonic()
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            while self.viewer.is_running():
                now = time.monotonic()
                dt, last = min(now-last, .1), now
                while not self.events.empty():
                    key = self.events.get()
                    if key in (81, 256):
                        return
                    if key == 32:
                        self.paused = not self.paused
                    elif key in (78, 262):
                        self.select(self.index+1)
                    elif key in (80, 263):
                        self.select(self.index-1)
                    elif key == 71:
                        self.show_grasps = not self.show_grasps
                    elif key in (61, 334, 45, 333):
                        self.visible = max(1, min(16, self.visible+(1 if key in (61, 334) else -1)))
                        self.visibility_cache.clear()
                    elif key == 82:
                        self.resamples += 1; self.select(self.index)
                if not self.paused:
                    self.elapsed += dt
                    self.angle = self.elapsed/self.seconds*2*np.pi
                if future is not None and future.done():
                    if submitted_generation == self.generation:
                        try:
                            self.add_batch(future.result())
                        except Exception as error:
                            self.completed_batches = self.batches
                            self.sampling_error = str(error)
                            print(f"Sampling failed: {error}", file=sys.stderr, flush=True)
                    future = None
                if (future is None and self.completed_batches < self.batches
                        and self.elapsed >= self.completed_batches*self.seconds/self.batches):
                    submitted_generation = self.generation
                    future = pool.submit(sample_batch, str(self.items[self.index].path), self.completed_batches,
                                         self.batch_size, self.jaw_width, self.seed+self.index*101+self.resamples*100003)
                if self.elapsed >= self.seconds and self.completed_batches == self.batches and not self.paused:
                    self.records.append(self.record(self.sampling_error))
                    visited += 1
                    if once and visited == len(self.items):
                        return
                    self.select(self.index+1)
                self.update_pose()
                self.draw()
                self.frame_times.append(time.monotonic())
                next_frame += 1/self.fps
                if next_frame < time.monotonic():
                    next_frame = time.monotonic()
                time.sleep(max(0., next_frame-time.monotonic()))


class GridGallery(Gallery):
    """Forty live, independently sampled objects on an 8 by 5 display wall."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, grid=True, **kwargs)
        self.round = 0

    def select(self, index=0):
        count = len(self.items)
        self.candidates = [[] for _ in self.items]
        self.batch_counts, self.latest = [0]*count, [0]*count
        self.arrivals, self.errors = [0.]*count, [None]*count
        self.elapsed, self.angle = 0., 0.
        self.generation += 1
        for geom in self.geoms:
            self.model.geom_rgba[geom, 3] = 1
        self.update_pose()
        if self.viewer:
            self.viewer.sync()
        print(f"8 x 5 gallery / {count} objects / {self.seconds:g}s per revolution", flush=True)

    def update_pose(self):
        self.data.mocap_pos[self.mocaps] = self.positions+[0, 0, .16]
        self.data.mocap_quat[self.mocaps] = [np.cos(self.angle/2), 0, 0, np.sin(self.angle/2)]
        mj.mj_forward(self.model, self.data)

    def add_grid_batch(self, index, batch):
        self.latest[index] = len(self.candidates[index])
        combined = unique_grasps([*self.candidates[index], *batch])
        self.latest[index] = max(0, self.latest[index]-max(0, len(combined)-400))
        self.candidates[index] = combined[-400:]
        self.arrivals[index] = time.monotonic()
        self.batch_counts[index] += 1
        print(f"[{index+1:02d}/40] {self.items[index].label}: batch {self.batch_counts[index]}/{self.batches}, "
              f"{len(self.candidates[index])} candidates", flush=True)

    def decorate(self, scene):
        for i in range(len(self.items)):
            self.decorate_item(scene, i, self.candidates[i], self.latest[i], self.arrivals[i])

    def cell_record(self, index):
        item = self.items[index]
        return {"asset": str(item.path.relative_to(ROOT)), "label": item.label,
                "source_extents_m": item.extents.tolist(), "display_scale": item.scale,
                "unique_candidates": len(self.candidates[index]), "batches": self.batch_counts[index],
                "max_opening_m": max((g["width"] for g in self.candidates[index]), default=None),
                "rotation_degrees": float(np.degrees(self.angle)), "error": self.errors[index]}

    def run(self, start=0, once=False):
        self.select()
        futures, cursor = {}, 0
        last, next_frame = time.monotonic(), time.monotonic()
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            while self.viewer.is_running():
                now = time.monotonic()
                dt, last = min(now-last, .1), now
                while not self.events.empty():
                    key = self.events.get()
                    if key in (81, 256):
                        return
                    if key == 32:
                        self.paused = not self.paused
                    elif key == 71:
                        self.show_grasps = not self.show_grasps
                    elif key in (61, 334, 45, 333):
                        self.visible = max(1, min(16, self.visible+(1 if key in (61, 334) else -1)))
                        self.visibility_cache.clear()
                    elif key == 82:
                        self.resamples += 1; self.select()
                if not self.paused:
                    self.elapsed += dt
                    self.angle = self.elapsed/self.seconds*2*np.pi
                for future, (generation, i) in list(futures.items()):
                    if not future.done():
                        continue
                    if generation == self.generation:
                        try:
                            self.add_grid_batch(i, future.result())
                        except Exception as error:
                            self.errors[i] = str(error)
                            self.batch_counts[i] = self.batches
                            print(f"{self.items[i].label}: {error}", file=sys.stderr, flush=True)
                    del futures[future]
                busy = {i for generation, i in futures.values() if generation == self.generation}
                while len(futures) < 2 and not self.paused:
                    eligible = [i for i in range(len(self.items)) if i not in busy
                                and self.batch_counts[i] < self.batches
                                and self.elapsed >= self.batch_counts[i]*self.seconds/self.batches]
                    if not eligible:
                        break
                    i = min(eligible, key=lambda j: (self.batch_counts[j], (j-cursor) % len(self.items)))
                    cursor = (i+1) % len(self.items)
                    future = pool.submit(sample_batch, str(self.items[i].path), self.batch_counts[i],
                                         self.batch_size, self.jaw_width,
                                         self.seed+i*101+(self.round+self.resamples)*100003)
                    futures[future] = (self.generation, i)
                    busy.add(i)
                if self.elapsed >= self.seconds and all(n == self.batches for n in self.batch_counts):
                    self.records.extend(self.cell_record(i) for i in range(len(self.items)))
                    if once:
                        return
                    self.round += 1
                    self.elapsed, self.angle = 0., 0.
                    self.batch_counts = [0]*len(self.items)
                self.update_pose()
                self.draw()
                self.frame_times.append(time.monotonic())
                next_frame += 1/self.fps
                if next_frame < time.monotonic():
                    next_frame = time.monotonic()
                time.sleep(max(0., next_frame-time.monotonic()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("grid", "turntable"), default="grid")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--jaw-width", type=float, default=.08)
    parser.add_argument("--visible-grasps", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir())/"grasp-sampler-gallery")
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in (args.seconds, args.fps, args.batches, args.batch_size,
                                    args.jaw_width, args.visible_grasps)):
        parser.error("Timing, sampling and jaw-width values must be positive")
    if args.jaw_width > FRANKA_MAX_OPENING_M:
        parser.error("The Franka gripper cannot open beyond 0.08 m (80 mm)")
    paths = inventory()
    print(f"Preparing {len(paths)} original-color object assets", flush=True)
    items = [prepare_item(path, args.cache) for path in paths]
    gallery_type = GridGallery if args.layout == "grid" else Gallery
    gallery = gallery_type(items, args.cache, fps=args.fps, seconds=args.seconds, batches=args.batches,
                      batch_size=args.batch_size, jaw_width=args.jaw_width, visible=args.visible_grasps,
                      seed=args.seed, gui=not args.headless)
    try:
        if args.headless:
            if args.layout == "grid":
                gallery.select()
                gallery.angle = 2*np.pi; gallery.update_pose()
            for i in range(len(items)):
                if args.layout == "turntable":
                    gallery.select(i)
                error = None
                try:
                    for batch in range(args.batches):
                        result = sample_batch(str(items[i].path), batch, args.batch_size,
                                              args.jaw_width, args.seed+i*101)
                        if args.layout == "grid":
                            gallery.add_grid_batch(i, result)
                        else:
                            gallery.add_batch(result)
                except Exception as caught:
                    error = str(caught)
                if args.layout == "grid":
                    gallery.errors[i] = error
                    gallery.records.append(gallery.cell_record(i))
                else:
                    gallery.angle = 2*np.pi; gallery.update_pose()
                    gallery.records.append(gallery.record(error))
        else:
            print("SPACE pause / G grasps / R resample / Q quit" +
                  (" / N next / P previous" if args.layout == "turntable" else ""), flush=True)
            gallery.run(args.start, args.once)
        if args.snapshot:
            if args.headless and args.layout == "turntable":
                gallery.select(args.start)
                for batch in range(args.batches):
                    gallery.add_batch(sample_batch(str(items[gallery.index].path), batch, args.batch_size,
                                                   args.jaw_width, args.seed+gallery.index*101))
                gallery.angle = np.pi/5; gallery.update_pose()
            if args.layout == "grid":
                gallery.angle = np.pi/6; gallery.update_pose()
            gallery.snapshot(args.snapshot)
    except KeyboardInterrupt:
        pass
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({"assets": gallery.records, "inventory_count": len(items),
                "jaw_width_m": args.jaw_width, "franka_max_opening_m": FRANKA_MAX_OPENING_M,
                "physics_execution": False, "layout": args.layout,
                "scene_update_hz": ((len(gallery.frame_times)-1)/(gallery.frame_times[-1]-gallery.frame_times[0])
                                    if len(gallery.frame_times) > 1 else None),
                "candidate_colors": {"violet": "primitive / OBB proposals", "cyan": "antipodal candidates",
                                     "gold": "newly generated candidates"}}, indent=2)+"\n")
        if gallery.viewer:
            gallery.viewer.close()
    if args.headless and any(record["error"] for record in gallery.records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
