"""Native MuJoCo FR3v2 workstation with grasp sampling, selection and physical picks.

python tools/prepare_fr3_mujoco.py
mjpython examples/pick_place_mujoco.py --gui --hold   # macOS viewer
python examples/pick_place_mujoco.py --headless --report /tmp/mujoco-picks.json

Uses lateral/oblique antipodal grasps, named-site Jacobian IK, RRT-Connect,
native joint actuators, and frictional finger contact. No object attachment or
teleportation. The bottle goes from the lower bay to the upper tier; the other
existing mesh assets are extracted and reoriented for tabletop placement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Open3D and MuJoCo ship conflicting GLFW Objective-C classes on macOS.
# Use the package's existing trimesh collision backend in this viewer process.
if sys.platform == "darwin":
    os.environ.setdefault("GRASP_SAMPLER_DISABLE_OPEN3D", "1")

from mujoco_workstation_scene import BASE_Z, SHELF_Z, TABLE_Z, UPPER_Z, build_scene  # noqa: E402

from grasp_sampler import GraspConfig, GraspSampler, ObjMesh, PlanningGrasps, tcp_to_ee  # noqa: E402
from grasp_sampler.integrations.mujoco import solve_site_ik  # noqa: E402


class Workstation:
    def __init__(self, count=3, gui=False):
        xml, self.objects, fixtures = build_scene(count)
        self.model = mj.MjModel.from_xml_string(xml)
        self.data = mj.MjData(self.model)
        self.scratch = mj.MjData(self.model)
        self.count, self.gui, self.viewer = count, gui, None
        self.joint_names = tuple(f"fr3v2_joint{i}" for i in range(1, 8))
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        self.q_indices = self.model.jnt_qposadr[self.joint_ids]
        self.dofs = self.model.jnt_dofadr[self.joint_ids]
        self.actuators = [self.model.actuator(name).id for name in self.joint_names]
        self.gripper = self.model.actuator("gripper_actuator8").id
        self.site = self.model.site("hand_tcp").id
        self.body_ids = [self.model.body(f"item_{i}").id for i in range(count)]
        self.finger_ids = {self.model.body(f"gripper_{side}_finger").id for side in ("left", "right")}
        self.robot_bodies = {self.model.body("base").id}
        for i in range(self.model.nbody):
            parent = i
            while parent > 0:
                if parent in self.robot_bodies:
                    self.robot_bodies.add(i); break
                parent = self.model.body_parentid[parent]
        self.fixture_ids = {self.model.geom(name).id for name in fixtures}
        self.home = np.array([0., -.8, 0., -2.3, 0., 1.6, .8])
        self.data.qpos[self.q_indices] = self.home
        self.data.ctrl[self.actuators] = self.home
        for name in ("gripper_finger_joint1", "gripper_finger_joint2"):
            self.data.qpos[self.model.joint(name).qposadr[0]] = .04
        self.data.ctrl[self.gripper] = 255
        mj.mj_forward(self.model, self.data)
        self.sampler = GraspSampler(GraspConfig(approach_facets=((1, 0, 0),),
                                    directed_spread=3, cone_deg=28, antipodal_samples=600, antipodal_max_grasps=600,
                                    antipodal_min_width=.01))
        self.phase, self.active, self.completed = "INITIALIZING", 0, 0
        self.held_index = None
        self.planning_calls = 0
        self.grasps, self.world, self.selected = [], np.eye(4), None
        self.pick_path = []
        self.rejected, self.costs, self.active_target = {}, {}, None
        self.figure = mj.MjvFigure()
        mj.mjv_defaultFigure(self.figure)
        self.figure.title = "GRASP SELECTION / travel + centering cost"
        self.figure.xlabel = "Candidate ID (lower cost is better)"
        self.figure.flg_extend = 0
        self.figure.figurergba[:] = [.035, .055, .085, .94]
        self.figure.panergba[:] = [.04, .065, .095, 1]
        if gui:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                       show_left_ui=False, show_right_ui=False)
            self.viewer.cam.lookat[:] = [.44, .02, .99]
            self.viewer.cam.distance = 1.9
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -28
            self.viewer.opt.geomgroup[3] = False  # native collision proxies remain hidden
            self.viewer.opt.geomgroup[4] = False
        self.step(.8)

    def step(self, seconds):
        start = time.monotonic()
        count = max(1, int(seconds / self.model.opt.timestep))
        for k in range(count):
            # Gravity/coriolis feedforward on robot joints; contact and motors
            # remain native dynamics, with the model's actuator force limits.
            self.data.qfrc_applied[self.dofs] = self.data.qfrc_bias[self.dofs]
            mj.mj_step(self.model, self.data)
            if self.viewer and k % 8 == 0:
                if not self.viewer.is_running():
                    raise KeyboardInterrupt
                self.update_viewer()
                delay = start + (k + 1) * self.model.opt.timestep - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

    def pose(self):
        out = np.eye(4)
        out[:3, 3] = self.data.site_xpos[self.site]
        out[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        return out

    def move(self, target, seconds=1.2):
        start = self.pose()
        rotation = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
        seed = self.data.qpos[self.q_indices].copy()
        count = max(1, round(seconds / .02))
        for i in range(count):
            t = (i + 1) / count
            t = t * t * (3 - 2 * t)
            pose = np.eye(4)
            pose[:3, 3] = (1 - t) * start[:3, 3] + t * target[:3, 3]
            pose[:3, :3] = rotation([t]).as_matrix()[0]
            result = solve_site_ik(self.model, self.data, "hand_tcp", pose, self.joint_names,
                                   seed=seed, max_iterations=120, position_tolerance=.0004,
                                   orientation_tolerance=.01, damping=.02)
            if not result.success and result.position_error > .006:
                raise RuntimeError(f"IK failed during {self.phase}: {result.position_error:.4f}m")
            seed = result.positions
            self.data.ctrl[self.actuators] = seed
            self.step(seconds / count)
        self.step(.25)

    def move_joint(self, target, seconds=1.2):
        """Collision-checked joint transit, with the held mesh in the query."""
        result = solve_site_ik(self.model, self.data, "hand_tcp", target, self.joint_names,
                               max_iterations=200, damping=.02)
        if not result.success and result.position_error > .004:
            raise RuntimeError(f"Transit IK failed during {self.phase}: {result.position_error:.4f}m")
        start = self.data.qpos[self.q_indices].copy()
        path = self.joint_path(start, result.positions)
        total = sum(np.linalg.norm(b-a) for a, b in zip(path, path[1:]))
        for a, b in zip(path, path[1:]):
            duration = max(seconds*np.linalg.norm(b-a)/max(total, 1e-8), np.max(np.abs(b-a))/.65, .08)
            count = max(1, round(duration/.02))
            for i in range(count):
                t = (i+1)/count; t = t*t*(3-2*t)
                self.data.ctrl[self.actuators] = (1-t)*a+t*b
                self.step(duration/count)
        self.step(.3)

    def joint_path(self, start, goal):
        """Small bidirectional RRT-Connect fallback, discretized at .04 rad.

        Queries only scratch data. The physical object is never attached or
        repositioned; its measured tool-relative pose is used for planning.
        """
        scratch = mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        held, relative, free_address = None, None, None
        if self.held_index is not None:
            held = self.body_ids[self.held_index]
            obj = np.eye(4); obj[:3, 3] = self.data.xpos[held]
            obj[:3, :3] = self.data.xmat[held].reshape(3, 3)
            relative = np.linalg.inv(self.pose()) @ obj
            free_address = self.model.joint(f"item_{self.held_index}_free").qposadr[0]
        gripper_bodies = self.finger_ids | {self.model.body("gripper_hand").id}

        def valid(q):
            scratch.qpos[self.q_indices] = q
            if held is not None:
                mj.mj_kinematics(self.model, scratch)
                tool = np.eye(4); tool[:3, 3] = scratch.site_xpos[self.site]
                tool[:3, :3] = scratch.site_xmat[self.site].reshape(3, 3)
                obj = tool @ relative
                scratch.qpos[free_address:free_address+3] = obj[:3, 3]
                quat = Rotation.from_matrix(obj[:3, :3]).as_quat()
                scratch.qpos[free_address+3:free_address+7] = quat[[3, 0, 1, 2]]
            mj.mj_forward(self.model, scratch)
            for contact in scratch.contact[:scratch.ncon]:
                if contact.dist >= -.001:
                    continue
                a, b = map(int, self.model.geom_bodyid[[contact.geom1, contact.geom2]])
                if held is not None and ((a == held and b in gripper_bodies)
                                          or (b == held and a in gripper_bodies)):
                    continue
                if a in self.robot_bodies or b in self.robot_bodies or a == held or b == held:
                    return False
            return True

        def edge(a, b):
            count = max(1, int(np.ceil(np.max(np.abs(b-a))/.04)))
            return all(valid(a+(b-a)*t) for t in np.linspace(0, 1, count+1))

        if not valid(start):
            raise RuntimeError(f"Collision at transit start during {self.phase}")
        if not valid(goal):
            raise RuntimeError(f"Collision at transit goal during {self.phase}")
        if edge(start, goal):
            return [start, goal]
        self.planning_calls += 1
        rng = np.random.default_rng(77+self.planning_calls)
        limits = self.model.jnt_range[self.joint_ids]
        trees = [([start], [-1]), ([goal], [-1])]
        deadline = time.monotonic()+12

        def extend(tree, target):
            nodes, parents = tree
            near = int(np.argmin([np.linalg.norm(n-target) for n in nodes]))
            delta = target-nodes[near]
            distance = np.linalg.norm(delta)
            q = nodes[near]+delta*min(1., .30/max(distance, 1e-12))
            if not edge(nodes[near], q):
                return None
            nodes.append(q); parents.append(near)
            return len(nodes)-1

        def ancestry(tree, index):
            out = []
            while index >= 0:
                out.append(tree[0][index]); index = tree[1][index]
            return out[::-1]

        for iteration in range(1200):
            if time.monotonic() > deadline:
                break
            side = iteration % 2
            sample = trees[1-side][0][-1] if rng.random() < .25 else rng.uniform(limits[:, 0], limits[:, 1])
            index = extend(trees[side], sample)
            if index is None:
                continue
            point = trees[side][0][index]
            for _ in range(40):
                other = extend(trees[1-side], point)
                if other is None:
                    break
                if np.linalg.norm(trees[1-side][0][other]-point) < 1e-6:
                    left, right = ancestry(trees[side], index), ancestry(trees[1-side], other)
                    path = left+right[-2::-1] if side == 0 else right+left[-2::-1]
                    # Shortcut only when the held-object collision check passes.
                    for _ in range(40):
                        if len(path) < 3:
                            break
                        a, b = sorted(rng.choice(len(path), 2, replace=False))
                        if b > a+1 and edge(path[a], path[b]):
                            path = path[:a+1]+path[b:]
                    print(f"  RRT-Connect: {len(path)} collision-checked waypoints", flush=True)
                    return path
        raise RuntimeError(f"No collision-free transit found during {self.phase}")

    def grip(self, opening, settle=.65):
        self.data.ctrl[self.gripper] = float(np.clip(opening / .08, 0, 1) * 255)
        self.step(settle)

    def make_targets(self, index):
        mesh, _, _ = self.objects[index]
        world = np.eye(4)
        world[:3, 3] = self.data.xpos[self.body_ids[index]]
        world[:3, :3] = self.data.xmat[self.body_ids[index]].reshape(3, 3)
        # Approach facets belong to the object's mesh frame. Re-express the
        # frontal and oblique world directions after physical settling.
        world_facets = np.array([[1., 0., 0.], [.94, .25, .2], [.94, -.25, -.2]])
        local_facets = world_facets @ world[:3, :3]
        sampler = GraspSampler(replace(self.sampler.config, approach_facets=tuple(map(tuple, local_facets))))
        # MuJoCo uses the dynamic mesh's convex hull for contact. Match that
        # geometry for these decimated, occasionally open scan meshes.
        contact_mesh = mesh.convex_hull
        grasps = sampler.generate(ObjMesh(contact_mesh, mesh.extents, np.eye(4), mesh.extents, "irregular"),
                                  methods=("antipodal",), seed=index)
        grasps = [g for g in grasps if abs(g.pose[2, 3]) < mesh.extents[2] * .40
                  and (world[:3, :3] @ g.pose[:3, 2])[0] > .75
                  and abs((world[:3, :3] @ g.pose[:3, 0])[2]) < .35]
        grasps.sort(key=lambda g: float(np.linalg.norm(g.pose[:3, 3])))
        calibration = tcp_to_ee(np.eye(4)[None], standoff=0)[0]
        targets = PlanningGrasps.from_grasps(grasps, world, frame_id="world", tool_link="hand_tcp",
                                             tcp_to_tool=calibration, approach_distance=.15,
                                             lift_offset=(-.20, 0, .15))
        return targets, grasps, world

    def screen(self, targets):
        rejected, costs = {}, {}
        self.scratch.qpos[:] = self.data.qpos
        for name in ("gripper_finger_joint1", "gripper_finger_joint2"):
            self.scratch.qpos[self.model.joint(name).qposadr[0]] = .04
        start = self.data.qpos[self.q_indices].copy()
        for i in np.linspace(0, len(targets)-1, min(len(targets), 32), dtype=int):
            i = int(i)
            seed, reason = start.copy(), None
            for t in np.linspace(0, 1, 5):
                pose = targets.grasp[i].copy()
                pose[:3, 3] = (1 - t) * targets.pregrasp[i, :3, 3] + t * targets.grasp[i, :3, 3]
                result = solve_site_ik(self.model, self.scratch, "hand_tcp", pose, self.joint_names, seed=seed)
                if not result.success:
                    reason = "IK / joint limits"; break
                seed = result.positions
                self.scratch.qpos[self.q_indices] = seed
                mj.mj_forward(self.model, self.scratch)
                for contact in self.scratch.contact[:self.scratch.ncon]:
                    a, b = int(contact.geom1), int(contact.geom2)
                    ba, bb = self.model.geom_bodyid[[a, b]]
                    robot_a, robot_b = ba in self.robot_bodies, bb in self.robot_bodies
                    if contact.dist < -.0005 and (robot_a or robot_b):
                        reason = "scene / self collision"; break
                if reason:
                    break
            if reason is None and self.active == 0:
                # A grasp must work at both tiers, not merely at pickup.
                place = targets.grasp[i].copy()
                place[2, 3] += UPPER_Z - SHELF_Z + .012
                placement = solve_site_ik(self.model, self.scratch, "hand_tcp", place,
                                           self.joint_names, seed=seed, max_iterations=150)
                if not placement.success:
                    reason = "upper-tier IK"
                else:
                    self.scratch.qpos[self.q_indices] = placement.positions
                    mj.mj_forward(self.model, self.scratch)
                    for contact in self.scratch.contact[:self.scratch.ncon]:
                        a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                        if contact.dist < -.0005 and (a in self.robot_bodies or b in self.robot_bodies):
                            reason = "upper-tier collision"; break
            if reason:
                rejected[i] = reason
            else:
                costs[i] = float(np.linalg.norm(seed - start) + 20 * np.linalg.norm(self.grasps[i].pose[:3, 3]))
        if not costs:
            raise RuntimeError(f"No feasible sampled grasp: {rejected}")
        return min(costs, key=costs.get), rejected, costs

    def stage(self, name):
        self.phase = name
        print(f"[{self.active + 1}/{self.count}] {name}", flush=True)
        if self.viewer:
            self.update_viewer()

    def contacts(self, index):
        touched = set()
        body = self.body_ids[index]
        for contact in self.data.contact[:self.data.ncon]:
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if a == body and b in self.finger_ids:
                touched.add(int(b))
            elif b == body and a in self.finger_ids:
                touched.add(int(a))
        return len(touched) == 2

    def placement_status(self, index, goal):
        """Verify released, load-bearing support on the requested tier/table.

        Orientation is free: a bottle resting on its side can satisfy this
        retrieval task, while a bottle on the wrong shelf cannot.
        """
        mj.mj_forward(self.model, self.data)
        body = self.body_ids[index]
        names = ("upper_tier", "upper_tier_edge") if index == 0 else ("table",)
        supports = {self.model.geom(name).id for name in names}
        gripper_bodies = self.finger_ids | {self.model.body("gripper_hand").id}
        supported, released, normal_force = False, True, 0.
        for number, contact in enumerate(self.data.contact[:self.data.ncon]):
            ga, gb = int(contact.geom1), int(contact.geom2)
            ba, bb = map(int, self.model.geom_bodyid[[ga, gb]])
            if body not in (ba, bb):
                continue
            other_body, other_geom = (bb, gb) if ba == body else (ba, ga)
            if other_body in gripper_bodies:
                released = False
            if other_geom in supports and abs(contact.frame[2]) > .7:
                force = np.zeros(6)
                mj.mj_contactForce(self.model, self.data, number, force)
                normal_force += max(0., float(force[0]))
                supported |= force[0] > .01
        mesh = self.objects[index][0]
        position = self.data.xpos[body].copy()
        rotation = self.data.xmat[body].reshape(3, 3)
        floor = UPPER_Z if index == 0 else TABLE_Z
        bottom = float((mesh.vertices @ rotation.T + position)[:, 2].min())
        error = float(np.linalg.norm(position[:2]-goal[:2]))
        settled = (np.linalg.norm(self.data.cvel[body, 3:]) < .08
                   and np.linalg.norm(self.data.cvel[body, :3]) < .5)
        passed = bool(supported and released and settled and abs(bottom-floor) < .015 and error < .04)
        return passed, {"support_surface": names[0], "supported": bool(supported),
                        "released": released, "settled": bool(settled), "support_normal_force": normal_force,
                        "object_bottom_height": bottom, "placement_error_xy": error}

    def run(self):
        records = []
        for i, (_, label, goal) in enumerate(self.objects):
            initial_height = float(self.data.xpos[self.body_ids[i], 2])
            self.active, self.selected, self.rejected, self.costs = i, None, {}, {}
            self.active_target = None
            self.pick_path = []
            self.stage(f"SAMPLING / {label.upper()}")
            targets, self.grasps, self.world = self.make_targets(i)
            if not len(targets):
                raise RuntimeError(f"No antipodal candidates for item {i}")
            if self.viewer:
                self.step(1.2)
            self.stage("IK + COLLISION SCREEN")
            self.selected, self.rejected, self.costs = self.screen(targets)
            chosen = targets.select([self.selected])
            self.active_target = chosen
            unseat = chosen.grasp[0].copy(); unseat[2, 3] += .015
            retreat = chosen.pregrasp[0].copy(); retreat[2, 3] = unseat[2, 3]
            lift = retreat.copy(); lift[2, 3] += .14 if i == 0 else .20
            if i > 0:
                down_tcp = np.diag([1., -1., -1., 1.])
                lift[:3, :3] = tcp_to_ee(down_tcp[None], standoff=0)[0, :3, :3]
            self.pick_path = [chosen.pregrasp[0], chosen.grasp[0], unseat, retreat, lift]
            self.stage(f"SELECTED #{self.selected} / {len(self.rejected)} REJECTED")
            if self.viewer:
                self.step(1.5)
            self.grip(.08, .15)
            self.stage("APPROACH")
            self.move_joint(chosen.pregrasp[0], 1.5)
            self.move(chosen.grasp[0], .9)
            self.stage("CLOSE / BILATERAL CONTACT")
            self.grip(0.)
            contact = self.contacts(i)
            print(f"  bilateral finger contact: {contact}", flush=True)
            if not contact:
                raise RuntimeError("No bilateral grasp contact; refusing to report a successful pick")
            self.held_index = i
            self.stage("EXTRACT FROM SHELF")
            self.move(unseat, .5)
            self.move(retreat, 1.2)
            self.stage("LIFT / VERIFY" if i == 0 else "LIFT / REORIENT")
            self.move_joint(lift, 1.1)
            lifted = float(self.data.xpos[self.body_ids[i], 2])
            grasped = contact and lifted > initial_height + .08
            self.stage("TRANSFER TO UPPER TIER" if i == 0 else "TRANSFER TO TABLETOP")
            # Place the current physical orientation on the table. This also
            # handles naturally settled, horizontal scan assets such as markers.
            mesh = self.objects[i][0]
            actual_object = np.eye(4)
            actual_object[:3, :3] = self.data.xmat[self.body_ids[i]].reshape(3, 3)
            actual_object[:3, 3] = self.data.xpos[self.body_ids[i]]
            relative = np.linalg.inv(self.pose()) @ actual_object
            destination = self.pose()
            if i == 0:
                destination[:3, :3] = chosen.grasp[0, :3, :3]
            object_rotation = destination[:3, :3] @ relative[:3, :3]
            goal = goal.copy()
            floor = UPPER_Z if i == 0 else TABLE_Z
            goal[2] = floor - float((mesh.vertices @ object_rotation.T)[:, 2].min())
            clearance = .003 if i == 0 else .012
            destination[:3, 3] = goal + [0, 0, clearance] - destination[:3, :3] @ relative[:3, 3]
            overhead = destination.copy()
            if i == 0:
                overhead[0, 3] -= .15
            else:
                overhead[2, 3] = max(TABLE_Z+.36, destination[2, 3]+.12)
            self.move_joint(overhead, 1.5)
            self.stage("PLACE / RELEASE")
            self.move(destination, 1.)
            correction = goal + [0, 0, clearance] - self.data.xpos[self.body_ids[i]]
            if grasped and np.linalg.norm(correction) > .003:
                corrected = self.pose(); corrected[:3, 3] += correction
                self.move(corrected, .6)
            self.grip(.08, .6)
            self.held_index = None
            self.move(overhead, .8)
            self.step(.8)
            placed, support = self.placement_status(i, goal)
            if not support["settled"]:
                self.step(1.)
                placed, support = self.placement_status(i, goal)
            final = self.data.xpos[self.body_ids[i]].copy()
            error = support["placement_error_xy"]
            success = bool(grasped and placed)
            self.completed += int(success)
            angle = float(np.degrees(np.arccos(np.clip(
                self.world[:3, :3] @ self.grasps[self.selected].pose[:3, 2] @ np.array([1., 0., 0.]), -1, 1))))
            records.append({"object": i, "shape": label, "sampled_candidates": len(targets),
                "destination": "upper tier" if i == 0 else "tabletop", "approach_obliquity_deg": angle,
                "selected_candidate": self.selected, "rejected": self.rejected, "costs": self.costs,
                "bilateral_contact": contact, "lifted_height": lifted, "goal": goal.tolist(),
                "final_position": final.tolist(), **support, "success": success})
            print(f"  contact={contact}, lift={lifted-BASE_Z:.3f}m above base, "
                  f"placement_error={error:.4f}m, success={success}", flush=True)
        self.stage(f"COMPLETE / {self.completed} OF {self.count} RETRIEVED")
        return {"simulator": "MuJoCo", "version": mj.__version__, "robot": "FR3v2",
                "objects": records, "successes": self.completed, "total": self.count,
                "attachment_constraints": False, "rrt_connect_queries": self.planning_calls,
                "joint_collision_resolution_rad": .04, "placement_orientation": "free"}

    def decorate(self, scene):
        def capsule(a, b, rgba, width=.001):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                            np.asarray(rgba, np.float32))
            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a), np.asarray(b))
            scene.ngeom += 1
        def glyphs(world, scale):
            indices = np.linspace(0, len(self.grasps) - 1, min(12, len(self.grasps)), dtype=int).tolist()
            if self.selected is not None:
                indices = list(dict.fromkeys([*indices, self.selected]))
            for i in indices:
                g = self.grasps[i]
                pose = g.pose.copy(); pose[:3, 3] *= scale
                pose = world @ pose
                half, depth = g.width * scale / 2, .04 * scale
                points = np.array([[half, 0, 0], [half, 0, -depth], [-half, 0, 0],
                                   [-half, 0, -depth], [0, 0, -depth], [0, 0, -depth-.035*scale]])
                points = points @ pose[:3, :3].T + pose[:3, 3]
                color = ([1, .72, .12, 1] if i == self.selected else [.95, .26, .27, .65]
                         if i in self.rejected else [.1, .8, .95, .4])
                for a, b in [(0, 1), (2, 3), (1, 3), (4, 5)]:
                    capsule(points[a], points[b], color, (.0025 if i == self.selected else .0008)*scale)
        if self.grasps:
            glyphs(self.world, 1.)
        if self.pick_path:
            for a, b in zip(self.pick_path, self.pick_path[1:]):
                capsule(a[:3, 3], b[:3, 3], [1, .66, .1, .9], .002)

    def update_viewer(self):
        if not self.viewer:
            return
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            self.decorate(self.viewer.user_scn)
        if hasattr(self.viewer, "set_texts"):
            reasons = ", ".join(f"{n} {reason}" for reason, n in Counter(self.rejected.values()).items())
            self.viewer.set_texts([
                (mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT,
                 "FRANKA FR3v2  /  GRASP LAB\n\nObject\nPhase\nCandidates\nRejected\nSelected\nPlaced",
                 f"MuJoCo native workstation\n\n{self.active+1:02d} / {self.count:02d}\n{self.phase}\n"
                 f"{len(self.grasps)}\n{len(self.rejected)}\n{self.selected if self.selected is not None else '-'}\n"
                 f"{self.completed} / {self.count}"),
                (mj.mjtFontScale.mjFONTSCALE_100, mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "CYAN  sampled candidates\nCORAL  rejected by IK / collision\nGOLD  selected grasp + approach\n\n"
                 "Physical fingers / no object attachments", reasons),
            ])
        if self.costs and hasattr(self.viewer, "set_figures"):
            self.figure.linepnt[:] = 0
            for line, (i, cost) in enumerate(self.costs.items()):
                self.figure.linepnt[line] = 2
                self.figure.linedata[line, :4] = [i, 0, i, cost]
                self.figure.linergb[line] = [1, .72, .12] if i == self.selected else [.1, .8, .95]
            self.figure.range[0] = [-1, max(self.costs)+1]
            self.figure.range[1] = [0, max(self.costs.values()) * 1.15]
            width = max(self.viewer.viewport.width, 1280)
            self.viewer.set_figures((mj.MjrRect(width-390, 20, 370, 190), self.figure))
        self.viewer.sync()

    def snapshot(self, path):
        from PIL import Image
        with mj.Renderer(self.model, height=1000, width=1600) as renderer:
            camera = mj.MjvCamera()
            camera.lookat[:] = [.44, .02, .99]
            camera.distance, camera.azimuth, camera.elevation = 1.9, 45, -28
            opt = mj.MjvOption(); opt.geomgroup[3:5] = False
            renderer.update_scene(self.data, camera=camera, scene_option=opt)
            self.decorate(renderer.scene)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true")
    mode.add_argument("--headless", action="store_true")
    parser.add_argument("--objects", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    demo = Workstation(args.objects, gui=args.gui)
    try:
        report = demo.run()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        if args.snapshot:
            demo.snapshot(args.snapshot)
        if args.gui and args.hold:
            while demo.viewer.is_running():
                demo.step(.05)
        if report["successes"] != report["total"]:
            raise SystemExit(1)
    except KeyboardInterrupt:
        pass
    finally:
        if demo.viewer:
            demo.viewer.close()


if __name__ == "__main__":
    main()
