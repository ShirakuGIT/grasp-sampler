"""Mounted FR3v2 workstation with visible grasp sampling, selection and execution.

Run with PYTHONPATH=src python examples/pick_place_pybullet.py --gui --hold
or --headless --report /tmp/pick-place.json for a reproducible physics check.
No grasp attachment constraints or object teleportation are used. This open
tabletop demo uses Cartesian waypoints + IK, rather than a collision-aware
whole-arm planner; replace ``move`` with cuRobo/MoveIt for cluttered scenes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation, Slerp
from workstation_scene import BASE_Z, TABLE_Z, Overlay, build_workstation, object_mesh

from grasp_sampler import GraspConfig, GraspSampler, ObjMesh, PlanningGrasps, tcp_to_ee
from grasp_sampler.integrations.pybullet import to_pybullet


class Demo:
    def __init__(self, gui=False, realtime=False, robot="fr3v2"):
        self.gui, self.realtime = gui, realtime
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
        floor = p.loadURDF("plane.urdf")
        p.changeVisualShape(floor, -1, textureUniqueId=-1, rgbaColor=[.12, .15, .19, 1])
        self.fixtures, self.place_surface = build_workstation()
        self.robot_name = robot
        urdf = (Path(__file__).resolve().parents[1] / "assets/robots/fr3v2/fr3v2.urdf"
                if robot == "fr3v2" else "franka_panda/panda.urdf")
        if robot == "fr3v2" and not urdf.exists():
            raise FileNotFoundError("Prepare the official FR3v2 first: python tools/prepare_fr3.py")
        self.robot = p.loadURDF(str(urdf), basePosition=[0, 0, BASE_Z], useFixedBase=True,
                                flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        info = {p.getJointInfo(self.robot, j)[1].decode(): p.getJointInfo(self.robot, j)
                for j in range(p.getNumJoints(self.robot))}
        prefix = "fr3v2" if robot == "fr3v2" else "panda"
        self.arm = [info[f"{prefix}_joint{i}"][0] for i in range(1, 8)]
        self.fingers = [info[f"{prefix}_finger_joint{i}"][0] for i in (1, 2)]
        self.tool_link = "fr3v2_hand_tcp" if robot == "fr3v2" else "panda_grasptarget"
        self.ee = next(j for j in range(p.getNumJoints(self.robot))
                       if p.getJointInfo(self.robot, j)[12].decode() == self.tool_link)
        self.arm_limits = np.array([p.getJointInfo(self.robot, j)[8:10] for j in self.arm])
        self.arm_forces = [p.getJointInfo(self.robot, j)[10] for j in self.arm]
        self.home = [0., -.4, 0., -2.2, 0., 1.8, .8]
        for j, q in zip(self.arm, self.home):
            p.resetJointState(self.robot, j, q)
        for j in self.fingers:
            p.resetJointState(self.robot, j, .04)
            p.changeDynamics(self.robot, j, lateralFriction=2., spinningFriction=.01)
        self.opening = .08
        self.grip(self.opening)
        p.resetDebugVisualizerCamera(1.9, 47, -32, [.43, .02, .91])
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self.overlay = Overlay()
        if robot == "panda":
            self.overlay.title("FRANKA  PANDA  /  GRASP LAB", "Tabletop integration example")
        self.sampler = GraspSampler(GraspConfig(
            approach_facets=((0., 0., -1.),), directed_spread=1,
            antipodal_samples=180, antipodal_max_grasps=80,
        ))

    def step(self, count):
        for _ in range(count):
            p.stepSimulation()
            if self.realtime:
                time.sleep(1 / 240)

    def grip(self, width, settle=0):
        self.opening = float(np.clip(width, 0, .08))
        for j in self.fingers:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=self.opening / 2, force=20, maxVelocity=.15)
        self.step(settle)

    def ee_pose(self):
        state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])

    def ik(self, position, quaternion):
        return p.calculateInverseKinematics(
            self.robot, self.ee, position, quaternion,
            maxNumIterations=160, residualThreshold=1e-6,
        )[:7]

    def move(self, position, quaternion, seconds=1.2):
        start, start_q = self.ee_pose()
        rotations = Slerp([0, 1], Rotation.from_quat([start_q, quaternion]))
        count = max(1, int(seconds * 240))
        for i in range(count):
            t = (i + 1) / count
            t = t * t * (3 - 2 * t)
            xyz = start * (1 - t) + np.asarray(position) * t
            q = rotations([t]).as_quat()[0]
            joints = self.ik(xyz.tolist(), q.tolist())
            p.setJointMotorControlArray(self.robot, self.arm, p.POSITION_CONTROL,
                                        targetPositions=joints, forces=self.arm_forces,
                                        positionGains=[.2] * 7)
            self.step(1)
        self.step(60)

    def spawn(self, count):
        objects = []
        colors = [[.95, .35, .25, 1], [.2, .55, .95, 1], [.95, .7, .15, 1],
                  [.55, .35, .9, 1], [.2, .8, .55, 1], [.95, .4, .7, 1]]
        for i in range(count):
            mesh, name = object_mesh(i)
            size = mesh.extents
            position = [.36 + .085 * (i % 3), -.24 + .11 * (i // 3), TABLE_Z + size[2] / 2]
            yaw = .18 * (i - 2)
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            vertices, indices = mesh.vertices.tolist(), mesh.faces.reshape(-1).tolist()
            collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
            visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=colors[i])
            body = p.createMultiBody(.06, collision, visual, position, quat)
            p.changeDynamics(body, -1, lateralFriction=1.4, spinningFriction=.002,
                             rollingFriction=.001, restitution=0)
            goal = np.array([.33 + .075 * i, .27, self.place_surface + size[2] / 2])
            marker = p.createVisualShape(p.GEOM_BOX, halfExtents=[.03, .03, .001],
                                         rgbaColor=[*colors[i][:3], .35])
            p.createMultiBody(0, -1, marker, [goal[0], goal[1], self.place_surface + .001])
            p.addUserDebugText(f"{i + 1:02d}", [goal[0] - .008, goal[1] + .05, self.place_surface + .006],
                               [.3, .9, .85], textSize=1.1)
            objects.append((body, mesh, goal, name))
        self.step(240)
        return objects

    def targets(self, body, mesh, seed):
        size = mesh.extents
        obj = ObjMesh(mesh, size, np.eye(4), size, "box")
        grasps = self.sampler.generate(obj, methods=("antipodal",), seed=seed)
        # Keep central horizontal contact pairs to leave clearance above the floor.
        grasps = [g for g in grasps if abs(g.pose[2, 3]) < .009 and g.pose[2, 2] < -.99]
        grasps.sort(key=lambda g: float(np.linalg.norm(g.pose[:3, 3])))
        position, quaternion = p.getBasePositionAndOrientation(body)
        world = np.eye(4)
        world[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        world[:3, 3] = position
        # The PyBullet Panda TCP closes along +Y; our sampler closes along +X.
        # Both configured TCP links already sit at the fingertips. In the FR3v2
        # description hand_tcp is 0.1034 m from the hand, rather than Panda's .105.
        calibration = tcp_to_ee(np.eye(4)[None], standoff=0)[0]
        targets = PlanningGrasps.from_grasps(
            grasps, world, frame_id="world", tool_link=self.tool_link,
            tcp_to_tool=calibration, approach_distance=.14, lift_offset=(0, 0, .22),
        )
        return targets, grasps, world

    def select_grasp(self, targets, bodies):
        """Screen IK, limits and fixture/object collisions, then minimize joint travel.

        This checks pregrasp, grasp, and four straight-approach samples. The
        overhead transfer remains an IK-waypoint demo, not a general planner.
        """
        state = p.saveState()
        start = np.array([p.getJointState(self.robot, j)[0] for j in self.arm])
        rejected, feasible = {}, []
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        try:
            for i in range(min(len(targets), 24)):
                p.restoreState(state)
                for joint in self.fingers:
                    p.resetJointState(self.robot, joint, .04)
                position, quaternion = to_pybullet(targets.select([i]))[0]
                pre = targets.pregrasp[i, :3, 3]
                reason = None
                for t in np.linspace(0, 1, 5):
                    xyz = (1 - t) * pre + t * np.array(position)
                    q = np.array(self.ik(xyz, quaternion))
                    if ((q < self.arm_limits[:, 0] - 1e-3) | (q > self.arm_limits[:, 1] + 1e-3)).any():
                        reason = "joint limits"; break
                    for joint, value in zip(self.arm, q):
                        p.resetJointState(self.robot, joint, value)
                    actual_position, actual_q = self.ee_pose()
                    angle = (Rotation.from_quat(actual_q).inv() * Rotation.from_quat(quaternion)).magnitude()
                    if np.linalg.norm(actual_position - xyz) > .004 or angle > .06:
                        reason = "IK residual"; break
                    p.performCollisionDetection()
                    if any(p.getClosestPoints(self.robot, other, distance=-.0005)
                           for other in [*self.fixtures, *bodies]):
                        reason = "scene collision"; break
                if reason:
                    rejected[i] = reason
                else:
                    cost = float(np.linalg.norm(q - start) + 12 * np.linalg.norm(
                        targets.grasp[i, :3, 3] - np.mean(targets.grasp[:, :3, 3], axis=0)))
                    feasible.append((cost, i))
        finally:
            p.restoreState(state)
            p.removeState(state)
            if self.gui:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        if not feasible:
            raise RuntimeError(f"No feasible sampled grasp: {rejected}")
        cost, selected = min(feasible)
        return selected, rejected, cost

    def run(self, count):
        results = []
        objects = self.spawn(count)
        for i, (body, mesh, goal, name) in enumerate(objects):
            size = mesh.extents
            self.overlay.phase(f"SAMPLING  /  {name.upper()}", i, count)
            targets, grasps, world = self.targets(body, mesh, i)
            if not len(targets):
                raise RuntimeError(f"No sampled top-down antipodal grasps for object {i}")
            self.overlay.candidates(mesh, grasps, world, None, {}, None)
            if self.gui:
                self.step(180)
            self.overlay.phase("SCREENING  /  IK + COLLISION", i, count)
            selected, rejected, cost = self.select_grasp(targets, [o[0] for o in objects])
            target = targets.select([selected])
            self.overlay.candidates(mesh, grasps, world, selected, rejected, cost)
            self.overlay.path(target)
            self.overlay.phase("SELECTED  /  IK + COLLISION SCREEN", i, count)
            if self.gui:
                self.step(360)
            pre, grasp, lift = [to_pybullet(target, stage=s)[0] for s in ("pregrasp", "grasp", "lift")]
            print(f"Object {i + 1}/{count}: {len(targets)} sampled candidates; "
                  f"selected #{selected}, {len(rejected)} rejected, width {target.widths[0]:.3f} m", flush=True)
            self.grip(.08)
            self.overlay.phase("APPROACH", i, count)
            transit = np.array(pre[0]); transit[2] = TABLE_Z + .36
            self.move(transit, pre[1], 1.5)
            self.move(*pre, seconds=.8)
            self.move(*grasp, seconds=1.)
            self.overlay.phase("CLOSE  /  CONTACT", i, count)
            self.grip(0., settle=180)
            contacts = [bool(p.getContactPoints(self.robot, body, linkIndexA=j)) for j in self.fingers]
            self.overlay.phase("LIFT  /  VERIFY", i, count)
            self.move(*lift, seconds=1.2)
            lifted_height = p.getBasePositionAndOrientation(body)[0][2]
            grasped = all(contacts) and lifted_height > TABLE_Z + size[2] / 2 + .10
            # Preserve the object-to-tool offset while translating to its destination.
            initial = np.array(p.getBasePositionAndOrientation(body)[0])
            destination = np.array(lift[0]) + goal - initial
            destination[2] += .012  # release just above the floor
            self.overlay.phase("TRANSFER  /  SORT", i, count)
            overhead = destination.copy(); overhead[2] = TABLE_Z + .36
            self.move(overhead, grasp[1], 1.5)
            self.move(destination, grasp[1], 1.2)
            # Correct residual tracking/slip from the measured held-object pose.
            # This moves the robot physically; object poses are never reset.
            current = np.array(p.getBasePositionAndOrientation(body)[0])
            correction = goal + [0, 0, .012] - current
            if grasped and np.linalg.norm(correction) > .003:
                xyz, q = self.ee_pose()
                self.move(xyz + correction, q, .7)
            self.overlay.phase("PLACE  /  RELEASE", i, count)
            self.grip(.08, settle=180)
            self.move(overhead, grasp[1], .9)
            self.step(120)
            final = np.array(p.getBasePositionAndOrientation(body)[0])
            error = float(np.linalg.norm(final[:2] - goal[:2]))
            placed = bool(grasped and error < .04 and final[2] < self.place_surface + size[2] + .02)
            record = {"object": i, "shape": name, "sampled_candidates": len(targets),
                      "candidate_id": int(target.candidate_ids[0]), "rejected": rejected, "planning_cost": cost,
                      "jaw_width": float(target.widths[0]), "bilateral_contact": all(contacts),
                      "lifted_height": lifted_height, "goal": goal.tolist(), "final_position": final.tolist(),
                      "placement_error_xy": error, "success": placed}
            results.append(record)
            print(f"  bilateral contact={all(contacts)}, lift={lifted_height:.3f} m, "
                  f"placement error={error:.3f} m, success={placed}", flush=True)
        self.overlay.phase("COMPLETE  /  OBJECTS SORTED", count - 1, count)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true")
    mode.add_argument("--headless", action="store_true")
    parser.add_argument("--objects", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--robot", choices=["fr3v2", "panda"], default="fr3v2")
    parser.add_argument("--hold", action="store_true", help="keep GUI open after finishing")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--snapshot", type=Path, help="save a rendered workstation image")
    parser.add_argument("--realtime", action="store_true", help="pace headless simulation too")
    args = parser.parse_args()
    demo = Demo(gui=args.gui, realtime=args.gui or args.realtime, robot=args.robot)
    try:
        results = demo.run(args.objects)
        report = {"robot": args.robot, "objects": results, "successes": sum(r["success"] for r in results),
                  "total": len(results), "attachment_constraints": False}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Placed {report['successes']}/{report['total']} objects", flush=True)
        if args.snapshot:
            from PIL import Image
            view = p.computeViewMatrixFromYawPitchRoll([.43, .02, .91], 1.9, 47, -32, 0, 2)
            projection = p.computeProjectionMatrixFOV(48, 16 / 10, .05, 5)
            frame = p.getCameraImage(1600, 1000, view, projection, shadow=1,
                                     lightDirection=[-2, -3, 5], renderer=p.ER_TINY_RENDERER)[2]
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(frame)[:, :, :3]).save(args.snapshot)
        if args.gui and args.hold:
            print("Demo finished. Close the PyBullet window to exit.", flush=True)
            while p.isConnected():
                demo.step(1)
        if report["successes"] != report["total"]:
            raise SystemExit(1)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
