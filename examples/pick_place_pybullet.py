"""Panda picks multiple objects using sampled antipodal grasps and physical fingers.

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
import trimesh
from scipy.spatial.transform import Rotation, Slerp

from grasp_sampler import GraspConfig, GraspSampler, ObjMesh, PlanningGrasps, tcp_to_ee
from grasp_sampler.integrations.pybullet import to_pybullet


class Demo:
    def __init__(self, gui=False, realtime=False):
        self.gui, self.realtime = gui, realtime
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.arm = list(range(7))
        self.fingers = [9, 10]
        self.ee = 11  # panda_grasptarget, 0.105 m beyond the hand
        self.home = [0., -.4, 0., -2.2, 0., 1.8, .8]
        for j, q in zip(self.arm, self.home):
            p.resetJointState(self.robot, j, q)
        for j in self.fingers:
            p.resetJointState(self.robot, j, .04)
            p.changeDynamics(self.robot, j, lateralFriction=2., spinningFriction=.01)
        self.opening = .08
        self.grip(self.opening)
        p.resetDebugVisualizerCamera(1.35, 45, -38, [.45, 0., .10])
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
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
                                    targetPosition=self.opening / 2, force=35, maxVelocity=.15)
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
                                        targetPositions=joints, forces=[160] * 7,
                                        positionGains=[.2] * 7)
            self.step(1)
        self.step(60)

    def spawn(self, count):
        objects = []
        colors = [[.95, .35, .25, 1], [.2, .55, .95, 1], [.95, .7, .15, 1],
                  [.55, .35, .9, 1], [.2, .8, .55, 1], [.95, .4, .7, 1]]
        for i in range(count):
            size = np.array([.034 + .003 * (i % 3), .036 + .004 * (i % 2), .05])
            position = [.36 + .085 * (i % 3), -.24 + .11 * (i // 3), size[2] / 2]
            yaw = .18 * (i - 2)
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(size / 2).tolist())
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(size / 2).tolist(), rgbaColor=colors[i])
            body = p.createMultiBody(.06, collision, visual, position, quat)
            p.changeDynamics(body, -1, lateralFriction=1.4, spinningFriction=.002,
                             rollingFriction=.001, restitution=0)
            goal = np.array([.34 + .075 * i, .23, size[2] / 2])
            marker = p.createVisualShape(p.GEOM_BOX, halfExtents=[.03, .03, .001],
                                         rgbaColor=[*colors[i][:3], .35])
            p.createMultiBody(0, -1, marker, [goal[0], goal[1], .001])
            objects.append((body, size, goal))
        self.step(240)
        return objects

    def targets(self, body, size, seed):
        mesh = trimesh.creation.box(extents=size)
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
        # Link 11 already sits at the fingertips, so no flange standoff is needed.
        calibration = tcp_to_ee(np.eye(4)[None], standoff=0)[0]
        return PlanningGrasps.from_grasps(
            grasps, world, frame_id="world", tool_link="panda_grasptarget",
            tcp_to_tool=calibration, approach_distance=.14, lift_offset=(0, 0, .22),
        )

    def run(self, count):
        results = []
        objects = self.spawn(count)
        for i, (body, size, goal) in enumerate(objects):
            targets = self.targets(body, size, i)
            if not len(targets):
                raise RuntimeError(f"No sampled top-down antipodal grasps for object {i}")
            target = targets.select([0])
            pre, grasp, lift = [to_pybullet(target, stage=s)[0] for s in ("pregrasp", "grasp", "lift")]
            print(f"Object {i + 1}/{count}: {len(targets)} sampled candidates; "
                  f"selected width {target.widths[0]:.3f} m", flush=True)
            self.grip(.08)
            transit = np.array(pre[0]); transit[2] = .36
            self.move(transit, pre[1], 1.5)
            self.move(*pre, seconds=.8)
            self.move(*grasp, seconds=1.)
            self.grip(0., settle=180)
            contacts = [bool(p.getContactPoints(self.robot, body, linkIndexA=j)) for j in self.fingers]
            self.move(*lift, seconds=1.2)
            lifted_height = p.getBasePositionAndOrientation(body)[0][2]
            grasped = all(contacts) and lifted_height > size[2] / 2 + .10
            # Preserve the object-to-tool offset while translating to its destination.
            initial = np.array(p.getBasePositionAndOrientation(body)[0])
            destination = np.array(lift[0]) + goal - initial
            destination[2] += .012  # release just above the floor
            overhead = destination.copy(); overhead[2] = .36
            self.move(overhead, grasp[1], 1.5)
            self.move(destination, grasp[1], 1.2)
            self.grip(.08, settle=180)
            self.move(overhead, grasp[1], .9)
            self.step(120)
            final = np.array(p.getBasePositionAndOrientation(body)[0])
            error = float(np.linalg.norm(final[:2] - goal[:2]))
            placed = bool(grasped and error < .04 and final[2] < size[2] + .02)
            record = {"object": i, "sampled_candidates": len(targets), "candidate_id": int(target.candidate_ids[0]),
                      "jaw_width": float(target.widths[0]), "bilateral_contact": all(contacts),
                      "lifted_height": lifted_height, "goal": goal.tolist(), "final_position": final.tolist(),
                      "placement_error_xy": error, "success": placed}
            results.append(record)
            print(f"  bilateral contact={all(contacts)}, lift={lifted_height:.3f} m, "
                  f"placement error={error:.3f} m, success={placed}", flush=True)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true")
    mode.add_argument("--headless", action="store_true")
    parser.add_argument("--objects", type=int, default=5, choices=range(1, 7))
    parser.add_argument("--hold", action="store_true", help="keep GUI open after finishing")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--realtime", action="store_true", help="pace headless simulation too")
    args = parser.parse_args()
    demo = Demo(gui=args.gui, realtime=args.gui or args.realtime)
    try:
        results = demo.run(args.objects)
        report = {"objects": results, "successes": sum(r["success"] for r in results),
                  "total": len(results), "attachment_constraints": False}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Placed {report['successes']}/{report['total']} objects", flush=True)
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
