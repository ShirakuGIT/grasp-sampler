"""Plan to an exported target set in an already configured cuRobo environment.

Current: python examples/plan_curobo.py /tmp/grasps.json --robot franka.yml
Legacy:  python examples/plan_curobo.py /tmp/grasps.json --api legacy --robot franka.yml

Uses the robot's retract/default configuration as start. Targets must already
be in the robot base frame for exactly the configured tool link. This minimal
example has no scene obstacles; supply your application's collision scene
before planning real tasks. Prints planning success, never executes a robot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grasp_sampler import PlanningGrasps
from grasp_sampler.integrations.curobo import to_curobo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Path)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--api", choices=["current", "legacy"], default="current")
    parser.add_argument("--stage", choices=["grasp", "pregrasp", "lift"], default="pregrasp")
    args = parser.parse_args()
    targets = PlanningGrasps.from_dict(json.loads(args.targets.read_text()))
    goals = to_curobo(targets, api=args.api, stage=args.stage)
    if args.api == "current":
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import JointState

        planner = MotionPlanner(MotionPlannerCfg.create(robot=args.robot, max_goalset=len(targets)))
        if list(planner.tool_frames) != [targets.tool_link]:
            raise ValueError(f"Configured tool frames {planner.tool_frames} do not match {targets.tool_link}")
        planner.warmup()
        start = JointState.from_position(planner.default_joint_state.position.unsqueeze(0),
                                         joint_names=planner.joint_names)
        result = planner.plan_pose(goals, start)
    else:
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

        planner = MotionGen(MotionGenConfig.load_from_robot_config(args.robot))
        if planner.kinematics.ee_link != targets.tool_link:
            raise ValueError("Configured cuRobo ee_link does not match exported tool_link")
        planner.warmup(n_goalset=len(targets))
        start = JointState.from_position(planner.get_retract_config().view(1, -1),
                                         joint_names=planner.kinematics.joint_names)
        result = planner.plan_goalset(start, goals, MotionGenPlanConfig())
    if result is None or not bool(result.success.any()):
        raise SystemExit("No feasible plan found")
    print("Planning succeeded. Inspect result.get_interpolated_plan() before execution.")


if __name__ == "__main__":
    main()
