"""Sample a mesh and export frame-labelled grasp/pregrasp/lift targets as JSON.

PYTHONPATH=src python examples/export_planning.py assets/meshes/pocky_box_metric_fp.glb \
    --object-position .5 0 .2 --frame-id panda_link0 --tool-link panda_link8 \
    --panda-flange --output /tmp/grasps.json

Position/quaternion describe the original file frame in world coordinates.
Use --world-to-frame for a non-world planning frame and --tcp-to-tool for a
custom calibrated tool transform. Both matrix files must be JSON 4x4 arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grasp_sampler import GraspSampler, PlanningGrasps, mesh_world_pose, tcp_to_ee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-position", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--object-quaternion", nargs=4, type=float, default=[0, 0, 0, 1],
                        help="source/file frame quaternion, xyzw")
    parser.add_argument("--frame-id", required=True, help="label of output frame; provide its transform below")
    parser.add_argument("--tool-link", required=True)
    parser.add_argument("--world-to-frame", type=Path, help="JSON T_frame_world, defaults to identity")
    tool = parser.add_mutually_exclusive_group()
    tool.add_argument("--tcp-to-tool", type=Path, help="JSON T_tcp_tool, defaults to identity (sampler TCP)")
    tool.add_argument("--panda-flange", action="store_true", help="use the existing 0.105 m Panda flange mapping")
    parser.add_argument("--approach-distance", type=float, default=.1)
    parser.add_argument("--lift-offset", nargs=3, type=float, default=[0, 0, .1], help="world XYZ vector (m)")
    parser.add_argument("--max-grasps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_grasps < 1:
        parser.error("--max-grasps must be positive")
    source_world = np.eye(4)
    source_world[:3, 3] = args.object_position
    source_world[:3, :3] = Rotation.from_quat(args.object_quaternion).as_matrix()
    sampler = GraspSampler()
    obj = sampler.load(args.mesh)
    grasps = sampler.generate(obj, seed=args.seed)[:args.max_grasps]
    calibration = (tcp_to_ee(np.eye(4)[None])[0] if args.panda_flange else
                   json.loads(args.tcp_to_tool.read_text()) if args.tcp_to_tool else None)
    targets = PlanningGrasps.from_grasps(
        grasps, mesh_world_pose(obj, source_world), frame_id=args.frame_id, tool_link=args.tool_link,
        tcp_to_tool=calibration,
        world_to_frame=json.loads(args.world_to_frame.read_text()) if args.world_to_frame else None,
        approach_distance=args.approach_distance, lift_offset=args.lift_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(targets.to_dict(), indent=2) + "\n")
    print(f"Exported {len(targets)} grasp/pregrasp/lift targets to {args.output}")


if __name__ == "__main__":
    main()
