"""Generate grasps for a mesh and convert them to robot poses.

    python examples/basic.py path/to/object.glb
"""
import sys

import numpy as np

from grasp_sampler import GraspSampler, stack_poses, tcp_to_ee, to_world


def main(mesh_path: str) -> None:
    sampler = GraspSampler()
    grasps = sampler.sample(mesh_path)
    print(f"generated {len(grasps)} grasps")

    by_kind: dict[str, int] = {}
    for g in grasps:
        by_kind[g.kind] = by_kind.get(g.kind, 0) + 1
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:10s} {count}")

    # Place the object at the world origin (replace with its real 4x4 pose), then
    # express grasps in the robot's flange frame for IK / motion planning.
    object_pose = np.eye(4)
    tcp_world = to_world(stack_poses(grasps), object_pose)
    ee_world = tcp_to_ee(tcp_world)
    print(f"flange poses ready: {ee_world.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/basic.py <mesh-file>")
    main(sys.argv[1])
