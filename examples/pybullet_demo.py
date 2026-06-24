"""Sample grasps, filter them by collision in PyBullet, and view the result.

    python examples/pybullet_demo.py assets/meshes/pocky_box_metric_fp.glb
    python examples/pybullet_demo.py object.glb --no-gui   # filter only

Requires pybullet: pip install pybullet
"""
import argparse

from grasp_sampler import GraspSampler
from grasp_sampler.pybullet_sim import collision_filter, show_grasps


def main() -> None:
    ap = argparse.ArgumentParser(description="PyBullet collision filter + viewer")
    ap.add_argument("mesh")
    ap.add_argument("--clearance", type=float, default=0.0,
                    help="reject grasps whose body comes within this of the object (m)")
    ap.add_argument("--max", type=int, default=30, help="cap grasps shown in the GUI")
    ap.add_argument("--no-gui", action="store_true", help="filter only, no window")
    args = ap.parse_args()

    sampler = GraspSampler()
    grasps = sampler.sample(args.mesh)
    mask = collision_filter(args.mesh, grasps, clearance=args.clearance)
    kept = [g for g, m in zip(grasps, mask) if m]
    print(f"{len(kept)}/{len(grasps)} grasps pass the body-collision filter")

    if not args.no_gui:
        show_grasps(args.mesh, kept, max_grasps=args.max)


if __name__ == "__main__":
    main()
