"""Generate and visualize grasps for a mesh.

    python examples/visualize.py path/to/object.glb
    python examples/visualize.py object.glb --methods antipodal --max 40

Opens an interactive window: the object mesh with one gripper wireframe per
grasp, colored by generator (blue=top, green=side, orange=obb, red=antipodal).
"""
import argparse

from grasp_sampler import GraspSampler
from grasp_sampler.viz import show_grasps


def main() -> None:
    ap = argparse.ArgumentParser(description="visualize sampled grasps")
    ap.add_argument("mesh", help="mesh file (GLB, OBJ, STL, ...)")
    ap.add_argument("--methods", nargs="+",
                    default=["primitives", "obb_face", "antipodal"],
                    help="which generators to run")
    ap.add_argument("--max", type=int, default=None,
                    help="subsample to at most this many grasps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sampler = GraspSampler()
    obj = sampler.load(args.mesh)
    grasps = sampler.generate(obj, methods=tuple(args.methods), seed=args.seed)
    print(f"{len(grasps)} grasps  (geom class: {obj.geom_class})")

    show_grasps(obj, grasps, max_grasps=args.max)


if __name__ == "__main__":
    main()
