"""Inspect fixed grasp commands against object perturbations in interactive 3D."""
import argparse

import numpy as np

from grasp_sampler import (
    GeometryUncertainty,
    GraspConfig,
    GraspSampler,
    PoseUncertainty,
    RobustGraspEvaluator,
    sample_hypotheses,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", help="Mesh in metres; original frame is preserved")
    parser.add_argument("--hypotheses", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--translation-mm", type=float, nargs=3, default=[10, 10, 10], metavar=("X", "Y", "Z"))
    parser.add_argument("--rotation-deg", type=float, nargs=3, default=[5, 5, 5], metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--scale-percent", type=float, nargs=3, default=[3, 3, 3], metavar=("X", "Y", "Z"))
    parser.add_argument("--world-xyz", type=float, nargs=3, default=[0.4, 0, 0.2])
    parser.add_argument("--snapshot", help="Save a PNG instead of opening a GUI")
    parser.add_argument("--sample", type=int, default=0, help="Initial hypothesis index; 0 is nominal")
    args = parser.parse_args()
    if args.hypotheses < 1 or args.candidates < 1 or not 0 <= args.sample <= args.hypotheses:
        parser.error("positive hypothesis/candidate counts and an in-range sample are required")
    values = np.array([args.translation_mm, args.rotation_deg, args.scale_percent])
    if not np.isfinite(values).all() or (values < 0).any() or max(args.scale_percent) >= 100:
        parser.error("uncertainty magnitudes must be finite and nonnegative; scale must be below 100%")
    if args.snapshot:
        import matplotlib
        matplotlib.use("Agg")
    from grasp_sampler.robust_viz import RobustGraspViewer
    cfg = GraspConfig(contact_geometry_policy="mesh", antipodal_samples=500)
    sampler = GraspSampler(cfg)
    obj = sampler.load(args.mesh, center=False, upright_transform=None)
    print("Sampling nominal antipodal grasps...", flush=True)
    candidates = sampler.generate(obj, methods=("antipodal",), seed=args.seed)
    if not candidates:
        parser.error("no nominal antipodal candidates found on this mesh")
    indices = np.linspace(0, len(candidates)-1, min(args.candidates, len(candidates)), dtype=int)
    grasps = [candidates[i] for i in indices]
    print(f"Showing {len(grasps)} of {len(candidates)} candidates (evenly subsampled).", flush=True)
    nominal_pose = np.eye(4)
    nominal_pose[:3, 3] = args.world_xyz
    translation = np.asarray(args.translation_mm) / 1000
    rotation = np.deg2rad(args.rotation_deg)
    scale = np.asarray(args.scale_percent) / 100
    hypotheses = sample_hypotheses(obj.mesh, nominal_pose, args.hypotheses, seed=args.seed,
        pose_uncertainty=PoseUncertainty(
            translation_bounds_xyz=np.column_stack([-translation, translation]),
            rotation_bounds_rpy=np.column_stack([-rotation, rotation])),
        geometry_uncertainty=GeometryUncertainty(scale_bounds_xyz=np.column_stack([1-scale, 1+scale])))
    viewer = RobustGraspViewer(obj.mesh, grasps, nominal_pose, hypotheses, RobustGraspEvaluator(obj.mesh, cfg))
    viewer.sample_slider.set_val(args.sample)
    if args.snapshot:
        viewer.fig.savefig(args.snapshot, dpi=150)
        print(f"Saved {args.snapshot}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
