"""Generate a robust grasp set for one mesh and inspect it in the uncertainty viewer.

Every grasp shown passed the R threshold on an independent validation bank of
perturbed object poses/sizes; the gripper command stays fixed while the object moves.
"""
import argparse
import time
import numpy as np

from grasp_sampler import (GraspConfig, GeometryUncertainty, PoseUncertainty,
    FinitePadFixedObjectBackend, FinitePadQuasistaticBackend, RobustGraspEvaluator, RobustSetConfig, generate_robust_grasp_set, load_mesh,
    sample_hypotheses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--threshold", type=float, default=.95)
    p.add_argument("--hypotheses", type=int, default=50)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--backend", choices=["point", "pad", "quasistatic"], default="quasistatic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--translation-mm", type=float, nargs=3, default=[5, 5, 5])
    p.add_argument("--rotation-deg", type=float, nargs=3, default=[3, 3, 3])
    p.add_argument("--scale-percent", type=float, nargs=3, default=[1, 1, 1])
    p.add_argument("--world-xyz", type=float, nargs=3, default=[0.4, 0, 0.2])
    p.add_argument("--shown", type=int, default=100, help="max grasps loaded in the viewer")
    p.add_argument("--snapshot")
    p.add_argument("--sample", type=int, default=1)
    a = p.parse_args()
    if a.snapshot:
        import matplotlib
        matplotlib.use("Agg")
    from grasp_sampler.robust_viz import RobustGraspViewer

    mesh = load_mesh(a.mesh, center=False, upright_transform=None).mesh
    pose = np.eye(4); pose[:3, 3] = a.world_xyz
    t = np.asarray(a.translation_mm) / 1000
    r = np.deg2rad(a.rotation_deg)
    s = np.asarray(a.scale_percent) / 100
    pu = PoseUncertainty(translation_bounds_xyz=np.column_stack([-t, t]),
                         rotation_bounds_rpy=np.column_stack([-r, r]))
    gu = GeometryUncertainty(scale_bounds_xyz=np.column_stack([1 - s, 1 + s]))
    cfg = GraspConfig(contact_geometry_policy="mesh")
    backend = {"point": None, "pad": FinitePadFixedObjectBackend,
               "quasistatic": FinitePadQuasistaticBackend}[a.backend]
    backend = backend(cfg) if backend else None
    t0 = time.time()
    res = generate_robust_grasp_set(mesh, pose, requested_count=a.count,
        robustness_threshold=a.threshold, hypothesis_count=a.hypotheses,
        pose_uncertainty=pu, geometry_uncertainty=gu, grasp_config=cfg, backend=backend,
        config=RobustSetConfig(candidate_budget=a.budget), seed=a.seed)
    d = res.diagnostics
    print(f"{d['returned_count']}/{a.count} grasps with validation R >= {a.threshold} "
          f"({d['raw_generated']} raw, {d['unique_candidates']} unique, "
          f"{d['robust_candidates']} robust) in {time.time()-t0:.1f}s")
    if not res.selected:
        raise SystemExit("no robust grasps found")
    Rs = [x.validation_R for x in res.selected]
    print(f"validation R of returned grasps: min {min(Rs):.2f} mean {np.mean(Rs):.2f}")
    grasps = res.grasps[:a.shown]
    hyps = sample_hypotheses(mesh, pose, a.hypotheses, seed=a.seed + 7,
        pose_uncertainty=pu, geometry_uncertainty=gu)
    viewer = RobustGraspViewer(mesh, grasps, pose, hyps, RobustGraspEvaluator(mesh, cfg, backend=backend))
    viewer.sample_slider.set_val(min(a.sample, a.hypotheses))
    if a.snapshot:
        viewer.fig.savefig(a.snapshot, dpi=150)
        print("saved", a.snapshot)
    else:
        viewer.show()


if __name__ == "__main__":
    main()
