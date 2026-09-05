"""Fixed robot command, varying object: python examples/robust_grasp_demo.py mesh.glb."""
import argparse
import numpy as np
from grasp_sampler import (GraspSampler, GraspConfig, PoseUncertainty,
    GeometryUncertainty, sample_hypotheses, RobustGraspEvaluator, rank_grasps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh")
    parser.add_argument("--hypotheses", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=10)
    args = parser.parse_args()
    sampler = GraspSampler(GraspConfig(contact_geometry_policy="mesh", antipodal_samples=300))
    obj = sampler.load(args.mesh, center=False, upright_transform=None)
    grasps = sampler.generate(obj, methods=("antipodal",), seed=7)[:args.candidates]
    estimated_pose = np.eye(4)
    estimated_pose[:3, 3] = [0.4, 0, 0.2]
    hypotheses = sample_hypotheses(obj.mesh, estimated_pose, args.hypotheses, seed=42,
        pose_uncertainty=PoseUncertainty(
            translation_bounds_xyz=[[-0.01, 0.01]] * 3,
            rotation_bounds_rpy=[[-np.deg2rad(5), np.deg2rad(5)]] * 3),
        geometry_uncertainty=GeometryUncertainty(scale_bounds_xyz=[[0.97, 1.03]] * 3))
    evaluator = RobustGraspEvaluator(obj.mesh)
    rows = rank_grasps(grasps, evaluator, nominal_object_pose=estimated_pose, hypotheses=hypotheses)
    # Commands are computed from the ESTIMATE once, never from hypothetical poses.
    for row in rows:
        fixed_command = estimated_pose @ grasps[row.grasp_id].pose
        for hypothesis, outcome in zip(hypotheses, row.result.per_hypothesis_results):
            np.testing.assert_allclose(outcome.world_gripper_command, fixed_command)
            np.testing.assert_allclose(outcome.object_gripper_command,
                np.linalg.inv(hypothesis.world_pose) @ fixed_command)
    print("No nominal scores are emitted by this sampler; N/A preserves that fact.")
    print(" id  nominal_score  nominal_valid  robust_success  worst_case")
    for row in rows:
        r = row.result
        print(f"{row.grasp_id:3d}  {'N/A':>13s}  {str(r.nominal_valid):>13s}  "
              f"{r.weighted_success_probability:14.3f}  {r.worst_case_valid}")
    if not rows:
        print("No nominal antipodal grasps found on this mesh.")


if __name__ == "__main__":
    main()
