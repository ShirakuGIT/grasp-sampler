# Robust grasp evaluation

This layer models uncertainty in the perceived **object**, never robot joints,
FK, execution, or controller noise. Coordinates use metres and radians.

## Mesh frames

`load_mesh(path)` retains the default Y-up → Z-up rotation and AABB centering
(the old docstring incorrectly called this OBB centering). To keep the original
reconstruction frame:

```python
obj = sampler.load(path, center=False, upright_transform=None)
```

Scene-node transforms are now respected, including multiple instances. The
preserved frame is the file's scene frame, not each node's untransformed vertex
frame. This corrects old loading of transformed scenes. No metric unit conversion
is inferred; provide a mesh in metres. Optional `process=False` also disables
trimesh's usual vertex merging/cleanup when exact vertex topology is needed.
`obj.source_to_mesh` records the applied
normalization. If normalization is enabled, a FoundationPose estimate for the
original file must be composed as:

```python
T_W_mesh = T_W_original @ np.linalg.inv(obj.source_to_mesh)
```

Primitive and fallback OBB grasp placement respects nonzero mesh centers.

## Sampling and fixed commands

```python
import numpy as np
from grasp_sampler import (GraspSampler, GraspConfig, PoseUncertainty,
    GeometryUncertainty, sample_hypotheses, RobustGraspEvaluator, rank_grasps)

sampler = GraspSampler(GraspConfig(contact_geometry_policy="mesh"))
obj = sampler.load(path, center=False, upright_transform=None)
grasps = sampler.generate(obj, methods=("antipodal",))
T_W_O_est = np.eye(4)  # replace with perception estimate in this mesh frame
hypotheses = sample_hypotheses(obj.mesh, T_W_O_est, 100, seed=42,
    pose_uncertainty=PoseUncertainty(
        translation_bounds_xyz=[[-.010, .010], [-.007, .007], [-.012, .012]],
        rotation_bounds_rpy=np.deg2rad([[-3, 3], [-3, 3], [-5, 5]])),
    geometry_uncertainty=GeometryUncertainty(scale_bounds_xyz=[[.97, 1.03]]*3))
evaluator = RobustGraspEvaluator(obj.mesh)
rows = rank_grasps(grasps, evaluator,
                  nominal_object_pose=T_W_O_est, hypotheses=hypotheses)

# Nominal-first, failure-guided set generation. Validation is independent and
# is the only bank used for the R >= threshold acceptance decision.
from grasp_sampler import RobustSetConfig, generate_robust_grasp_set
validation = sample_hypotheses(obj.mesh, T_W_O_est, 500, seed=43,
    pose_uncertainty=PoseUncertainty(
        translation_std_xyz=[.003, .003, .003],
        rotation_std_rpy=np.deg2rad([3, 3, 3])),
    geometry_uncertainty=GeometryUncertainty(scale_std_xyz=[.03, .03, .03]))
result = generate_robust_grasp_set(obj.mesh, T_W_O_est, requested_count=20,
    robustness_threshold=.95, search_hypotheses=hypotheses,
    validation_hypotheses=validation,
    config=RobustSetConfig(strategy="failure_guided", candidate_budget=200), seed=7)
commands = result.world_commands  # immutable world-frame robot commands
```

For every candidate, the evaluator computes `T_W_G_cmd = T_W_O_est @ grasp.pose`
once. For hypothesis `i`, it checks `inv(T_W_O_i) @ T_W_G_cmd`. It never computes
`T_W_O_i @ grasp.pose`: that would let the robot follow the unknown true object.
TCP +X is closing, +Z is approach, and +Y completes the right-handed frame.
Pregrasp is a fixed translation backwards along commanded TCP Z.

`PoseUncertainty` supports per-axis Gaussian `translation_std_xyz` and
`rotation_std_rpy`, or `(3, 2)` bounded intervals, independently for translation
and rotation. Do not specify both std and bounds for the same component.
RPY deltas explicitly mean `D = Rz(yaw) @ Ry(pitch) @ Rx(roll)`:

| frame | rotation | translation |
|---|---|---|
| `world` (default) | `D @ R_est` | `t_est + dt` |
| `object` | `R_est @ D` | `t_est + R_est @ dt` |

World rotations are about the object origin, not the world origin.
Geometry scales act on a copy as `pivot + scale_xyz * (vertices - pivot)`;
the default pivot is the original mesh origin, with no recentering. Gaussian
`scale_std_xyz` is also available; nonpositive sampled scales raise an error.
Pass either `seed` or a NumPy `rng` for reproducibility.

Empirical/correlated perception runs can directly supply
`ObjectHypothesis(mesh, world_pose, weight=1.0, metadata={...})`. Each pose must
refer to its paired mesh's local frame. No distribution is required, and meshes
are not mutated by evaluation. Future reconstruction ensembles/deformations
require no change to the evaluator.

## Validity, diagnostics, and limits

`RobustGraspEvaluator` accepts a nominal mesh, optional `GraspConfig`, and
`opening_width` (default configured maximum). This is the same deterministic
pre-opening for every hypothesis. `Grasp.width` remains the nominal required
contact span, not a closure command. The model assumes symmetric closure to
contact without moving the object. First inward ray hits from the two open
fingertip positions define candidate contacts; no replacement grasp is sampled.
Widths must fit, contacts must straddle TCP, and both outward normals must pass
the same Coulomb friction-cone test used by the nominal sampler.

`contact_tolerance` (default 2 mm) limits the difference in the two fingers'
required closing travel. It is an approximation to pad compliance, not robot
pose uncertainty. Set zero for ideal rigid simultaneous symmetric contacts.
Off-center objects that would require pushing/recentering fail this check.
Point-contact force closure here refers to the existing antipodal criterion,
not a general 6D wrench-space proof or evidence of a successful lift.

The default `GeometricGraspBackend` checks sampled palm/stem clearance against
the actual mesh at the grasp and along the approach (`approach_distance=0.05`,
`approach_steps=6`). This inherited approximation does not check finite finger
shafts, the complete robot, external obstacles, or continuous swept volumes.
It can miss collisions between samples. `check_body=False` deliberately disables
this part and is recorded in metadata. Non-watertight meshes and inconsistent
normals can make ray hits/containment unreliable; watertightness is recorded and
no automatic mesh repair is performed. Ray/proximity errors propagate rather
than being mislabeled as ordinary grasp failures.

Contact geometry policies are `mesh`, `convex_hull`, and `existing_auto_policy`.
The nominal sampler retains its automatic volume-ratio hull fallback by default.
Robust evaluation defaults to `mesh` because hulls can invent contact surfaces.
Both policy and actual selected geometry are recorded; collision geometry is
always the hypothesis mesh. Sampler metadata also records its contact geometry.

Results include unweighted and normalized weighted success fractions, all-sample
worst-case validity (including zero-weight samples), counts, nominal validity,
contact widths/points, failure reasons, hypothesis metadata, and world/local
command matrices. Empty sets, all-zero weights, and invalid transforms raise
errors. These fractions describe the supplied hypotheses, not a calibrated
real-world probability guarantee. `rank_grasps` supports `expected_success` and
`worst_case_valid`, preserving caller-provided `nominal_scores`. Existing
generators have no score, so the default nominal score is `None`, shown as N/A
in the demo. Pose/geometry failure attribution is deferred.

## Optional PyBullet and the next backend

```python
from grasp_sampler.pybullet_sim import fixed_pose_collision_check
evaluator = RobustGraspEvaluator(obj.mesh,
    collision_checker=fixed_pose_collision_check)
```

This adds checks with the existing palm/stem box proxy in an isolated DIRECT
client, using hypothesis vertices without reloading/normalizing. It corrects
proxy component offsets for this new path; the legacy `collision_filter` is
unchanged. It remains a static, discretized body check, with no finger closure
or lift. It creates a client for each check, so is intended for small batches.

A custom `backend` implements `ValidityBackend.evaluate(mesh,
object_world_pose=..., world_gripper_command=..., opening_width=...)`, returning
`FixedGraspResult`. This supplies the fixed world command explicitly for a future
Franka execution backend: deterministic pregrasp/approach, close, lift, and test
capture while changing only the hypothetical object. Dynamics are not mandatory
and are not implemented by this patch.
