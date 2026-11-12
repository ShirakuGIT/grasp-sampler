# Developing grasp-sampler

## Set up and verify changes

From the repository root:

```bash
pip install -e ".[dev,docs]"
pytest
ruff check src tests examples tools
ruff format --check src
vulture
PYTHONPATH=src python tools/publish_docs.py
```

Install simulator extras when changing adapters or collision checks:

```bash
pip install -e ".[sim,mujoco,fr3]" "mujoco==3.17.0"
```

Optional integration tests skip when dependencies are unavailable. A green
core-only run therefore does not validate simulator paths. The seeded robot
demo is checked with MuJoCo 3.17.0; upgrades need a new physical run.

Library code follows Ruff's formatter. Use `vulture` with no path arguments to
include the checked-in configuration and reviewed whitelist.

## Where a change belongs

| Responsibility | Module | Boundary |
| --- | --- | --- |
| Data and sampler parameters | `types.py` | Metres; mesh-local TCP grasps |
| Loading and appearance | `mesh.py` | Record source-to-mesh transforms; retain textures |
| Rigid transforms | `transforms.py` | Validate SE(3); return independent arrays |
| Candidate generation | `sampler.py`, `primitives.py`, `obb.py`, `antipodal.py` | Generate proposals without simulator imports |
| Geometric clearance | `collision.py` | Share contact/body definitions across generation and evaluation |
| Object distributions | `uncertainty.py` | Perturb objects, never the commanded robot pose |
| One-command evaluation | `robust.py` | Replace validity models through `ValidityBackend` |
| Proposal and selection | `robust_set.py` | Search first; accept on an independent validation bank |
| Planner target data | `planning.py` | Keep poses, IDs, widths and scores aligned |
| Framework conversion | `integrations/` | Import optional frameworks inside entry points |
| Rendering and simulation | `viz.py`, `robust_viz.py`, `pybullet_sim.py`, `examples/` | Own and release clients, files and rendering contexts |

The sampler is a facade over named generation strategies. Validity backends
provide a separate strategy boundary for fixed commands. Planner adapters
translate the resulting data into framework-specific types. Adding a generator
should not require a simulator; adding an adapter should not change generation.

## Add a sampling strategy

A `GraspGenerator` is any callable accepting `(obj, config, rng)` and returning
`list[Grasp]`. Operate in the processed mesh's local frame and leave the mesh
and configuration unchanged. Use the supplied RNG rather than global state.

```python
from grasp_sampler import GraspSampler, antipodal_grasps

def centered_antipodal(obj, config, rng):
    candidates = antipodal_grasps(obj, config, rng)
    return sorted(candidates, key=lambda g: float(g.pose[:3, 3] @ g.pose[:3, 3]))

sampler = GraspSampler(generators={"centered_antipodal": centered_antipodal})
grasps = sampler.sample("object.glb", methods="centered_antipodal", seed=7)
```

Registration is per instance. All registered methods run by default; pass
`methods` to select a subset. Execution follows registration order, preserving
the built-in order (`primitives`, `obb_face`, `antipodal`). Unknown names raise
`ValueError`; an empty method list returns no candidates. Existing generator
functions remain directly callable.

## Add a validity backend

Implement the `ValidityBackend` protocol exported by `grasp_sampler`:

```python
from grasp_sampler import FixedGraspResult

class ApplicationBackend:
    def evaluate(self, mesh, *, object_world_pose, world_gripper_command,
                 opening_width, nominal_object_pose=None) -> FixedGraspResult:
        # Replace this explicit failure with your application's checks.
        return FixedGraspResult(False, "application_check_not_implemented")
```

Pass an instance as `backend=` to `RobustGraspEvaluator` or
`generate_robust_grasp_set`. Configure custom backends directly; evaluator
`backend_options` apply only to the default geometric implementation.

Compute the command once:

```text
T_world_gripper = T_world_object_estimate @ T_object_gripper
T_object_i_gripper = inverse(T_world_object_i) @ T_world_gripper
```

Each hypothesis changes the object. Never recompute the world command from the
hypothetical pose. Evaluators pass independent pose/command arrays to backends
so a backend cannot corrupt stored commands. Treat the mesh as read-only.
Return failure reasons and mesh-local contacts instead of moving the TCP to
make a candidate succeed.

Search-bank failures guide proposals. Each round contributes its failures
before the next round chooses sources. Final acceptance uses validation-bank
weighted success. Explicit validation banks are preserved when search banks
are sampled. `jaw_symmetry=False` retains distinct commands for asymmetric jaws.

## Compatibility and dead code

Public exports live in `grasp_sampler.__all__`; result fields are also API.
A field need not be read internally to be useful externally. `validate_pose`
now lives in `transforms.py` and is exported at package level; the previous
`uncertainty.validate_pose` import remains supported.

Regressions should exercise caller contracts: frame alignment, deterministic
samples, stable candidate IDs, fixed commands, independent validation or client
ownership. Preserve upstream algorithm attribution in source and `LICENSE`.

Vulture runs at 60% confidence across the library, examples, tools and tests.
`vulture_whitelist.py` documents native MuJoCo properties, public result fields
and defensive test-double methods. Review findings before adding exceptions;
do not delete public API merely to silence static analysis.

## Resources and publishing

PyBullet collision helpers own isolated clients and remove temporary mesh
conversions after loading. All accepted file formats use the sampler's
`load_mesh` frame. A nonblocking `show_grasps` call returns its live client,
which the caller must disconnect. Mesh collision remains a static surface
check, without solid-containment or continuous-motion guarantees.

Commit source changes before `python tools/publish_docs.py --publish`. It
creates a clean local Sphinx build and pushes generated HTML to `gh-pages`.
Keep generated reports in temporary or user-selected output directories;
website demo assets live under `docs/_static/videos/`.
