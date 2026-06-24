# Configuration

Every tunable lives in {class}`~grasp_sampler.types.GraspConfig`. Defaults target
a Franka Panda gripper.

```python
from grasp_sampler import GraspSampler, GraspConfig

cfg = GraspConfig(
    gripper_max_width=0.08,          # max finger opening (m)
    antipodal_samples=1500,          # surface points ray-cast per object
    directed=True,                   # keep only grasps from `approach_facets`
    approach_facets=((0, -1, 0),),   # e.g. approach only from the front
    cone_deg=50.0,                   # half-angle of each approach cone
)
sampler = GraspSampler(cfg)
```

## Directed vs undirected antipodal

In **directed** mode (the default) the antipodal sampler keeps only grasps whose
approach falls within `cone_deg` of one of the `approach_facets`. This is useful
when the robot can only reach the object from certain sides (e.g. into a shelf).

Set `directed=False` for the full ring of approaches around each contact pair —
denser, but many will be unreachable in a constrained workspace.

## Key parameters

| Parameter             | Meaning                                            |
|-----------------------|----------------------------------------------------|
| `gripper_max_width`   | Max finger opening; wider contacts are rejected    |
| `gripper_min_width`   | Fingers effectively touch below this               |
| `antipodal_samples`   | Surface points sampled (more = denser, slower)     |
| `antipodal_max_grasps`| Cap on antipodal grasps kept                       |
| `friction_coef`       | Coulomb μ; friction-cone half-angle = `atan(μ)`    |
| `spine_clearance`     | Min gap required between gripper body and mesh (m)  |
| `approach_facets`     | Approach directions kept in directed mode          |
| `cone_deg`            | Half-angle of each approach cone                   |

See the full list in the {doc}`api`.
