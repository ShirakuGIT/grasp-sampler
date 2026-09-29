# grasp-sampler

A simple, dependency-light **parallel-jaw grasp sampler** for mesh objects.
Point it at a mesh, get back a set of grasp poses ready for IK / motion planning.

It combines three classic generators:

| Generator    | What it does |
|--------------|--------------|
| `primitives` | Top-down and side grasps on the object's bounding box |
| `obb_face`   | Grasps centered on the oriented bounding box faces |
| `antipodal`  | Dex-Net-style force-closure contacts found by ray casting |

No simulator, no robot model, no GPU. Just NumPy, trimesh, SciPy, and Open3D.

## Install

```bash
pip install -e .                 # core
pip install -e ".[viz]"          # + interactive viewer
```

Or just the dependencies: `pip install -r requirements.txt`.

## Usage

```python
from grasp_sampler import GraspSampler, stack_poses, to_world, tcp_to_ee
import numpy as np

sampler = GraspSampler()
grasps = sampler.sample("object.glb")          # list[Grasp], object-local TCP poses

# Place the object in the world and convert to Franka flange poses for IK:
object_pose = np.eye(4)                         # the object's real 4x4 world pose
tcp_world = to_world(stack_poses(grasps), object_pose)
flange_poses = tcp_to_ee(tcp_world)             # (N, 4, 4)
```

Each `Grasp` carries:

- `pose` — `(4, 4)` TCP pose in the object's local frame
- `kind` — `"top" | "side" | "obb_face" | "antipodal"`
- `width` — required finger opening (m)
- `meta` — generator-specific details

### Grasp frame convention

Grasps use the standard tool-center-point (TCP) convention:

```
+Z  approach axis   (the hand travels along +Z onto the object)
+X  closing axis    (the line between the two fingertips)
+Y  right-handed completion
```

`tcp_to_ee` converts to the Franka Panda flange frame (closing on +Y, flange
`0.105 m` behind the TCP). Pass a different `remap` / `standoff` for other grippers.

## Robust evaluation under perceived object uncertainty

See [the uncertainty guide](docs/uncertainty.md) and run:

```bash
python examples/robust_grasp_demo.py assets/meshes/pocky_box_metric_fp.glb
```

The robot command stays fixed; only hypothetical object geometry and pose vary.
`generate_robust_grasp_set` returns grasps whose success rate over sampled
perturbations meets a threshold, validated on an independent bank:

```bash
python examples/visualize_robust_set.py assets/meshes/ycb_cracker_box_metric_fp.glb
```

The existing generators and their default behavior remain available.

## Visualize grasps

```bash
python examples/visualize.py object.glb
python examples/visualize.py object.glb --methods antipodal --max 40
```

Opens the mesh with one gripper wireframe per grasp, colored by generator
(blue = top, green = side, orange = obb, red = antipodal).

## Collision-check in PyBullet (optional)

```bash
pip install -e ".[sim]"
python examples/pybullet_demo.py assets/meshes/pocky_box_metric_fp.glb
```

Spawns the mesh and a parallel-jaw gripper proxy (no robot URDF needed) to drop
grasps whose gripper body would hit the object:

```python
from grasp_sampler.pybullet_sim import collision_filter
mask = collision_filter("object.glb", grasps, clearance=0.0)   # bool per grasp
kept = [g for g, m in zip(grasps, mask) if m]
```

## Sample meshes

`assets/meshes/` holds 31 ready-to-use objects (YCB + household items),
decimated to ~2k faces and stripped of textures (~36 KB each). Re-decimate your
own library with:

```bash
python tools/decimate_meshes.py <src_dir> assets/meshes --faces 2000
```

## Configure

Every threshold lives in `GraspConfig`:

```python
from grasp_sampler import GraspSampler, GraspConfig

cfg = GraspConfig(
    gripper_max_width=0.08,        # max finger opening (m)
    antipodal_samples=1500,        # surface points sampled
    directed=True,                 # keep only grasps from `approach_facets`
    approach_facets=((0, -1, 0),), # e.g. only approach from the front
    cone_deg=50.0,                 # half-angle of each approach cone
)
sampler = GraspSampler(cfg)
```

Set `directed=False` for a full ring of approaches around each contact pair.

## Layout

```
src/grasp_sampler/
  types.py        Grasp, ObjMesh, GraspConfig
  mesh.py         mesh loading + geometry classification
  primitives.py   bounding-box top/side grasps      (ss-pybullet, MIT)
  obb.py          oriented-bounding-box face grasps
  antipodal.py    force-closure antipodal sampler    (Dex-Net, BSD-2)
  transforms.py   object-local -> world -> flange poses
  sampler.py      GraspSampler facade
  viz.py          trimesh gripper-wireframe viewer
  uncertainty.py  pose/geometry hypothesis sampling
  robust.py       fixed-command evaluation + validity backends
  robust_set.py   robust grasp-set generation
  robust_viz.py   matplotlib uncertainty viewer
examples/
  basic.py        generate + convert to robot poses
  visualize.py    generate + view
  robust_grasp_demo.py, visualize_robust.py, visualize_robust_set.py
```

## License

MIT. Includes algorithms adapted from ss-pybullet (MIT) and Dex-Net (BSD-2);
see [LICENSE](LICENSE) for attribution.
