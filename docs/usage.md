# Usage

## Sampling grasps

```python
from grasp_sampler import GraspSampler

sampler = GraspSampler()
grasps = sampler.sample("object.glb")          # list[Grasp]
```

You can also load once and run specific generators:

```python
obj = sampler.load("object.glb")               # ObjMesh (upright, centered)
grasps = sampler.generate(obj, methods=("antipodal",), seed=0)
```

Each {class}`~grasp_sampler.types.Grasp` carries:

- `pose` — `(4, 4)` TCP pose in the object's local frame
- `kind` — `"top" | "side" | "obb_face" | "antipodal"`
- `width` — required finger opening (m)
- `meta` — generator-specific details

## Grasp frame convention

Grasps use the standard tool-center-point (TCP) convention:

```
+Z  approach axis   (the hand travels along +Z onto the object)
+X  closing axis    (the line between the two fingertips)
+Y  right-handed completion
```

The TCP sits at the midpoint between the two fingertips.

## Placing grasps in the world and on the robot

The sampler returns grasps in the object's local frame. To use them, place the
object in the world and convert to your gripper's frame:

```python
import numpy as np
from grasp_sampler import stack_poses, to_world, tcp_to_ee

object_pose = np.eye(4)                         # the object's real 4x4 world pose
tcp_world = to_world(stack_poses(grasps), object_pose)
flange_poses = tcp_to_ee(tcp_world)             # (N, 4, 4) Franka flange poses
```

{func}`~grasp_sampler.transforms.tcp_to_ee` defaults to the Franka Panda flange
(closing on +Y, flange `0.105 m` behind the TCP). Pass a different `remap` matrix
or `standoff` for another gripper.

## Collision checking with PyBullet

```python
from grasp_sampler.pybullet_sim import collision_filter

mask = collision_filter("object.glb", grasps, clearance=0.0)
kept = [g for g, m in zip(grasps, mask) if m]
```

This spawns the mesh and a gripper-body proxy (no robot URDF needed) and rejects
grasps whose palm/stem would penetrate the object. The fingers are excluded —
they are meant to straddle the object.
