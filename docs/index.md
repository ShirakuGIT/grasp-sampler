# grasp-sampler

A simple, dependency-light **parallel-jaw grasp sampler** for mesh objects.
Point it at a mesh, get back grasp poses ready for IK / motion planning.

```python
from grasp_sampler import GraspSampler

sampler = GraspSampler()
grasps = sampler.sample("object.glb")
print(len(grasps), "grasps")
```

It combines three classic generators behind one API:

| Generator    | What it does                                          |
|--------------|-------------------------------------------------------|
| `primitives` | Top-down and side grasps on the bounding box          |
| `obb_face`   | Grasps centered on the oriented bounding box faces    |
| `antipodal`  | Dex-Net-style force-closure contacts via ray casting  |

No simulator, no robot model, no GPU — just NumPy, trimesh, and SciPy. An
optional PyBullet module adds collision checking and a 3-D viewer.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
usage
visualization
configuration
api
```
