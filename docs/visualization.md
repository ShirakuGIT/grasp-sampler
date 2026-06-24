# Visualization

## Trimesh viewer

```bash
pip install -e ".[viz]"
python examples/visualize.py assets/meshes/pocky_box_metric_fp.glb
python examples/visualize.py object.glb --methods antipodal --max 40
```

Opens the mesh with one gripper wireframe per grasp, colored by generator:
blue = top, green = side, orange = obb, red = antipodal.

From Python:

```python
from grasp_sampler import GraspSampler
from grasp_sampler.viz import show_grasps

sampler = GraspSampler()
obj = sampler.load("object.glb")
grasps = sampler.generate(obj)
show_grasps(obj, grasps, max_grasps=40)
```

## PyBullet viewer

```bash
pip install -e ".[sim]"
python examples/pybullet_demo.py assets/meshes/pocky_box_metric_fp.glb
```

Spawns the object and a parallel-jaw gripper proxy at each (collision-filtered)
grasp. Use `--no-gui` to filter only.

## Sample meshes

`assets/meshes/` ships 31 ready-to-use objects (YCB + household items),
decimated to ~2k faces with textures removed (~36 KB each). Re-decimate your own
library — large foundation scans run to many megabytes — with:

```bash
python tools/decimate_meshes.py <src_dir> assets/meshes --faces 2000
```

Decimation matters for speed too: sampling a 490k-face scan takes ~60 s; the same
object at 2k faces takes under a second.
