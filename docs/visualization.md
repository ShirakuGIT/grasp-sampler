# Visualization

## Interactive uncertainty viewer

```bash
PYTHONPATH=src conda run --no-capture-output -n mcr-robotics python examples/visualize_robust.py assets/meshes/pocky_box_metric_fp.glb
```

Uses Matplotlib (already available in `mcr-robotics`; elsewhere install
`pip install -e '.[robust-viz]'`). The left 3D view shows sampled nominal grasp
candidates. The right shows the selected **fixed world gripper command** in blue,
the nominal object as a gray ghost, and the hypothetical object in green/red for
geometric pass/fail. Orange markers show contact points when found. Both views
can be orbited with the mouse. The dashed blue line shows the fixed approach.

- Candidate slider selects a new robot command.
- Hypothesis slider varies only the object; index zero is the nominal object.
- Play/pause steps through discrete Monte Carlo samples.
- Next failure jumps to a failed hypothesis and shows its reason.
- Combined, Translation, Orientation, and Shape/size modes reuse the same draws
  to inspect uncertainty sources separately. Each mode has its own success rate.

Customize bounds with `--translation-mm 10 7 12 --rotation-deg 3 3 5
--scale-percent 3 3 3`. Use `--hypotheses 100 --candidates 12 --seed 42` to control
sample counts. Mesh frames are preserved and inputs must use metres. Shape/size
currently means anisotropic scaling about the mesh origin, not local deformation.
The viewer shows the generated grasp set, not internal sampler ray-search steps.
Success means the geometric evaluator passes, not dynamic lift success.

Headless snapshot (also useful over SSH):

```bash
PYTHONPATH=src conda run -n mcr-robotics python examples/visualize_robust.py assets/meshes/pocky_box_metric_fp.glb --snapshot /tmp/robust-grasp.png --sample 1
```

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
