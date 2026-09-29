# Installation

```bash
pip install -e .                 # core (numpy, trimesh, scipy, rtree, open3d)
pip install -e ".[viz]"          # + interactive trimesh viewer
pip install -e ".[robust-viz]"   # + matplotlib uncertainty viewer
pip install -e ".[sim]"          # + pybullet collision filter and viewer
```

Or install just the runtime dependencies:

```bash
pip install -r requirements.txt
```

Python 3.9+ is supported.

Open3D provides the gripper-body collision queries. Non-watertight meshes, or an
environment without Open3D, fall back to trimesh automatically (much slower).

## Optional extras

| Extra        | Adds                           | Pulls in              |
|--------------|--------------------------------|-----------------------|
| `viz`        | `grasp_sampler.viz` viewer     | `pyglet<2`            |
| `robust-viz` | uncertainty viewer             | `matplotlib`          |
| `sim`        | `grasp_sampler.pybullet_sim`   | `pybullet`, `imageio` |
| `docs`       | building this documentation    | `sphinx`, `furo`, …   |
| `dev`        | tests and linting              | `pytest`, `ruff`      |
