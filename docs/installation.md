# Installation

```bash
pip install -e .                 # core (numpy, trimesh, scipy)
pip install -e ".[viz]"          # + interactive trimesh viewer
pip install -e ".[sim]"          # + pybullet collision filter and viewer
```

Or install just the runtime dependencies:

```bash
pip install -r requirements.txt
```

Python 3.9+ is supported.

## Optional extras

| Extra  | Adds                          | Pulls in            |
|--------|-------------------------------|---------------------|
| `viz`  | `grasp_sampler.viz` viewer    | `pyglet<2`          |
| `sim`  | `grasp_sampler.pybullet_sim`  | `pybullet`, `imageio` |
| `docs` | building this documentation   | `sphinx`, `furo`, … |
