# Visualization

## Live 8 × 5 object gallery

The native MuJoCo gallery shows **all 40 object variants simultaneously**, with
one slow 360° revolution per minute. Genuine grasp batches are generated in two
sampling processes while the viewer keeps rotating. No prerecorded candidate
animation is used.

```{raw} html
:file: _includes/gallery-video.html
```

```bash
pip install -e ".[mujoco]"
# macOS
PYTHONPATH=src mjpython examples/grasp_gallery.py
# Linux
PYTHONPATH=src python examples/grasp_gallery.py
```

Bright contact beads, gripper outlines and approach arrows make candidates
visible against the original object colors. **Cyan** denotes antipodal
candidates, **violet** primitive/OBB proposals, and **gold** recent arrivals.
The view has a plain background with no HUD, text, stands or dials. The drawing
shows a spatially thinned subset (four per object initially) to keep the grid
readable; the full candidate counts are written to the console and JSON report.

| Control | Action |
| --- | --- |
| `Space` | Pause/resume rotation and new sampling submissions |
| `+` / `−` | Increase/decrease visible candidates per object |
| `G` | Show/hide grasp glyphs |
| `R` | Resample the complete grid |
| Mouse drag / wheel | Orbit / zoom |
| `Q` / `Esc` | Close |

```bash
# Slower rotation, or more visible candidates:
PYTHONPATH=src mjpython examples/grasp_gallery.py --seconds 90 --visible-grasps 8
# Single-object turntable, with N/P or arrow-key navigation:
PYTHONPATH=src mjpython examples/grasp_gallery.py --layout turntable --seconds 12
# Audit every asset and save an image without opening a window:
PYTHONPATH=src python examples/grasp_gallery.py --headless --batches 2 \
  --report /tmp/gallery.json --snapshot /tmp/gallery.png
```

The inventory includes 31 compact meshes plus nine scanned/reconstructed
household variants. OBJ/PNG sidecars and robot link meshes are not extra objects.
The compact GLBs were originally geometry-only; appearance-only sidecars restore
colors from their named original workspace reconstructions and YCB scans. This
transfer is approximate, using bounds alignment and a triangle UV atlas; it does
not preserve the dense scans' original UV fidelity. Existing household texture
appearances are retained. See the
[appearance provenance](https://github.com/ShirakuGIT/grasp-sampler/blob/master/assets/gallery_colors/README.md).
Lighting is neutral, and objects receive no palette tint.

Objects, grasp TCPs and jaw openings share the **same original metre scale**.
There is no per-object size normalization. The Franka gripper has a hard
**80 mm maximum opening**; `--jaw-width` can request a smaller opening.
The larger master-chef can has no feasible candidates at that width in the
40-object gallery; that cell has no grasp glyphs.
These are geometric grasp proposals, not robot IK, collision-screened motion
plans or verified physical picks. Glyph finger/stem lengths are schematic;
TCP positions, orientation and jaw opening follow the actual sampler outputs.

## Record the MuJoCo demos

Install FFmpeg on your system, then record without opening a viewer window:

```bash
pip install -e ".[mujoco,fr3]" "mujoco==3.17.0"
PYTHONPATH=src python examples/grasp_gallery.py --headless --once \
  --seconds 60 --fps 30 --batches 8 \
  --record docs/_static/videos/grasp-gallery-live.mp4 \
  --report /tmp/grasp-gallery-recording.json
PYTHONPATH=src python examples/pick_place_mujoco.py --headless \
  --fps 30 --speed 2 --record docs/_static/videos/mujoco-live-pick-place.mp4 \
  --report /tmp/mujoco-recording-pick-place.json
```

The gallery recording uses the same live worker loop as the interactive viewer;
candidate sets are generated during capture. Without `--record`, `--headless`
retains the faster asset-audit behavior. `--once` stops after a complete gallery
revolution and all requested batches. The MP4s use H.264, 30 FPS and progressive
loading. The optional gallery report records batch arrival times in video seconds.

The robot recording samples each object's current settled pose, displays the
candidate and selected grasp sets, and captures the native physics run at 2×.
Selection holds freeze physics just as they do in the interactive viewer.
Physics retains its 2 ms timestep. These recordings use MuJoCo 3.17.0; physical
outcomes can vary across simulator versions. The optional physical report records
measured contacts, lifts and placements for each capture.

## Interactive uncertainty viewer

```bash
python examples/visualize_robust.py assets/meshes/pocky_box_metric_fp.glb
```

Requires `pip install -e '.[robust-viz]'`. The left 3D view shows sampled nominal grasp
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

To view a generated robust set instead of raw candidates (each grasp passed the
`R >= threshold` test on an independent validation bank):

```bash
python examples/visualize_robust_set.py assets/meshes/ycb_cracker_box_metric_fp.glb
```

Options include `--count`, `--threshold`, `--hypotheses`, `--budget`,
`--backend point|pad|quasistatic` and the same uncertainty bounds as above.

Headless snapshot (also useful over SSH):

```bash
python examples/visualize_robust.py assets/meshes/pocky_box_metric_fp.glb --snapshot /tmp/robust-grasp.png --sample 1
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
