# Constrained shelf retrieval and transfer

The default MuJoCo demonstration uses a **single FR3v2 and a two-tier cabinet**.
It samples lateral and oblique grasps inside the lower bay, transfers the mustard
bottle to the upper tier, then extracts a can and foam brick for tabletop placement.
The 28 cm tier spacing makes approach clearance matter. Its purpose is to make
grasp generation, rejection, selection, physical closure, and extraction visible.
This is an illustrative scene inspired by the references below, not a
reproduction of their benchmarks or a claim of comparable performance.

| Reference | Published setup | Design consequence here |
| --- | --- | --- |
| [MotionBenchMaker (RA-L 2022)](https://arxiv.org/html/2112.06402v1) | Shelf picking, including a UR5/cylinder example with varied shelf/object poses; box picking with Panda | Use a compact shelf and object-relative grasps, with explicit clearance checks |
| [RoboCasa (RSS 2024)](https://robocasa.ai/assets/robocasa_rss24.pdf) | Cabinet-to-counter atomic pick-and-place starts with the cabinet already open | Keep retrieval separate from articulated door opening |
| [RoboTwin 2.0](https://robotwin-platform.github.io/doc/tasks/) | Dual-arm handover, dual-bottle picking, pot lifting and cabinet-placement tasks | Bimanual handover is a possible separate extension, with joint grasp-pair and inter-arm collision reasoning |
| [ALOHA 2](https://aloha-2.github.io/) | Bimanual teleoperation hardware, with an identified MuJoCo model | Useful reference for a future two-arm task; does not require adding a second arm to this grasp demonstration |

The selected scene uses existing `assets/meshes/` scans:

- `ycb_mustard_bottle_metric_fp.glb`
- `ycb_potted_meat_can_metric_fp.glb`
- `ycb_foam_brick_metric_fp.glb`

The native object bodies use those mesh vertices/faces. MuJoCo's convex dynamic
mesh contact geometry is matched by sampling on the convex hull; the visible
object remains the original decimated mesh. Mesh normalization and world poses
stay consistent. Frontal and tilted world approach facets are transformed into
the current object frame, including after physical settling/tipping. Candidate
generation is not restricted to a vertical approach.

The sequence approaches, closes, lifts slightly to unseat the object, extracts
through the cabinet opening, then lifts outside the shelf. The bottle is inserted
into the upper tier; the other objects are reoriented outside the cabinet before
tabletop placement. No object attachment constraints or object teleportation are used.
Joint IK and contact screening use a scratch MuJoCo data instance; the live
world is advanced with native actuators and frictional contacts. The bottle's
candidate screening checks upper-tier IK and collisions as well as pickup
feasibility. Selection combines joint travel and contact-centering costs.

Free-space joint moves are collision checked against the complete MuJoCo scene,
including the held object's measured tool-relative geometry. A direct joint path
is used when clear; otherwise a small bidirectional RRT-Connect planner finds a
route and collision-checks its shortcuts. All of this happens in scratch data:
the live object is carried by contact, not by a weld or state reset. Joint edges
are checked at a maximum 0.04-radian spacing; this is discrete collision checking,
not a continuous swept-volume guarantee. The planner adapters can replace this
demonstration controller with an application's production motion planner.

The HUD shows phase, candidate/rejection counts, selected ID, and placed-object
count. Cyan TCP glyphs are sampled candidates, coral glyphs are rejected, and
gold highlights the selected command and staging targets. A small native figure
shows feasible candidates' selection costs. Those costs are not robustness
probabilities. The robot's full source visual material regions preserve the
white shell, dark accents and markings; native Menagerie dynamics are retained.

```bash
pip install -e ".[mujoco,fr3]"
python tools/prepare_fr3_mujoco.py
# macOS: use mjpython for the native viewer
PYTHONPATH=src mjpython examples/pick_place_mujoco.py --gui --hold
# Linux: python examples/pick_place_mujoco.py --gui --hold
```

On macOS the example selects the trimesh collision backend via
`GRASP_SAMPLER_DISABLE_OPEN3D=1` to avoid Open3D/MuJoCo's duplicate GLFW classes.
The default backend for other callers is unchanged.

```bash
PYTHONPATH=src python examples/pick_place_mujoco.py --headless \
    --report /tmp/shelf-picks.json --snapshot /tmp/shelf.png
```

The [recorded MuJoCo run](results/mujoco-shelf-pick-place.json) checks bilateral
finger contact and a lift of at least 8 cm above the initial object height.
Placement requires load-bearing contact on the requested tier/table, separation
from the gripper, low settling velocity, a bottom height within 15 mm of the
surface, and less than 4 cm XY error. Placement orientation is free: a bottle
settled on its side can pass, while the same bottle on the wrong tier cannot.
Candidate IDs, rejection reasons, approach obliquity, selection costs and support
forces are recorded. This is a single seeded demonstration, not a multi-seed
benchmark success-rate estimate.
