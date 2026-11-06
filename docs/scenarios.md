# Constrained shelf retrieval and transfer

The default MuJoCo demonstration uses a **single FR3v2 and a two-tier cabinet**.
The default `--suite expanded` includes a Ritz box, mug, cup, plate, bowl,
curved mustard bottle and blue vase, preserving original mesh colors. The box, cup and vase move
from the lower tier to separate upper-tier slots. Mug, plate and bottle move
to the tabletop, while the bowl uses an oblique rim grasp to move across the
table. `--suite household` retains the earlier four-object demonstration. `--suite classic` selects the
earlier mustard-bottle, can and foam-brick sequence.
The lower bay retains 28 cm tier spacing; the expanded scene has a taller
40 cm upper bay for the bottle. Approach clearance still matters. Its purpose is to make
grasp generation, rejection, selection, physical closure, and extraction visible.
This is an illustrative scene inspired by the references below, not a
reproduction of their benchmarks or a claim of comparable performance.

| Reference | Published setup | Design consequence here |
| --- | --- | --- |
| [MotionBenchMaker (RA-L 2022)](https://arxiv.org/html/2112.06402v1) | Shelf picking, including a UR5/cylinder example with varied shelf/object poses; box picking with Panda | Use a compact shelf and object-relative grasps, with explicit clearance checks |
| [RoboCasa (RSS 2024)](https://robocasa.ai/assets/robocasa_rss24.pdf) | Cabinet-to-counter atomic pick-and-place starts with the cabinet already open | Keep retrieval separate from articulated door opening |
| [RoboTwin 2.0](https://robotwin-platform.github.io/doc/tasks/) | Dual-arm handover, dual-bottle picking, pot lifting and cabinet-placement tasks | Bimanual handover is a possible separate extension, with joint grasp-pair and inter-arm collision reasoning |
| [ALOHA 2](https://aloha-2.github.io/) | Bimanual teleoperation hardware, with an identified MuJoCo model | Useful reference for a future two-arm task; does not require adding a second arm to this grasp demonstration |

The scenes use original textured workspace scans in `assets/household/`.
YCB objects retain source UV coordinates and texture pixels. The Ritz box retains
the existing decimated geometry, with original dense reconstruction vertex colors
baked onto a UV atlas. No object palette overrides these appearances. See
[asset attribution and preparation](https://github.com/ShirakuGIT/grasp-sampler/blob/feat/planner-integrations/assets/household/README.md).

Mug, cup and plate contact geometry consists of separate CoACD convex pieces,
leaving the handle opening and vessel cavities unfilled. Grasp sampling uses their
actual scanned surfaces, including a handle-region filter for the mug and a rim
filter with vertical finger closure for the plate. Box and classic objects use
convex hull contact geometry. These decompositions approximate the scans; demo
masses are 60 g (30 g for the lightweight plate) and do not claim YCB inertia fidelity. The household gripper has
a 16 N tendon preload and higher closing stiffness, within the native +/-40 N actuator
limit, to hold thin handle and rim contacts. Household pads use six-dimensional
frictional contact, with 8 mm rolling-resistance length representing their
finite 17 mm rubber contact surface. The central pad proxy extends 1.5 mm toward
the midline so the pads meet at zero opening, removing the stock proxy's 3 mm
dead gap for thin rims. Placement remeasures the object after transit and uses
pose feedback during descent to account for settling within the finger pads.

Mesh normalization and world poses stay consistent. Frontal and tilted world approach facets are transformed into
the current object frame, including after physical settling/tipping. Candidate
generation is not restricted to a vertical approach.

The sequence approaches, closes, lifts slightly to unseat the object, extracts
through the cabinet opening, then lifts outside the shelf. The box is inserted
into the upper tier; vessels and the plate turn around the vertical axis to
approach the table edge while preserving their pickup orientation relative to
gravity. Rim release uses a small opening before retracting, keeping the lower
finger above the table. The classic suite reorients its table objects from the side.
No object attachment constraints or object teleportation are used.
Joint IK and contact screening use a scratch MuJoCo data instance; the live
world is advanced with native actuators and frictional contacts. The box's
candidate screening checks upper-tier IK and collisions as well as pickup
feasibility. Household screening also checks extraction/lift IK, collisions with
the carried geometry at lift, and table placement IK over several yaw angles.
The approach reuses its screened IK solution. Selection combines joint travel
and contact-centering costs, with a preference for horizontal rim insertion on
the plate. These costs describe selection heuristics, not grasp probabilities.

Free-space joint moves are collision checked against the complete MuJoCo scene,
including the held object's measured tool-relative geometry. A direct joint path
is used when clear; otherwise a small bidirectional RRT-Connect planner finds a
route and collision-checks its shortcuts. All of this happens in scratch data:
the live object is carried by contact, not by a weld or state reset. Joint edges
are checked at a maximum 0.04-radian spacing; this is discrete collision checking,
not a continuous swept-volume guarantee. The planner adapters can replace this
demonstration controller with an application's production motion planner.

Cyan TCP glyphs are sampled candidates, coral glyphs are rejected, and gold
highlights the selected command and staging targets. The viewer has no text HUD
or cost chart. Phase updates appear in the terminal, and feasible selection
costs and rejection reasons are saved in the JSON report. Those costs are not
robustness probabilities. The robot's full source visual material regions preserve the
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

The [household run](results/mujoco-household-pick-place.json) and
[classic MuJoCo run](results/mujoco-shelf-pick-place.json) check force-bearing bilateral
finger contact and a lift of at least 8 cm above the initial object height.
Placement requires load-bearing contact on the requested tier/table, separation
from the gripper, low settling velocity, a bottom height within 15 mm of the
surface, and less than 4 cm XY error in both suites. Reports expose the actual
error and the tolerance used.
Placement orientation is free: a bottle
settled on its side can pass, while the same bottle on the wrong tier cannot.
Candidate IDs, rejection reasons, approach obliquity, selection costs and support
forces are recorded. This is a single seeded demonstration, not a multi-seed
benchmark success-rate estimate.

## Playback performance

The native viewer defaults to `--speed 2 --fps 60`. Playback pacing and redraws
are independent, and rendering is limited by wall time rather than every fixed
number of physics steps. Sleep is batched while honoring the next frame deadline. Frame deadlines
include synchronization work rather than adding a full interval afterward.
Static glyphs are cached, and viewer synchronization sends integration state
instead of repeatedly copying the complete visual model.
Physics remains MuJoCo's compiled CPU solver at a 2 ms timestep; faster playback
does not skip contact integration or speed up motor commands in simulation time.
Candidate-display pauses refresh only the viewer; they do not advance physics.
This keeps live grasp sampling consistent with the headless validation sequence.
The requested frame rate is a target, not a hardware-independent guarantee.
JAX/MJX is useful for batched simulation workloads; this single native viewer
uses the existing CPU solver and does not introduce a JAX dependency.

The [expanded seven-object physical run](results/mujoco-expanded-pick-place.json)
completed all seven transfers, including three lower-to-upper shelf placements.
Maximum XY placement error was 3.82 cm against the unchanged 4 cm tolerance.
The run requires bilateral finger forces, an 8 cm lift, released and settled
load-bearing contact on the requested support, and the correct object-bottom
height. [Expanded scene snapshot](results/mujoco-expanded.png).

A local M4 / Retina timing check measured 60.0 scene updates per second, with
a 95th-percentile update interval of 16.94 ms, after removing bitmap-text HUDs,
disabling the table mirror pass, and using 2048-pixel shadows. The earlier loop
measured 27.1 updates/s. The [timing record](results/mujoco-viewer-timing.json) measures scene-state
synchronization, not physical display presentation; performance depends on the machine and workload.

## Arm mounting and curved vase

The arm sits on a 240 × 240 × 12 mm aluminum plate with chamfered corners,
beveled edges, four 22 mm counterbores and 9 mm through holes. The counterbore
shoulders support the washers and recessed cap screws. A 23 mm pedestal and
registration ring keep the original robot base height. This is illustrative
mounting hardware, rather than a manufacturer-certified plate design. It is
fixed visual geometry, as was the previous mount; robot dynamics are unchanged.

The blue vase has a bulbous body and narrow neck. Its original image and UVs
are retained, with 29 convex contact pieces. Sampling uses the neck's actual
surface, and a longer extraction clears the wide body before lifting. Its
lower-to-upper transfer passed with a 6 mm XY placement error in the complete
seven-object physical run. Like the other example objects, its mass and inertia
are illustrative rather than measured hardware parameters.
