"""Interactive Matplotlib inspection of fixed-command robustness (optional).

Candidate sampling is shown as the generated candidate set, not an animation of
the sampler's internal ray search. Hypothesis playback shows discrete Monte Carlo
samples, not physical object motion or simulated closure/lift.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .uncertainty import ObjectHypothesis
from .viz import gripper_glyph


def variation_sets(nominal_mesh, nominal_pose, hypotheses):
    """Paired ablations reuse the same draws; nominal is sample zero in each."""
    modes = {name: [ObjectHypothesis(nominal_mesh, nominal_pose, metadata={"nominal": True})]
             for name in ("Combined", "Translation", "Orientation", "Shape / size")}
    for h in hypotheses:
        translation = nominal_pose.copy()
        translation[:3, 3] = h.world_pose[:3, 3]
        orientation = nominal_pose.copy()
        orientation[:3, :3] = h.world_pose[:3, :3]
        modes["Combined"].append(h)
        modes["Translation"].append(ObjectHypothesis(nominal_mesh, translation, h.weight))
        modes["Orientation"].append(ObjectHypothesis(nominal_mesh, orientation, h.weight))
        modes["Shape / size"].append(ObjectHypothesis(h.mesh, nominal_pose, h.weight, h.metadata))
    return modes


class RobustGraspViewer:
    """Two 3D views, candidate/sample sliders, paired ablations and playback.

    Evaluation is cached per candidate/mode. Reported probabilities exclude the
    extra nominal frame at slider index zero. World commands never vary with
    the hypothesis slider; selecting a different candidate selects a new command.
    """
    def __init__(self, mesh, grasps, nominal_pose, hypotheses, evaluator):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button

        if not grasps or not hypotheses:
            raise ValueError("viewer needs at least one grasp and one hypothesis")
        self.mesh, self.grasps = mesh, list(grasps)
        self.nominal_pose = np.asarray(nominal_pose, float).copy()
        self.commands = [self.nominal_pose @ g.pose for g in self.grasps]
        self.modes = variation_sets(mesh, self.nominal_pose, hypotheses)
        self.evaluator, self.cache = evaluator, {}
        self.grasp_index, self.sample_index, self.mode = 0, 0, "Combined"
        self.playing = False
        self.fig = plt.figure(figsize=(15, 9))
        self.left = self.fig.add_axes([0.02, 0.34, 0.46, 0.60], projection="3d")
        self.right = self.fig.add_axes([0.51, 0.34, 0.46, 0.60], projection="3d")
        self.fig.suptitle("Object uncertainty · fixed world-frame robot command", fontsize=16)
        self.info = self.fig.text(0.35, 0.16, "", fontsize=10, va="top", family="monospace")
        self.fig.text(0.03, 0.02,
            "Drag to orbit · scroll to zoom | Blue: fixed open gripper | Gray: nominal mesh | "
            "Green/red: geometric pass/fail | Orange: contacts\n"
            "Playback steps through sampled hypotheses. No robot execution noise; no dynamic lift validation.", fontsize=10)
        self.candidate_slider = Slider(self.fig.add_axes([0.16, 0.29, 0.76, 0.025]),
            "Candidate", 0, max(1, len(grasps)-1), valinit=0, valstep=1)
        if len(grasps) == 1:
            self.candidate_slider.set_active(False)
        self.sample_slider = Slider(self.fig.add_axes([0.16, 0.245, 0.76, 0.025]),
            "Hypothesis", 0, len(hypotheses), valinit=0, valstep=1)
        self.radio = RadioButtons(self.fig.add_axes([0.03, 0.085, 0.17, 0.13]), list(self.modes))
        self.play_button = Button(self.fig.add_axes([0.22, 0.14, 0.09, 0.04]), "Play / pause")
        self.fail_button = Button(self.fig.add_axes([0.22, 0.085, 0.09, 0.04]), "Next failure")
        self.timer = self.fig.canvas.new_timer(interval=800)
        self.timer.add_callback(self._tick)
        self.candidate_slider.on_changed(self._candidate_changed)
        self.sample_slider.on_changed(self._sample_changed)
        self.radio.on_clicked(self._mode_changed)
        self.play_button.on_clicked(self._toggle_play)
        self.fail_button.on_clicked(self._next_failure)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.timer.stop())
        self._compute_limits()
        self.draw()

    def _compute_limits(self):
        # Stable bounds across all hypotheses and all fixed candidate commands.
        # Rotated AABB corners, not just its two diagonal corners.
        points = [trimesh.transform_points(trimesh.bounds.corners(h.mesh.bounds), h.world_pose)
                  for h in self.modes["Combined"]]
        for command in self.commands:
            points.append(gripper_glyph(command, self.evaluator.opening_width,
                finger_len=self.evaluator.config.finger_len).vertices)
        points = np.vstack(points)
        self.center = (points.min(axis=0) + points.max(axis=0)) / 2
        self.radius = max(float(np.ptp(points, axis=0).max()) * 0.6, 0.01)

    def _result(self):
        key = self.grasp_index, self.mode
        if key not in self.cache:
            print(f"Evaluating candidate {self.grasp_index}, {self.mode}...", flush=True)
            self.cache[key] = self.evaluator.evaluate(self.grasps[self.grasp_index],
                nominal_object_pose=self.nominal_pose, hypotheses=self.modes[self.mode][1:])
        return self.cache[key]

    def _axes(self, ax, title):
        elev, azim = ax.elev, ax.azim
        ax.clear()
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11)
        for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(self.center[i]-self.radius, self.center[i]+self.radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)"); ax.set_zlabel("World Z (m)")
        ax.tick_params(labelsize=8)

    @staticmethod
    def _mesh(ax, mesh, pose, color, alpha):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        vertices = trimesh.transform_points(mesh.vertices, pose)
        ax.add_collection3d(Poly3DCollection(vertices[mesh.faces],
            facecolor=color, edgecolor="none", alpha=alpha))

    def _gripper(self, ax, pose, color, alpha=1, width=2):
        glyph = gripper_glyph(pose, self.evaluator.opening_width,
            finger_len=self.evaluator.config.finger_len,
            stem=max(0, self.evaluator.config.wrist_stem-self.evaluator.config.finger_len))
        # The branched open gripper is not a closed path: Path3D.discrete can
        # be empty. Render the actual line entities instead.
        for entity in glyph.entities:
            line = entity.discrete(glyph.vertices)
            ax.plot(*line.T, color=color, alpha=alpha, linewidth=width)

    def draw(self):
        result = self._result()
        h = self.modes[self.mode][self.sample_index]
        validity = (result.nominal_result if self.sample_index == 0 else
                    result.per_hypothesis_results[self.sample_index-1].validity)
        command = self.commands[self.grasp_index]  # NEVER derived from h.world_pose
        color = "#269b56" if validity.valid else "#d84b46"
        self._axes(self.left, f"Nominal sampled candidates ({len(self.grasps)}) · selected {self.grasp_index}")
        self._mesh(self.left, self.mesh, self.nominal_pose, "#aab4c1", 0.5)
        for i, pose in enumerate(self.commands):
            if i != self.grasp_index:
                self._gripper(self.left, pose, "#888888", 0.25, 1)
        self._gripper(self.left, command, "#176cd3", 1, 3)
        self._axes(self.right, f"{self.mode} · sample {self.sample_index} · "
                   f"{'PASS' if validity.valid else 'FAIL'}: {validity.reason}")
        self._mesh(self.right, self.mesh, self.nominal_pose, "#aab4c1", 0.10)
        self._mesh(self.right, h.mesh, h.world_pose, color, 0.5)
        self._gripper(self.right, command, "#176cd3", 1, 3)
        if validity.contacts is not None:
            contacts = trimesh.transform_points(validity.contacts, h.world_pose)
            self.right.scatter(*contacts.T, color="#ff9d00", s=60, depthshade=False)
        start = command[:3, 3] - self.evaluator.backend.approach_distance * command[:3, 2]
        self.right.plot(*np.array([start, command[:3, 3]]).T, "--", color="#176cd3")
        self.right.scatter(*h.world_pose[:3, 3], color="black", marker="+", s=40)
        dt = (h.world_pose[:3, 3] - self.nominal_pose[:3, 3]) * 1000
        rpy = Rotation.from_matrix(h.world_pose[:3, :3] @ self.nominal_pose[:3, :3].T).as_euler("xyz", degrees=True)
        scale = h.metadata.get("scale_xyz", [1, 1, 1])
        width = "N/A" if validity.contact_width is None else f"{validity.contact_width*1000:.2f} mm"
        self.info.set_text(
            f"World translation delta (mm): {np.round(dt, 2)}\n"
            f"World rotation delta (deg):   {np.round(rpy, 2)}   scale XYZ: {np.round(scale, 3)}\n"
            f"Contact width: {width} | Nominal: {'PASS' if result.nominal_valid else 'FAIL'}\n"
            f"Sample success: {result.weighted_success_probability:.1%} | "
            f"Worst case: {result.worst_case_valid} | n={result.num_hypotheses} (excludes nominal)\n"
            "Shape variation = anisotropic scaling about mesh origin. Hypothesis 0 = nominal.")
        self.fig.canvas.draw_idle()

    def _candidate_changed(self, value):
        self.grasp_index = min(int(value), len(self.grasps)-1)
        self.draw()

    def _sample_changed(self, value):
        self.sample_index = int(value)
        self.draw()

    def _mode_changed(self, value):
        self.mode = value
        self.draw()

    def _toggle_play(self, event):
        self.playing = not self.playing
        self.timer.start() if self.playing else self.timer.stop()

    def _tick(self):
        self.sample_slider.set_val((self.sample_index+1) % len(self.modes[self.mode]))

    def _next_failure(self, event):
        outcomes = self._result().per_hypothesis_results
        indices = [i+1 for i, r in enumerate(outcomes) if not r.success]
        if indices:
            self.sample_slider.set_val(next((i for i in indices if i > self.sample_index), indices[0]))

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()
