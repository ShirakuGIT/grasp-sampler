"""Regression coverage for viewer-induced stutter without changing physics."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def workstation_module(monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"examples"))
    import pick_place_mujoco
    return pick_place_mujoco


def test_render_cost_is_inside_frame_budget_and_physics_is_unchanged(monkeypatch):
    module = workstation_module(monkeypatch)
    mj = module.mj
    model = mj.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body><joint type="slide"/><geom size=".01" mass="1"/></body></worldbody>
      </mujoco>''')
    data, reference = mj.MjData(model), mj.MjData(model)
    for state in (data, reference):
        state.qpos[:] = .1
        state.qvel[:] = .2
        mj.mj_forward(model, state)
    clock = SimpleNamespace(now=10.)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "sleep", lambda duration: setattr(clock, "now", clock.now+duration))
    demo = object.__new__(module.Workstation)
    demo.model, demo.data, demo.dofs = model, data, np.array([0])
    demo.speed, demo.fps, demo.next_frame = 2., 60, 10.
    demo.viewer = SimpleNamespace(is_running=lambda: True)
    frames = []

    def draw():
        frames.append(clock.now)
        clock.now += .004  # Four milliseconds of synchronization/render work.

    demo.update_viewer = draw
    demo.step(1.)
    mj.mj_step(model, reference, nstep=500)
    assert 29 <= len(frames) <= 31
    assert clock.now == pytest.approx(10.5, abs=.002)
    assert np.percentile(np.diff(frames), 95) < .019
    assert data.time == pytest.approx(1.)
    np.testing.assert_allclose(data.qpos, reference.qpos)
    np.testing.assert_allclose(data.qvel, reference.qvel)


def test_viewer_has_no_text_or_chart_and_only_syncs_state(monkeypatch):
    module = workstation_module(monkeypatch)
    from collections import deque

    class Viewer:
        user_scn = SimpleNamespace(ngeom=0)

        def __init__(self):
            self.syncs = []

        def lock(self):
            return nullcontext()

        def set_texts(self, *args):
            pytest.fail("The viewer must not reintroduce bitmap text HUDs")

        def set_figures(self, *args):
            pytest.fail("The cost chart was removed with the HUD")

        def sync(self, *, state_only=False):
            self.syncs.append(state_only)

    demo = object.__new__(module.Workstation)
    demo.viewer = Viewer()
    demo.grasps, demo.pick_path, demo.rejected = [], [], {}
    demo.selected, demo.phase, demo.overlay_key = None, "SAMPLING", None
    demo.frame_times, demo.sync_times = deque(), deque()
    demo.update_viewer()
    demo.update_viewer()
    assert demo.viewer.syncs == [True, True]
