"""Headless MuJoCo mocap target round trip (no robot model required).

PYTHONPATH=src python examples/mujoco_targets.py /tmp/grasps.json

For a robot scene, calibrate targets to its controlled site and connect the
mocap goal to your IK/controller. A mocap target by itself does not move an arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from grasp_sampler import PlanningGrasps
from grasp_sampler.integrations.mujoco import set_mocap_target, site_world_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Path)
    args = parser.parse_args()
    targets = PlanningGrasps.from_dict(json.loads(args.targets.read_text()))
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="target" mocap="true"><site name="tool" size="0.01"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for i in range(len(targets)):
        for stage in ("pregrasp", "grasp", "lift"):
            set_mocap_target(model, data, "target", targets, index=i, stage=stage)
            np.testing.assert_allclose(site_world_pose(model, data, "tool"), targets.poses(stage)[i], atol=1e-10)
    print(f"Verified {len(targets)} x 3 MuJoCo mocap/site targets")


if __name__ == "__main__":
    main()
