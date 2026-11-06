"""Table, mounting hardware, fixtures and inspection graphics for the demo."""
from __future__ import annotations

import numpy as np
import pybullet as p
import trimesh

TABLE_Z = .72
BASE_Z = .755
NAVY = [.045, .065, .10, 1]
STEEL = [.26, .32, .38, 1]


def box(size, position, color, collision=True):
    half = (np.asarray(size) / 2).tolist()
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half) if collision else -1
    return p.createMultiBody(0, shape, visual, position)


def build_workstation():
    fixtures = []
    # Neutral studio backdrop keeps the white arm legible and removes the
    # default checkerboard/horizon from the workstation view.
    box([6, .04, 3.2], [.45, 1.4, 1.6], [.085, .115, .15, 1], collision=False)
    box([.04, 6, 3.2], [-1.1, 0, 1.6], [.10, .135, .17, 1], collision=False)
    table = box([1.28, 1.06, .045], [.44, 0, TABLE_Z - .0225], [.72, .77, .78, 1])
    fixtures.append(table)
    # Structural aluminium frame with inset dark legs and rubber feet.
    for x in [-.13, 1.01]:
        for y in [-.45, .45]:
            box([.055, .055, .67], [x, y, .36], STEEL)
            box([.09, .09, .03], [x, y, .015], NAVY)
    for y in [-.45, .45]:
        box([1.18, .04, .065], [.44, y, .60], STEEL)
    box([.19, .19, BASE_Z - TABLE_Z], [0, 0, (BASE_Z + TABLE_Z) / 2], STEEL)
    for x in [-.075, .075]:
        for y in [-.075, .075]:
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.009, length=.004,
                                         rgbaColor=[.1, .12, .15, 1])
            p.createMultiBody(0, -1, visual, [x, y, BASE_Z + .002])
    # Picking mat and a divided destination tray with low physical walls.
    box([.36, .29, .002], [.45, -.18, TABLE_Z + .001], [.09, .14, .18, 1], collision=False)
    tray_z = TABLE_Z + .006
    fixtures.append(box([.53, .17, .012], [.52, .27, TABLE_Z + .006], NAVY))
    for y in [.18, .36]:
        fixtures.append(box([.55, .012, .028], [.52, y, TABLE_Z + .026], STEEL))
    for x in [.245, .795]:
        fixtures.append(box([.012, .19, .028], [x, .27, TABLE_Z + .026], STEEL))
    # A low fixture between lanes makes overhead transfer intentional.
    fixtures.append(box([.25, .04, .085], [.60, .065, TABLE_Z + .0425], [.22, .28, .34, 1]))
    # Rear instrument station: enclosure, status lamp, and camera mast.
    box([.18, .19, .10], [.91, -.33, TABLE_Z + .05], NAVY)
    box([.014, .014, .40], [.95, -.39, TABLE_Z + .30], STEEL)
    box([.12, .055, .045], [.91, -.39, TABLE_Z + .51], NAVY)
    lamp = p.createVisualShape(p.GEOM_SPHERE, radius=.013, rgbaColor=[.2, .95, .65, 1])
    p.createMultiBody(0, -1, lamp, [.91, -.33, TABLE_Z + .12])
    # Background inspection plinth, intentionally visual-only.
    box([.32, .30, .014], [.89, -.02, TABLE_Z + .24], NAVY, collision=False)
    box([.028, .028, .24], [.89, -.02, TABLE_Z + .12], STEEL, collision=False)
    return fixtures, tray_z + .006


def object_mesh(index):
    if index in (1, 4):
        radius, height = (.019, .065) if index == 1 else (.018, .055)
        return trimesh.creation.cylinder(radius=radius, height=height, sections=32), "can"
    if index == 3:
        return trimesh.creation.cylinder(radius=.022, height=.055, sections=6), "hexagonal jar"
    size = [[.034, .036, .05], [.03, .048, .06], [.042, .035, .07]][(0 if index == 0 else
                                                                                 1 if index == 2 else 2)]
    return trimesh.creation.box(extents=size), "carton"


class Overlay:
    """Persistent phase labels and real candidate inspection in the 3D scene."""
    def __init__(self):
        self.text = {}
        self.visuals = []
        self.paths = []
        self.title("FRANKA  FR3v2  /  GRASP LAB", "Six objects  |  physics-based pick & place")
        p.addUserDebugText("PICK / PERCEPTION", [.27, -.36, TABLE_Z + .015], [.12, .22, .28], textSize=1.25)
        p.addUserDebugText("SORT / PLACE", [.28, .40, TABLE_Z + .015], [.12, .22, .28], textSize=1.25)
        p.addUserDebugText("CANDIDATE INSPECTOR  x2.5", [.73, -.12, TABLE_Z + .62], [.12, .22, .28], textSize=1.15)

    def label(self, key, text, position, color=(.85, .94, .96), size=1.3):
        self.text[key] = p.addUserDebugText(text, position, color, textSize=size,
                                          replaceItemUniqueId=self.text.get(key, -1))

    def title(self, title, subtitle):
        self.label("title", title, [.02, .49, TABLE_Z + .42], size=1.65)
        self.label("subtitle", subtitle, [.02, .49, TABLE_Z + .36], size=1.1)

    def phase(self, text, index, count):
        self.label("phase", f"{index + 1:02d}/{count:02d}   {text}", [.02, .49, TABLE_Z + .27],
                   color=(.10, .48, .55), size=1.45)

    def clear_candidates(self):
        for body in self.visuals:
            p.removeBody(body)
        for item in self.paths:
            p.removeUserDebugItem(item)
        self.visuals.clear()
        self.paths.clear()

    def candidates(self, mesh, grasps, world, selected, rejected, cost):
        from grasp_sampler.integrations.pybullet import draw_grasps

        self.clear_candidates()
        self.visuals += draw_grasps(grasps, world, selected_index=selected, rejected_indices=rejected, max_grasps=10)
        inset = np.eye(4)
        inset[:3, 3] = [.89, -.02, TABLE_Z + .34]
        self.visuals += draw_grasps(grasps, inset, selected_index=selected,
                                     rejected_indices=rejected, max_grasps=12, scale=2.5)
        # The inset shows the actual sampled object geometry and TCP glyphs,
        # enlarged together without modifying the physical object's geometry.
        shape = p.createVisualShape(p.GEOM_MESH, vertices=(mesh.vertices * 2.5).tolist(),
                                     indices=mesh.faces.reshape(-1).tolist(), rgbaColor=[.7, .77, .80, 1])
        self.visuals.append(p.createMultiBody(0, -1, shape, inset[:3, 3].tolist()))
        selection = "awaiting selection" if selected is None else f"selected #{selected}"
        self.label("stats", f"{len(grasps)} candidates  /  {len(rejected)} rejected  /  {selection}",
                   [.02, .49, TABLE_Z + .19], size=1.1)
        suffix = "" if cost is None else f"  cost {cost:.2f}"
        self.label("legend", "CYAN sampled  |  RED infeasible  |  GOLD selected" + suffix,
                   [.02, .49, TABLE_Z + .13], size=.95)

    def path(self, targets):
        points = [targets.pregrasp[0, :3, 3], targets.grasp[0, :3, 3], targets.lift[0, :3, 3]]
        for start, end in zip(points, points[1:]):
            self.paths.append(p.addUserDebugLine(start, end, [1., .65, .08], lineWidth=3))

    def close(self):
        self.clear_candidates()
