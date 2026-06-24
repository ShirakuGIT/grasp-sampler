"""Core data types and the sampler configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Grasp:
    """A single parallel-jaw grasp in the object's local frame.

    The pose follows the standard tool-center-point (TCP) convention:

        +Z  approach axis  (the hand travels along +Z to reach the object)
        +X  closing axis   (the line joining the two fingertips)
        +Y  right-handed completion

    The TCP sits at the midpoint between the two fingertips.
    """

    pose: np.ndarray              # (4, 4) TCP pose in the object's local frame
    kind: str                     # "top" | "side" | "obb_face" | "antipodal"
    width: float                  # finger opening required (m)
    meta: dict = field(default_factory=dict)


@dataclass
class ObjMesh:
    """A loaded mesh, re-centered and oriented upright in its local frame."""

    mesh: object                  # trimesh.Trimesh, centered at the OBB centroid
    extents: np.ndarray           # (3,) axis-aligned full extents (m)
    obb_transform: np.ndarray     # (4, 4) oriented-bounding-box -> local frame
    obb_extents: np.ndarray       # (3,) oriented-bounding-box full extents (m)
    geom_class: str               # "box" | "cylinder" | "sphere" | "irregular"


# Default approach facets: front (-Y), both sides (+/-X) and top (+Z). In directed
# mode the antipodal sampler keeps only grasps approaching from these directions.
_DEFAULT_FACETS = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass
class GraspConfig:
    """All sampler tunables in one place. Defaults target a Franka Panda gripper."""

    # --- gripper limits ---------------------------------------------------
    gripper_max_width: float = 0.08      # max finger opening (m)
    gripper_min_width: float = 0.005     # below this the fingers effectively touch

    # --- bounding-box primitives (top / side) -----------------------------
    primitive_yaw_steps: int = 4         # yaws sampled for top-down grasps
    primitive_roll_steps: int = 2        # rolls sampled for side grasps

    # --- antipodal sampler ------------------------------------------------
    antipodal_min_width: float = 0.02    # reject pairs narrower than this (m)
    antipodal_samples: int = 1500        # surface points ray-cast per object
    antipodal_rolls: int = 8             # approaches per pair in undirected mode
    antipodal_max_grasps: int = 200      # cap kept grasps (keeps downstream fast)
    friction_coef: float = 0.5           # Coulomb mu; cone half-angle = atan(mu)
    wrist_stem: float = 0.10             # wrist sits this far behind the TCP (m)
    finger_len: float = 0.04             # fingertip-to-palm depth (m)
    spine_clearance: float = 0.005       # min gap between gripper body and mesh (m)
    hull_volume_ratio: float = 0.40      # below this mesh/hull ratio, treat as hollow

    # --- directed antipodal (approach filtering) --------------------------
    directed: bool = True                # keep only grasps near `approach_facets`
    approach_facets: tuple = _DEFAULT_FACETS
    cone_deg: float = 50.0               # half-angle of the per-facet approach cone
    directed_spread: int = 3             # approaches sampled within each cone

    def approach_facet_array(self) -> np.ndarray:
        """Unit approach facets as an (N, 3) array; degenerate facets dropped."""
        out = []
        for facet in self.approach_facets:
            v = np.asarray(facet, float)
            n = np.linalg.norm(v)
            if n > 1e-9:
                out.append(v / n)
        return np.asarray(out, float) if out else np.zeros((0, 3), float)
