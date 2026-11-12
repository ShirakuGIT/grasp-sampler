"""Robot-independent pose targets for IK, motion planning, and simulation.

All distances are metres. ``T_A_B`` maps coordinates in B into A. Tool
calibration is ``T_tcp_tool`` (the planner's tool link expressed in the grasp
TCP), so ``T_frame_tool = T_frame_tcp @ T_tcp_tool``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import stack_poses
from .types import Grasp, ObjMesh
from .uncertainty import validate_pose

Stage = Literal["grasp", "pregrasp", "lift"]
QuaternionOrder = Literal["xyzw", "wxyz"]


def _poses(value):
    poses = np.asarray(value, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    for pose in poses:
        validate_pose(pose)
    return poses.copy()


def pose_components(poses, *, quaternion_order: QuaternionOrder = "xyzw"):
    """Return position (N,3) and unit quaternion (N,4), including empty batches.

    A nonnegative scalar quaternion component makes serialization deterministic
    except at the usual 180-degree rotation ambiguity.
    """
    if quaternion_order not in ("xyzw", "wxyz"):
        raise ValueError("quaternion_order must be 'xyzw' or 'wxyz'")
    poses = _poses(poses)
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat() if len(poses) else np.empty((0, 4))
    q[q[:, 3] < 0] *= -1
    if quaternion_order == "wxyz":
        q = q[:, [3, 0, 1, 2]]
    return poses[:, :3, 3].copy(), q


def mesh_world_pose(obj: ObjMesh, source_world_pose) -> np.ndarray:
    """Convert a perception/file-frame pose to the processed mesh's world pose.

    Accounts for centering and upright rotation applied by ``load_mesh``.
    """
    return validate_pose(source_world_pose) @ np.linalg.inv(validate_pose(obj.source_to_mesh))


@dataclass(frozen=True)
class PlanningGrasps:
    """Aligned grasp/pregrasp/lift tool targets in one named planning frame.

    Use ``from_grasps`` for nominal/local grasps or ``from_robust_result`` to
    preserve the exact world commands accepted by robust evaluation. No IK,
    path feasibility, or physical grasp guarantee is implied by these targets.
    Widths are total jaw opening, not individual actuator positions. Scores,
    when present, are validation-bank contact-model frequencies.
    """

    grasp: np.ndarray
    pregrasp: np.ndarray
    lift: np.ndarray
    widths: np.ndarray
    candidate_ids: np.ndarray
    frame_id: str
    tool_link: str
    scores: np.ndarray | None = None

    def __post_init__(self):
        for name in ("frame_id", "tool_link"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("grasp", "pregrasp", "lift"):
            value = _poses(getattr(self, name))
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        n = len(self.grasp)
        if len(self.pregrasp) != n or len(self.lift) != n:
            raise ValueError("all stage batches must have the same length")
        ids = np.asarray(self.candidate_ids)
        if ids.dtype.kind not in "iu" or ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("candidate_ids must be unique integers with shape (N,)")
        for name in ("widths", "candidate_ids", "scores"):
            raw = getattr(self, name)
            if raw is None:
                if name == "scores":
                    continue
                raise ValueError(f"{name} is required")
            value = np.array(raw, dtype=int if name == "candidate_ids" else float, copy=True)
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape (N,)")
            if name == "widths" and (value < 0).any():
                raise ValueError("widths must be nonnegative")
            if name == "scores" and ((value < 0) | (value > 1)).any():
                raise ValueError("scores must lie in [0, 1]")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.grasp)

    @classmethod
    def from_grasps(cls, grasps: list[Grasp], object_world_pose, **kwargs):
        """Place local grasps using the world pose of their *mesh* frame.

        Use ``mesh_world_pose`` first when perception gives the file/source
        frame instead. Identity tool calibration means the planner targets the
        sampler TCP itself; specify ``tcp_to_tool`` for a flange or other link.
        """
        world = validate_pose(object_world_pose) @ _poses(stack_poses(grasps))
        return cls.from_world_commands(world, [g.width for g in grasps], **kwargs)

    @classmethod
    def from_robust_result(cls, result, **kwargs):
        """Use selected commands directly, without recomputing from object pose.

        IDs refer to indices in ``result.records``. Selection/order and held-out
        validation scores are preserved.
        """
        indices = {id(record): i for i, record in enumerate(result.records)}
        return cls.from_world_commands(
            result.world_commands,
            [r.candidate.grasp.width for r in result.selected],
            candidate_ids=[indices[id(r)] for r in result.selected],
            scores=[r.validation_R for r in result.selected],
            **kwargs,
        )

    @classmethod
    def from_world_commands(
        cls,
        world_tcp_poses,
        widths,
        *,
        frame_id: str,
        tool_link: str,
        tcp_to_tool=None,
        world_to_frame=None,
        approach_distance: float = 0.1,
        lift_offset=(0.0, 0.0, 0.1),
        candidate_ids=None,
        scores=None,
    ):
        """Build targets from fixed world TCP commands.

        ``tcp_to_tool`` is T_tcp_tool; ``world_to_frame`` is T_frame_world.
        Both default to identity. ``lift_offset`` is an XYZ vector in *world*
        axes. Pregrasp backs off along each original TCP's negative Z, before
        tool calibration; lift translates the grasp without changing rotation.
        The strings label the supplied frames; no TF lookup is performed.
        """
        world = _poses(world_tcp_poses)
        tool = validate_pose(np.eye(4) if tcp_to_tool is None else tcp_to_tool)
        frame = validate_pose(np.eye(4) if world_to_frame is None else world_to_frame)
        if not np.isfinite(approach_distance) or approach_distance < 0:
            raise ValueError("approach_distance must be finite and nonnegative")
        offset = np.asarray(lift_offset, float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("lift_offset must be a finite world XYZ vector")
        pre = world.copy()
        pre[:, :3, 3] -= approach_distance * world[:, :3, 2]
        lift = world.copy()
        lift[:, :3, 3] += offset
        ids = np.arange(len(world)) if candidate_ids is None else np.asarray(candidate_ids)
        if len(world) == 0 and ids.shape == (0,):
            ids = ids.astype(int)
        return cls(
            frame @ world @ tool,
            frame @ pre @ tool,
            frame @ lift @ tool,
            widths,
            ids,
            frame_id,
            tool_link,
            scores,
        )

    def poses(self, stage: Stage = "grasp") -> np.ndarray:
        """Return a copy of a stage's (N,4,4) tool transforms."""
        if stage not in ("grasp", "pregrasp", "lift"):
            raise ValueError("stage must be 'grasp', 'pregrasp', or 'lift'")
        return getattr(self, stage).copy()

    def components(self, stage: Stage = "grasp", *, quaternion_order: QuaternionOrder = "xyzw"):
        """Return position/quaternion arrays in an explicit component order."""
        return pose_components(self.poses(stage), quaternion_order=quaternion_order)

    def select(self, selection) -> PlanningGrasps:
        """Subset/reorder by integer indices or a boolean IK/planner success mask.

        IDs, widths and scores stay aligned. Repeated indices are rejected.
        Convert CUDA success tensors via ``success.detach().cpu().numpy()``.
        """
        selection = np.asarray(selection)
        if selection.shape == (0,) and selection.dtype.kind != "b":
            selection = selection.astype(int)
        if selection.ndim != 1 or selection.dtype.kind not in "biu":
            raise ValueError("selection must be a one-dimensional boolean mask or integer indices")
        if selection.dtype.kind == "b" and selection.shape != (len(self),):
            raise ValueError("boolean selection must have shape (N,)")
        indices = np.arange(len(self))[selection]
        return type(self)(
            self.grasp[indices],
            self.pregrasp[indices],
            self.lift[indices],
            self.widths[indices],
            self.candidate_ids[indices],
            self.frame_id,
            self.tool_link,
            None if self.scores is None else self.scores[indices],
        )

    def to_dict(self) -> dict:
        """JSON-compatible, versioned matrices in metres with explicit frame labels."""
        return {
            "schema_version": 1,
            "units": "metres",
            "frame_id": self.frame_id,
            "tool_link": self.tool_link,
            "candidate_ids": self.candidate_ids.tolist(),
            "widths": self.widths.tolist(),
            "scores": None if self.scores is None else self.scores.tolist(),
            **{stage: self.poses(stage).tolist() for stage in ("grasp", "pregrasp", "lift")},
        }

    @classmethod
    def from_dict(cls, value: dict) -> PlanningGrasps:
        """Read the schema emitted by ``to_dict`` (without pickle)."""
        if value.get("schema_version") != 1 or value.get("units") != "metres":
            raise ValueError("unsupported planning targets schema or units")
        n = len(value["widths"])
        return cls(
            *(
                np.asarray(value[s], float).reshape(0, 4, 4) if n == 0 else np.asarray(value[s], float)
                for s in ("grasp", "pregrasp", "lift")
            ),
            value["widths"],
            (np.asarray(value["candidate_ids"], dtype=int) if n == 0 else np.asarray(value["candidate_ids"])),
            value["frame_id"],
            value["tool_link"],
            value.get("scores"),
        )
