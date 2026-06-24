"""grasp_sampler -- a simple, mesh-based parallel-jaw grasp sampler.

Quick start
-----------
>>> from grasp_sampler import GraspSampler, stack_poses, to_world, tcp_to_ee
>>> sampler = GraspSampler()
>>> grasps = sampler.sample("object.glb")        # object-local TCP grasps
>>> poses = to_world(stack_poses(grasps), T_world_obj)   # place in the world
>>> ee = tcp_to_ee(poses)                         # Franka flange poses for IK
"""
from .antipodal import antipodal_grasps
from .mesh import load_mesh
from .obb import obb_face_grasps
from .primitives import side_grasps, top_grasps
from .sampler import GraspSampler
from .transforms import stack_poses, tcp_to_ee, to_world
from .types import Grasp, GraspConfig, ObjMesh

__version__ = "0.1.0"

__all__ = [
    "GraspSampler",
    "GraspConfig",
    "Grasp",
    "ObjMesh",
    "load_mesh",
    "top_grasps",
    "side_grasps",
    "obb_face_grasps",
    "antipodal_grasps",
    "stack_poses",
    "to_world",
    "tcp_to_ee",
]
