# API reference

## Sampler

```{eval-rst}
.. autoclass:: grasp_sampler.GraspSampler
   :members:
```

## Data types

```{eval-rst}
.. autoclass:: grasp_sampler.types.Grasp
   :members:

.. autoclass:: grasp_sampler.types.ObjMesh
   :members:

.. autoclass:: grasp_sampler.types.GraspConfig
   :members:
```

## Mesh loading

```{eval-rst}
.. autofunction:: grasp_sampler.load_mesh
```

## Generators

```{eval-rst}
.. autofunction:: grasp_sampler.top_grasps
.. autofunction:: grasp_sampler.side_grasps
.. autofunction:: grasp_sampler.obb_face_grasps
.. autofunction:: grasp_sampler.antipodal_grasps
```

## Transforms

```{eval-rst}
.. autofunction:: grasp_sampler.stack_poses
.. autofunction:: grasp_sampler.to_world
.. autofunction:: grasp_sampler.tcp_to_ee
```

## Robust evaluation under object uncertainty

See {doc}`uncertainty` for the workflow.

```{eval-rst}
.. autoclass:: grasp_sampler.PoseUncertainty
.. autoclass:: grasp_sampler.GeometryUncertainty
.. autoclass:: grasp_sampler.ObjectHypothesis
.. autofunction:: grasp_sampler.sample_hypotheses
.. autoclass:: grasp_sampler.RobustGraspEvaluator
   :members:
.. autofunction:: grasp_sampler.rank_grasps
.. autoclass:: grasp_sampler.RobustSetConfig
.. autofunction:: grasp_sampler.generate_robust_grasp_set
.. autoclass:: grasp_sampler.RobustGraspSetResult
```

Validity backends: `GeometricGraspBackend`, `SymmetricPointBackend`,
`FinitePadFixedObjectBackend`, `FinitePadQuasistaticBackend`.

## PyBullet integration

```{eval-rst}
.. autofunction:: grasp_sampler.pybullet_sim.collision_filter
.. autofunction:: grasp_sampler.pybullet_sim.show_grasps
```
