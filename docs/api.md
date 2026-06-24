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

## PyBullet integration

```{eval-rst}
.. autofunction:: grasp_sampler.pybullet_sim.collision_filter
.. autofunction:: grasp_sampler.pybullet_sim.show_grasps
```
