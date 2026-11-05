"""ROS 2 PoseStamped targets for MoveIt 2's moveit_py planning component."""
from __future__ import annotations

from ..planning import PlanningGrasps, Stage


def to_pose_stamped(targets: PlanningGrasps, *, stage: Stage = "grasp", stamp=None):
    """Create geometry_msgs/PoseStamped messages in ``targets.frame_id``.

    Source your ROS 2 workspace first. ``stamp`` may be a builtin_interfaces/Time
    (e.g. node.get_clock().now().to_msg()); otherwise ROS's zero stamp is used.
    No TF conversion is performed by this adapter.
    """
    try:
        from geometry_msgs.msg import PoseStamped
    except ImportError as exc:
        raise ImportError("MoveIt export requires ROS 2 geometry_msgs; source your ROS workspace") from exc
    p, q = targets.components(stage, quaternion_order="xyzw")
    messages = []
    for position, quaternion in zip(p, q):
        msg = PoseStamped()
        msg.header.frame_id = targets.frame_id
        if stamp is not None:
            msg.header.stamp = stamp
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = map(float, quaternion)
        messages.append(msg)
    return messages
