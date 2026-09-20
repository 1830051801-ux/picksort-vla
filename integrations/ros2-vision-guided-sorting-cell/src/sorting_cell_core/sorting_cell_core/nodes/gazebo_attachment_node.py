from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from sorting_cell_core.kinematics import ArmKinematics


class GazeboAttachmentNode(Node):
    """Keeps the active workpiece at the tool pose through Gazebo's SetEntityPose service."""

    def __init__(self) -> None:
        super().__init__("gazebo_attachment")
        self.declare_parameter("world_name", "sorting_cell")
        self.declare_parameter("service_timeout_s", 5.0)
        self.declare_parameter("request_timeout_s", 1.0)
        world_name = str(self.get_parameter("world_name").value)
        self.service_timeout_s = float(self.get_parameter("service_timeout_s").value)
        self.request_timeout_s = float(self.get_parameter("request_timeout_s").value)
        self.kinematics = ArmKinematics()
        self.joints: dict[str, float] = {}
        self.attached_object = ""
        self.pending_request = None
        self.pending_object_id = ""
        self.pending_request_started = 0.0
        self.attach_started = 0.0
        self.attach_confirmed = False
        self.client = self.create_client(SetEntityPose, f"/world/{world_name}/set_pose")
        self.status_publisher = self.create_publisher(
            String, "/sorting_cell/attachment_status", 10
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 20)
        self.create_subscription(String, "/sorting_cell/attachment_command", self._on_command, 10)
        self.create_timer(0.05, self._tick)

    def _on_joint_state(self, message: JointState) -> None:
        self.joints.update(zip(message.name, message.position))

    def _on_command(self, message: String) -> None:
        action, separator, object_id = message.data.partition(":")
        if not separator or not object_id:
            self.get_logger().warning(f"ignored malformed attachment command: {message.data}")
            return
        if action == "attach":
            if self.attached_object and self.attached_object != object_id:
                self._publish_status("failed", object_id, "another object is already attached")
                return
            self.attached_object = object_id
            self.attach_started = time.monotonic()
            self.attach_confirmed = False
            self.get_logger().info(f"attachment requested for {object_id}")
        elif action == "detach":
            if self.attached_object and self.attached_object != object_id:
                self._publish_status("failed", object_id, "detach object does not match")
                return
            self.get_logger().info(f"detached {object_id}")
            self.attached_object = ""
            self.attach_confirmed = False
            self._publish_status("detached", object_id, "released")
        else:
            self._publish_status("failed", object_id, f"unknown action {action}")

    def _publish_status(self, state: str, object_id: str, detail: str) -> None:
        message = String()
        message.data = f"{state}:{object_id}:{detail}"
        self.status_publisher.publish(message)

    def _tick(self) -> None:
        if not self.attached_object:
            return
        now = time.monotonic()
        if self.pending_request is not None:
            if now - self.pending_request_started > self.request_timeout_s:
                object_id = self.attached_object
                self.pending_request.cancel()
                self.pending_request = None
                self.pending_object_id = ""
                self.attached_object = ""
                self.attach_confirmed = False
                self._publish_status("failed", object_id, "set_pose request timed out")
            return
        if not all(name in self.joints for name in self.kinematics.joint_names):
            if now - self.attach_started > self.service_timeout_s:
                object_id = self.attached_object
                self.attached_object = ""
                self._publish_status("failed", object_id, "joint state unavailable")
            return
        if not self.client.service_is_ready():
            if now - self.attach_started > self.service_timeout_s:
                object_id = self.attached_object
                self.attached_object = ""
                self._publish_status("failed", object_id, "set_pose service unavailable")
            return
        joint_values = tuple(self.joints[name] for name in self.kinematics.joint_names)
        tool, _ = self.kinematics.forward(joint_values)
        request = SetEntityPose.Request()
        request.entity.name = self.attached_object
        request.entity.type = Entity.MODEL
        request.pose.position.x = tool.x
        request.pose.position.y = tool.y
        request.pose.position.z = max(0.065, tool.z - 0.040)
        request.pose.orientation.w = 1.0
        self.pending_request = self.client.call_async(request)
        self.pending_object_id = self.attached_object
        self.pending_request_started = now
        self.pending_request.add_done_callback(self._on_pose_result)

    def _on_pose_result(self, future) -> None:
        if future.cancelled():
            self.pending_request = None
            self.pending_object_id = ""
            return
        request_object_id = self.pending_object_id
        try:
            response = future.result()
            if not request_object_id or request_object_id != self.attached_object:
                return
            if response is None or not response.success:
                object_id = request_object_id
                self.attached_object = ""
                self.attach_confirmed = False
                self._publish_status("failed", object_id, "Gazebo rejected pose update")
                self.get_logger().warning("Gazebo rejected a workpiece pose update")
            elif not self.attach_confirmed:
                self.attach_confirmed = True
                self._publish_status("attached", request_object_id, "pose updates active")
                self.get_logger().info(f"attached {request_object_id}")
        except Exception as error:  # ROS futures surface transport errors here.
            object_id = request_object_id or self.attached_object
            self.attached_object = ""
            self.attach_confirmed = False
            self._publish_status("failed", object_id, str(error))
            self.get_logger().warning(f"workpiece pose update failed: {error}")
        finally:
            self.pending_request = None
            self.pending_object_id = ""


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboAttachmentNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
