from __future__ import annotations

from dataclasses import dataclass

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from sorting_cell_interfaces.msg import CellStatus, DetectedObject


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    class_name: str
    x: float
    y: float
    z: float
    confidence: float


class SyntheticPerceptionNode(Node):
    """Publishes deterministic Gazebo ground-truth detections one cycle at a time."""

    def __init__(self) -> None:
        super().__init__("synthetic_perception")
        self.declare_parameter("initial_delay_s", 2.5)
        self.declare_parameter("inter_cycle_delay_s", 1.2)
        self.publisher = self.create_publisher(DetectedObject, "/sorting_cell/detections", 10)
        status_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            CellStatus, "/sorting_cell/status", self._on_status, status_qos
        )
        self.objects = [
            SceneObject("part_blue", "accepted", 0.47, -0.13, 0.075, 0.98),
            SceneObject("part_red", "scratch", 0.50, 0.02, 0.075, 0.96),
            SceneObject("part_orange", "unknown", 0.43, 0.15, 0.075, 0.91),
        ]
        self.index = 0
        self.waiting_for_cycle = False
        self.next_publish_time = self.get_clock().now().nanoseconds / 1e9 + float(
            self.get_parameter("initial_delay_s").value
        )
        self.create_timer(0.1, self._tick)

    def _on_status(self, message: CellStatus) -> None:
        if message.state in ("COMPLETE", "FAILED") and self.waiting_for_cycle:
            self.waiting_for_cycle = False
            self.next_publish_time = self.get_clock().now().nanoseconds / 1e9 + float(
                self.get_parameter("inter_cycle_delay_s").value
            )

    def _tick(self) -> None:
        if self.waiting_for_cycle or self.index >= len(self.objects):
            return
        if self.publisher.get_subscription_count() == 0:
            return
        now_s = self.get_clock().now().nanoseconds / 1e9
        if now_s < self.next_publish_time:
            return
        scene_object = self.objects[self.index]
        message = DetectedObject()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.object_id = scene_object.object_id
        message.class_name = scene_object.class_name
        message.pose.position.x = scene_object.x
        message.pose.position.y = scene_object.y
        message.pose.position.z = scene_object.z
        message.pose.orientation.w = 1.0
        message.confidence = scene_object.confidence
        self.publisher.publish(message)
        self.get_logger().info(
            f"detected {scene_object.object_id}: class={scene_object.class_name}, "
            f"position=({scene_object.x:.3f}, {scene_object.y:.3f}, {scene_object.z:.3f})"
        )
        self.index += 1
        self.waiting_for_cycle = True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SyntheticPerceptionNode()
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
