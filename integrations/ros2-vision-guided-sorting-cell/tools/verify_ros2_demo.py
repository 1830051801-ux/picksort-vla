from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, JointState

from sorting_cell_core.gz_pose import parse_gz_pose_vector
from sorting_cell_interfaces.msg import CellStatus, DetectedObject


EXPECTED_DESTINATIONS = {
    "part_blue": ("accepted_bin", 0.30, 0.30),
    "part_red": ("reject_bin", 0.30, -0.30),
    "part_orange": ("rework_bin", 0.48, 0.25),
}
EXPECTED_CLASSES = {
    "part_blue": "accepted",
    "part_red": "scratch",
    "part_orange": "unknown",
}
GAZEBO_POSE_TOPIC = "/world/sorting_cell/dynamic_pose/info"


def query_gazebo_positions(
    names: set[str], timeout_s: float = 10.0
) -> tuple[dict[str, tuple[float, float, float]], str]:
    """Read authoritative named model poses directly from Gazebo Transport."""

    try:
        completed = subprocess.run(
            ["gz", "topic", "-e", "-n", "1", "-t", GAZEBO_POSE_TOPIC],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"Gazebo pose query failed: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        return {}, f"Gazebo pose query failed: {detail}"
    positions = parse_gz_pose_vector(completed.stdout, names)
    missing = sorted(names - set(positions))
    if missing:
        return positions, f"Gazebo pose query omitted: {', '.join(missing)}"
    return positions, ""


class DemoVerifier(Node):
    def __init__(self) -> None:
        super().__init__("sorting_cell_demo_verifier")
        self.terminal_messages: dict[str, CellStatus] = {}
        self.terminal_counts: dict[str, int] = {}
        self.failures: list[str] = []
        self.camera_frames = 0
        self.joint_state_messages = 0
        self.detections: dict[str, dict[str, object]] = {}
        self.entity_positions: dict[str, tuple[float, float, float]] = {}
        status_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            CellStatus, "/sorting_cell/status", self._on_status, status_qos
        )
        self.create_subscription(
            Image,
            "/sorting_cell/camera/image_raw",
            self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 20)
        self.create_subscription(
            DetectedObject, "/sorting_cell/detections", self._on_detection, 20
        )
        self.controller_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )

    def _on_status(self, message: CellStatus) -> None:
        if message.state not in ("COMPLETE", "FAILED"):
            return
        object_id = message.active_object_id
        self.terminal_counts[object_id] = self.terminal_counts.get(object_id, 0) + 1
        self.terminal_messages[object_id] = message
        if message.state == "FAILED":
            self.failures.append(f"{object_id}: {message.message}")

    def _on_image(self, _message: Image) -> None:
        self.camera_frames += 1

    def _on_joint_state(self, _message: JointState) -> None:
        self.joint_state_messages += 1

    def _on_detection(self, message: DetectedObject) -> None:
        if message.object_id not in EXPECTED_DESTINATIONS:
            return
        self.detections[message.object_id] = {
            "class_name": message.class_name,
            "frame_id": message.header.frame_id,
            "confidence": round(float(message.confidence), 4),
            "position": [
                round(float(message.pose.position.x), 5),
                round(float(message.pose.position.y), 5),
                round(float(message.pose.position.z), 5),
            ],
        }

    def controllers_active(self, timeout_s: float = 30.0) -> tuple[bool, dict[str, str]]:
        required = {"cell_controller", "joint_state_broadcaster"}
        deadline = time.monotonic() + timeout_s
        states: dict[str, str] = {}
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if not self.controller_client.wait_for_service(
                timeout_sec=min(1.0, remaining)
            ):
                continue
            future = self.controller_client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(
                self, future, timeout_sec=min(2.0, remaining)
            )
            if not future.done():
                continue
            response = future.result()
            if response is not None:
                states = {
                    controller.name: controller.state
                    for controller in response.controller
                }
                if all(states.get(name) == "active" for name in required):
                    return True, states
            time.sleep(0.2)
        return False, states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a running ROS 2 sorting-cell demo from live feedback."
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = DemoVerifier()
    started = time.monotonic()
    deadline = started + args.timeout
    try:
        controllers_ok, controller_states = node.controllers_active()
        while time.monotonic() < deadline and not node.failures:
            rclpy.spin_once(node, timeout_sec=0.2)
            if set(node.terminal_messages) == set(EXPECTED_DESTINATIONS):
                settle_deadline = time.monotonic() + 2.0
                while time.monotonic() < settle_deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                break

        node.entity_positions, pose_query_error = query_gazebo_positions(
            set(EXPECTED_DESTINATIONS)
        )
        active_nodes = set(node.get_node_names())
        checks: dict[str, bool] = {
            "controllers_active": controllers_ok,
            "color_perception_node_active": "color_perception" in active_nodes,
            "camera_stream_received": node.camera_frames > 0,
            "joint_states_received": node.joint_state_messages >= 5,
            "three_vision_detections_received": set(node.detections)
            == set(EXPECTED_DESTINATIONS),
            "detections_are_base_link_and_expected_classes": all(
                node.detections.get(object_id, {}).get("frame_id") == "base_link"
                and node.detections.get(object_id, {}).get("class_name") == class_name
                for object_id, class_name in EXPECTED_CLASSES.items()
            ),
            "three_unique_terminal_cycles": set(node.terminal_messages)
            == set(EXPECTED_DESTINATIONS),
            "no_failed_cycles": not node.failures,
            "single_terminal_event_per_object": all(
                count == 1 for count in node.terminal_counts.values()
            ),
            "gazebo_named_pose_query_succeeded": not pose_query_error,
        }
        for object_id, (destination, expected_x, expected_y) in EXPECTED_DESTINATIONS.items():
            message = node.terminal_messages.get(object_id)
            checks[f"{object_id}_reported_{destination}"] = bool(
                message
                and message.state == "COMPLETE"
                and destination in message.message
            )
            position = node.entity_positions.get(object_id)
            checks[f"{object_id}_physically_in_{destination}"] = bool(
                position
                and abs(position[0] - expected_x) <= 0.055
                and abs(position[1] - expected_y) <= 0.050
                and 0.055 <= position[2] <= 0.130
            )

        payload = {
            "passed": all(checks.values()),
            "elapsed_wall_s": round(time.monotonic() - started, 3),
            "checks": checks,
            "controller_states": controller_states,
            "camera_frames": node.camera_frames,
            "joint_state_messages": node.joint_state_messages,
            "terminal_counts": node.terminal_counts,
            "detections": node.detections,
            "entity_positions": node.entity_positions,
            "failures": node.failures,
            "pose_query_error": pose_query_error,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0 if payload["passed"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
