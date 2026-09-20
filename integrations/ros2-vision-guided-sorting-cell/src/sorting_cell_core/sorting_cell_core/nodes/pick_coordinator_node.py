from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from sorting_cell_interfaces.msg import CellStatus, DetectedObject

from sorting_cell_core.kinematics import ArmKinematics
from sorting_cell_core.models import CellFrame, CyclePlan, CycleState, Detection, Vector3
from sorting_cell_core.workflow import WorkflowPlanner


@dataclass(frozen=True)
class _TrajectorySegment:
    """One contiguous workflow state sent as one controller action goal."""

    state: CycleState
    frames: tuple[CellFrame, ...]
    trajectory: JointTrajectory
    target: tuple[float, float, float, float, float, float]
    duration_s: float


class PickCoordinatorNode(Node):
    """Executes each planned workflow segment through ros2_control's action API."""

    _COMMAND_STATES = {
        CycleState.PREGRASP,
        CycleState.APPROACH,
        CycleState.GRASP,
        CycleState.LIFT,
        CycleState.TRANSFER,
        CycleState.PLACE,
        CycleState.RELEASE,
        CycleState.RETREAT,
        CycleState.HOME,
    }

    def __init__(self) -> None:
        super().__init__("pick_coordinator")
        self.declare_parameter(
            "controller_action", "/cell_controller/follow_joint_trajectory"
        )
        # Kept for launch/config compatibility; it now controls planner sampling
        # density, not a command publishing timer.
        self.declare_parameter("command_period_s", 0.04)
        self.declare_parameter("controller_wait_timeout_s", 8.0)
        self.declare_parameter("goal_response_timeout_s", 5.0)
        self.declare_parameter("segment_timeout_margin_s", 5.0)
        self.declare_parameter("cancellation_timeout_s", 5.0)
        self.declare_parameter("joint_state_wait_timeout_s", 8.0)
        self.declare_parameter("joint_state_max_age_s", 1.0)
        self.declare_parameter("verification_timeout_s", 1.0)
        self.declare_parameter("arm_goal_tolerance_rad", 0.12)
        self.declare_parameter("gripper_goal_tolerance_m", 0.01)
        self.declare_parameter("grasp_contact_tolerance_m", 0.031)
        self.declare_parameter("attachment_timeout_s", 5.0)
        self.declare_parameter("cell_config", "")

        self.sample_period_s = max(
            0.001, float(self.get_parameter("command_period_s").value)
        )
        self.controller_wait_timeout_s = max(
            0.1, float(self.get_parameter("controller_wait_timeout_s").value)
        )
        self.goal_response_timeout_s = max(
            0.1, float(self.get_parameter("goal_response_timeout_s").value)
        )
        self.segment_timeout_margin_s = max(
            0.1, float(self.get_parameter("segment_timeout_margin_s").value)
        )
        self.cancellation_timeout_s = max(
            0.1, float(self.get_parameter("cancellation_timeout_s").value)
        )
        self.joint_state_wait_timeout_s = max(
            0.1, float(self.get_parameter("joint_state_wait_timeout_s").value)
        )
        self.joint_state_max_age_s = max(
            0.05, float(self.get_parameter("joint_state_max_age_s").value)
        )
        self.verification_timeout_s = max(
            0.05, float(self.get_parameter("verification_timeout_s").value)
        )
        self.arm_goal_tolerance_rad = max(
            0.0, float(self.get_parameter("arm_goal_tolerance_rad").value)
        )
        self.gripper_goal_tolerance_m = max(
            0.0, float(self.get_parameter("gripper_goal_tolerance_m").value)
        )
        self.grasp_contact_tolerance_m = max(
            self.gripper_goal_tolerance_m,
            float(self.get_parameter("grasp_contact_tolerance_m").value),
        )
        self.attachment_timeout_s = max(
            0.1, float(self.get_parameter("attachment_timeout_s").value)
        )

        self.kinematics = ArmKinematics()
        configured_path = str(self.get_parameter("cell_config").value).strip()
        config_path = Path(configured_path) if configured_path else (
            Path(get_package_share_directory("sorting_cell_core")) / "config" / "cell.yaml"
        )
        self.planner = WorkflowPlanner.from_yaml(
            config_path,
            kinematics=self.kinematics,
            sample_period=self.sample_period_s,
        )
        self.joint_names = tuple(self.kinematics.joint_names) + ("left_finger_joint",)

        # Detection ownership remains local to the coordinator: an object ID is
        # queued at most once, even if perception republishes it.
        self.queue: deque[Detection] = deque()
        self.seen_ids: set[str] = set()
        self.object_positions: dict[str, Vector3] = {}

        self.measured_positions: dict[str, float] = {}
        self.measured_at_monotonic: dict[str, float] = {}

        self.active_detection: Detection | None = None
        self.active_plan: CyclePlan | None = None
        self.segments: list[_TrajectorySegment] = []
        self.segment_index = 0
        self.phase = "idle"
        self.phase_deadline_monotonic = 0.0
        self.execution_deadline_ros_ns = 0
        self.execution_watchdog_deadline_monotonic = 0.0
        self.cycle_started_monotonic = 0.0
        self.segment_sent_monotonic = 0.0
        self.active_goal_handle = None
        self.pending_failure_reason = ""
        self.cycle_token = 0
        self.terminal_published = False
        self.attachment_active = False
        self.pending_attachment_action = ""

        self.completed = 0
        self.failed = 0

        controller_action = str(self.get_parameter("controller_action").value)
        self.action_client = ActionClient(
            self, FollowJointTrajectory, controller_action
        )
        status_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_publisher = self.create_publisher(
            CellStatus, "/sorting_cell/status", status_qos
        )
        self.attachment_publisher = self.create_publisher(
            String, "/sorting_cell/attachment_command", 10
        )
        self.create_subscription(
            DetectedObject, "/sorting_cell/detections", self._on_detection, 10
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 20)
        self.create_subscription(
            String,
            "/sorting_cell/attachment_status",
            self._on_attachment_status,
            10,
        )
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"coordinator ready; action server={controller_action}, "
            f"trajectory sample period={self.sample_period_s:.3f}s"
        )

    def _on_detection(self, message: DetectedObject) -> None:
        if message.object_id in self.seen_ids:
            return

        self.seen_ids.add(message.object_id)
        if message.header.frame_id != "base_link":
            received_frame = message.header.frame_id or "<empty>"
            reason = (
                "detection frame must be 'base_link'; "
                f"received '{received_frame}'"
            )
            self.failed += 1
            self._publish_terminal(
                object_id=message.object_id,
                success=False,
                elapsed_s=0.0,
                reason=reason,
            )
            self.get_logger().error(f"rejected {message.object_id}: {reason}")
            return

        detection = Detection(
            object_id=message.object_id,
            class_name=message.class_name,
            position=Vector3(
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ),
            confidence=float(message.confidence),
        )
        self.object_positions[message.object_id] = detection.position
        self.queue.append(detection)
        self.get_logger().info(
            f"queued {message.object_id} ({message.class_name}) in base_link"
        )

    def _on_joint_state(self, message: JointState) -> None:
        received_at = time.monotonic()
        for name, position in zip(message.name, message.position):
            value = float(position)
            if name in self.joint_names and math.isfinite(value):
                self.measured_positions[name] = value
                self.measured_at_monotonic[name] = received_at

    def _tick(self) -> None:
        now = time.monotonic()
        if self.active_detection is None:
            if self.queue:
                self._begin_cycle(self.queue.popleft(), now)
            return

        if self.phase == "waiting_for_runtime":
            self._try_prepare_cycle(now)
        elif self.phase == "sending_goal" and now >= self.phase_deadline_monotonic:
            self._halt_active(
                "controller did not accept or reject the goal before timeout; "
                "goal ownership is unknown"
            )
        elif self.phase == "executing_goal":
            ros_now_ns = self.get_clock().now().nanoseconds
            simulation_timeout = (
                self.execution_deadline_ros_ns > 0
                and ros_now_ns >= self.execution_deadline_ros_ns
            )
            wall_watchdog_timeout = (
                self.execution_watchdog_deadline_monotonic > 0.0
                and now >= self.execution_watchdog_deadline_monotonic
            )
            if simulation_timeout or wall_watchdog_timeout:
                reason = (
                    "trajectory action exceeded its simulation-time deadline"
                    if simulation_timeout
                    else "trajectory action wall-clock watchdog expired"
                )
                self._cancel_active_goal(reason, now)
        elif self.phase == "cancelling_goal" and now >= self.phase_deadline_monotonic:
            reason = self.pending_failure_reason or "trajectory cancellation requested"
            self._halt_active(
                f"{reason}; controller did not report a terminal goal state "
                "before the cancellation timeout"
            )
        elif self.phase == "verifying_goal":
            self._verify_segment(now)
        elif self.phase == "waiting_for_attachment" and now >= self.phase_deadline_monotonic:
            self._fail_active(
                f"attachment {self.pending_attachment_action or 'operation'} timed out"
            )

    def _begin_cycle(self, detection: Detection, now: float) -> None:
        self.cycle_token += 1
        self.active_detection = detection
        self.active_plan = None
        self.segments = []
        self.segment_index = 0
        self.phase = "waiting_for_runtime"
        self.phase_deadline_monotonic = now + max(
            self.controller_wait_timeout_s, self.joint_state_wait_timeout_s
        )
        self.execution_deadline_ros_ns = 0
        self.execution_watchdog_deadline_monotonic = 0.0
        self.cycle_started_monotonic = now
        self.segment_sent_monotonic = 0.0
        self.active_goal_handle = None
        self.pending_failure_reason = ""
        self.terminal_published = False
        self.attachment_active = False
        self.pending_attachment_action = ""
        self._publish_segment_status(
            CycleState.VALIDATING,
            "waiting for controller and a fresh measured joint state",
        )
        self._try_prepare_cycle(now)

    def _try_prepare_cycle(self, now: float) -> None:
        measured = self._fresh_measured_state(now)
        controller_ready = self.action_client.server_is_ready()
        if measured is not None and controller_ready:
            self._plan_and_execute(measured)
            return

        if now < self.phase_deadline_monotonic:
            return

        missing: list[str] = []
        if not controller_ready:
            missing.append("FollowJointTrajectory action server unavailable")
        if measured is None:
            missing.append("fresh /joint_states unavailable or incomplete")
        self._fail_active("; ".join(missing) or "runtime prerequisites unavailable")

    def _fresh_measured_state(
        self, now: float
    ) -> tuple[float, float, float, float, float, float] | None:
        if not all(name in self.measured_positions for name in self.joint_names):
            return None
        if not all(
            name in self.measured_at_monotonic
            and now - self.measured_at_monotonic[name] <= self.joint_state_max_age_s
            for name in self.joint_names
        ):
            return None
        return tuple(self.measured_positions[name] for name in self.joint_names)  # type: ignore[return-value]

    def _plan_and_execute(
        self, measured: tuple[float, float, float, float, float, float]
    ) -> None:
        detection = self.active_detection
        if detection is None:
            return

        self.active_plan = self.planner.plan_cycle(
            detection,
            start_joints=measured[:5],
            start_gripper=measured[5],
            object_positions=self.object_positions,
        )
        report = self.active_plan.report
        if report is None or not report.success:
            reason = report.reason if report is not None else "planner returned no cycle report"
            self._fail_active(reason or "trajectory planning failed")
            return

        self.segments = self._build_segments(self.active_plan.frames)
        if not self.segments:
            self._fail_active("planner returned no executable trajectory segments")
            return

        self.get_logger().info(
            f"planned {detection.object_id}: destination={report.destination}, "
            f"segments={len(self.segments)}, points={sum(len(s.frames) for s in self.segments)}"
        )
        self.segment_index = 0
        self._send_current_segment()

    def _build_segments(self, frames: list[CellFrame]) -> list[_TrajectorySegment]:
        grouped: list[list[CellFrame]] = []
        for frame in frames:
            if frame.state not in self._COMMAND_STATES:
                continue
            if not grouped or grouped[-1][-1].state != frame.state:
                grouped.append([frame])
            else:
                grouped[-1].append(frame)

        segments: list[_TrajectorySegment] = []
        sample_ns = max(1, round(self.sample_period_s * 1_000_000_000))
        for group in grouped:
            trajectory = JointTrajectory()
            trajectory.joint_names = list(self.joint_names)
            first_time_s = group[0].time_s
            previous_ns = 0
            for frame in group:
                relative_s = frame.time_s - first_time_s + self.sample_period_s
                point_ns = max(previous_ns + 1, round(relative_s * 1_000_000_000))
                # Preserve the planner cadence even if floating-point frame times
                # happen to quantize onto the same integer nanosecond.
                if previous_ns and point_ns - previous_ns < sample_ns // 4:
                    point_ns = previous_ns + sample_ns
                point = JointTrajectoryPoint()
                point.positions = list(frame.joints) + [frame.gripper_m]
                point.time_from_start.sec = point_ns // 1_000_000_000
                point.time_from_start.nanosec = point_ns % 1_000_000_000
                trajectory.points.append(point)
                previous_ns = point_ns

            last = group[-1]
            target = tuple(last.joints) + (last.gripper_m,)
            segments.append(
                _TrajectorySegment(
                    state=group[0].state,
                    frames=tuple(group),
                    trajectory=trajectory,
                    target=target,  # type: ignore[arg-type]
                    duration_s=previous_ns / 1_000_000_000.0,
                )
            )
        return segments

    def _send_current_segment(self) -> None:
        if self.active_detection is None:
            return
        if self.segment_index >= len(self.segments):
            self._finish_active_success()
            return

        segment = self.segments[self.segment_index]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = segment.trajectory
        if segment.state == CycleState.GRASP:
            # A position-controlled gripper normally stops on the workpiece
            # instead of reaching its empty-gripper closed position.  Override
            # only this segment's finger tolerance; RELEASE stays strict.
            contact_tolerance = JointTolerance()
            contact_tolerance.name = "left_finger_joint"
            contact_tolerance.position = self.grasp_contact_tolerance_m
            goal.goal_tolerance = [contact_tolerance]
        now = time.monotonic()
        self.segment_sent_monotonic = now
        self.phase = "sending_goal"
        self.phase_deadline_monotonic = now + self.goal_response_timeout_s
        self.execution_deadline_ros_ns = 0
        self.execution_watchdog_deadline_monotonic = 0.0
        self.active_goal_handle = None
        self._publish_segment_status(
            segment.state,
            f"executing segment {self.segment_index + 1}/{len(self.segments)} "
            f"({len(segment.trajectory.points)} points)",
        )

        token = self.cycle_token
        try:
            future = self.action_client.send_goal_async(goal)
        except Exception as error:
            self._fail_active(f"could not send trajectory goal: {error}")
            return
        future.add_done_callback(
            lambda completed, cycle_token=token: self._on_goal_response(
                completed, cycle_token
            )
        )

    def _on_goal_response(self, future, token: int) -> None:
        if not self._callback_is_current(token, "sending_goal"):
            return
        try:
            goal_handle = future.result()
        except Exception as error:
            self._halt_active(
                f"trajectory goal response failed and goal ownership is unknown: {error}"
            )
            return
        if goal_handle is None or not goal_handle.accepted:
            self._fail_active("trajectory controller rejected the goal")
            return

        self.active_goal_handle = goal_handle
        segment = self.segments[self.segment_index]
        self.phase = "executing_goal"
        simulation_timeout_s = segment.duration_s + self.segment_timeout_margin_s
        self.execution_deadline_ros_ns = self.get_clock().now().nanoseconds + round(
            simulation_timeout_s * 1_000_000_000
        )
        # ROS actions execute against /clock.  WSL rendering can reduce real-time
        # factor substantially, so wall time is only a long-stop for a frozen
        # simulator or broken action server.
        self.execution_watchdog_deadline_monotonic = time.monotonic() + max(
            60.0, simulation_timeout_s * 10.0
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, cycle_token=token: self._on_goal_result(
                completed, cycle_token
            )
        )

    def _on_goal_result(self, future, token: int) -> None:
        if self.active_detection is None or token != self.cycle_token:
            return
        if self.phase not in {"executing_goal", "cancelling_goal"}:
            return
        if self.terminal_published:
            return
        cancelling = self.phase == "cancelling_goal"
        try:
            wrapped_result = future.result()
        except Exception as error:
            if cancelling:
                reason = self.pending_failure_reason or "trajectory cancellation requested"
                self._halt_active(f"{reason}; could not receive terminal action result: {error}")
            else:
                self._halt_active(
                    f"trajectory action result unavailable; terminal state is unknown: {error}"
                )
            return

        if cancelling:
            reason = self.pending_failure_reason or "trajectory cancellation requested"
            self.active_goal_handle = None
            self.execution_deadline_ros_ns = 0
            self.execution_watchdog_deadline_monotonic = 0.0
            self.phase_deadline_monotonic = 0.0
            status_detail = (
                "controller confirmed cancellation"
                if wrapped_result.status == GoalStatus.STATUS_CANCELED
                else f"controller reached terminal action status {wrapped_result.status}"
            )
            self._fail_active(f"{reason}; {status_detail}")
            return

        result = wrapped_result.result

        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            detail = result.error_string or "no controller detail"
            self._fail_active(
                f"trajectory action ended with status {wrapped_result.status}; "
                f"controller error {result.error_code}: {detail}"
            )
            return
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            detail = result.error_string or "no controller detail"
            self._fail_active(
                f"trajectory controller error {result.error_code}: {detail}"
            )
            return

        self.active_goal_handle = None
        self.execution_deadline_ros_ns = 0
        self.execution_watchdog_deadline_monotonic = 0.0
        self.phase = "verifying_goal"
        self.phase_deadline_monotonic = time.monotonic() + self.verification_timeout_s
        self._verify_segment(time.monotonic())

    def _cancel_active_goal(self, reason: str, now: float) -> None:
        if self.active_goal_handle is None:
            self._halt_active(f"{reason}; no active goal handle was available to cancel")
            return

        self.phase = "cancelling_goal"
        self.pending_failure_reason = reason
        self.phase_deadline_monotonic = now + self.cancellation_timeout_s
        self.execution_deadline_ros_ns = 0
        self.execution_watchdog_deadline_monotonic = 0.0
        token = self.cycle_token
        try:
            cancel_future = self.active_goal_handle.cancel_goal_async()
        except Exception as error:
            self._halt_active(f"{reason}; goal cancellation request failed: {error}")
            return
        cancel_future.add_done_callback(
            lambda completed, cycle_token=token: self._on_cancel_response(
                completed, cycle_token
            )
        )
        self.get_logger().warning(
            f"{reason}; cancellation requested and new cycles are blocked until "
            "the controller reports a terminal goal state"
        )

    def _on_cancel_response(self, future, token: int) -> None:
        if not self._callback_is_current(token, "cancelling_goal"):
            return
        reason = self.pending_failure_reason or "trajectory cancellation requested"
        try:
            response = future.result()
        except Exception as error:
            self._halt_active(f"{reason}; goal cancellation response failed: {error}")
            return
        if response is None or not response.goals_canceling:
            self._halt_active(f"{reason}; controller rejected the cancellation request")
            return
        self.get_logger().warning(
            "controller accepted cancellation; waiting for the terminal action result"
        )

    def _verify_segment(self, now: float) -> None:
        if self.active_detection is None or self.segment_index >= len(self.segments):
            return
        measured = self._fresh_measured_state(now)
        segment = self.segments[self.segment_index]
        if (
            measured is not None
            and all(
                self.measured_at_monotonic[name] >= self.segment_sent_monotonic
                for name in self.joint_names
            )
        ):
            arm_errors = [
                abs(actual - target)
                for actual, target in zip(measured[:5], segment.target[:5])
            ]
            gripper_error = abs(measured[5] - segment.target[5])
            gripper_tolerance = (
                self.grasp_contact_tolerance_m
                if segment.state == CycleState.GRASP
                else self.gripper_goal_tolerance_m
            )
            if (
                max(arm_errors, default=0.0) <= self.arm_goal_tolerance_rad
                and gripper_error <= gripper_tolerance
            ):
                self._complete_segment(segment, measured)
                return

        if now < self.phase_deadline_monotonic:
            return

        if measured is None:
            self._fail_active("no fresh measured joint state for segment verification")
            return
        arm_error = max(
            abs(actual - target)
            for actual, target in zip(measured[:5], segment.target[:5])
        )
        gripper_error = abs(measured[5] - segment.target[5])
        gripper_tolerance = (
            self.grasp_contact_tolerance_m
            if segment.state == CycleState.GRASP
            else self.gripper_goal_tolerance_m
        )
        self._fail_active(
            f"measured target mismatch after {segment.state.value}: "
            f"arm={arm_error:.4f}rad (limit {self.arm_goal_tolerance_rad:.4f}), "
            f"gripper={gripper_error:.4f}m "
            f"(limit {gripper_tolerance:.4f})"
        )

    def _complete_segment(
        self,
        segment: _TrajectorySegment,
        measured: tuple[float, float, float, float, float, float],
    ) -> None:
        if self.phase != "verifying_goal":
            return
        # Claim the transition before any publisher side effect. This keeps a
        # timer callback and an action-result callback from advancing twice.
        self.phase = "advancing_segment"
        detection = self.active_detection
        if detection is None:
            return

        self.object_positions = dict(segment.frames[-1].object_positions)
        self.measured_positions.update(zip(self.joint_names, measured))
        self.segment_index += 1

        # Attachment transitions are acknowledged by the Gazebo adapter before
        # motion continues, so a missing service can never become a false success.
        if segment.state == CycleState.GRASP:
            gripper_error = abs(measured[5] - segment.target[5])
            if gripper_error > self.gripper_goal_tolerance_m:
                self.get_logger().info(
                    f"gripper contact inferred at {measured[5]:.4f}m "
                    f"(empty-close target {segment.target[5]:.4f}m)"
                )
            tool, _ = self.kinematics.forward(measured[:5])
            distance = tool.distance_to(detection.position)
            if distance > 0.08:
                self._fail_active(
                    f"tool is {distance:.3f}m from {detection.object_id} at grasp"
                )
                return
            self._begin_attachment_transition("attach", detection.object_id)
            return
        if segment.state == CycleState.RELEASE:
            self._begin_attachment_transition("detach", detection.object_id)
            return
        self._send_current_segment()

    def _begin_attachment_transition(self, action: str, object_id: str) -> None:
        self.pending_attachment_action = action
        self.phase = "waiting_for_attachment"
        self.phase_deadline_monotonic = time.monotonic() + self.attachment_timeout_s
        self._publish_attachment(action, object_id)

    def _on_attachment_status(self, message: String) -> None:
        if self.phase != "waiting_for_attachment" or self.active_detection is None:
            return
        state, separator, remainder = message.data.partition(":")
        object_id, detail_separator, detail = remainder.partition(":")
        if not separator or not detail_separator:
            return
        if object_id != self.active_detection.object_id:
            return
        if state == "failed":
            self._fail_active(f"attachment failed: {detail}")
            return
        expected_state = (
            "attached" if self.pending_attachment_action == "attach" else "detached"
        )
        if state != expected_state:
            return
        self.attachment_active = state == "attached"
        self.pending_attachment_action = ""
        self.get_logger().info(f"attachment confirmed: {state}:{object_id}")
        self._send_current_segment()

    def _publish_attachment(self, action: str, object_id: str) -> None:
        message = String()
        message.data = f"{action}:{object_id}"
        self.attachment_publisher.publish(message)
        self.get_logger().info(f"attachment command: {message.data}")

    def _publish_segment_status(self, state: CycleState, detail: str) -> None:
        detection = self.active_detection
        if detection is None or self.terminal_published:
            return
        message = CellStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.state = state.value
        message.active_object_id = detection.object_id
        message.completed_cycles = self.completed
        message.failed_cycles = self.failed
        message.current_cycle_time_s = float(
            max(0.0, time.monotonic() - self.cycle_started_monotonic)
        )
        message.message = detail
        self.status_publisher.publish(message)

    def _finish_active_success(self) -> None:
        report = self.active_plan.report if self.active_plan is not None else None
        if report is None or not report.success:
            self._fail_active(
                report.reason if report is not None else "cycle report unavailable"
            )
            return
        self.completed += 1
        elapsed_s = time.monotonic() - self.cycle_started_monotonic
        self._publish_terminal(
            object_id=report.object_id,
            success=True,
            elapsed_s=elapsed_s,
            reason=f"routed to {report.destination}",
        )
        self.terminal_published = True
        self.get_logger().info(
            f"cycle finished: object={report.object_id}, state=COMPLETE, "
            f"elapsed={elapsed_s:.2f}s"
        )
        self._clear_active_cycle()

    def _fail_active(self, reason: str) -> None:
        detection = self._record_active_failure(reason)
        if detection is None:
            return
        if self.attachment_active:
            self._publish_attachment("detach", detection.object_id)
            self.attachment_active = False
        self._clear_active_cycle()

    def _halt_active(self, reason: str) -> None:
        detection = self._record_active_failure(reason)
        if detection is None:
            return
        self.phase = "faulted"
        self.phase_deadline_monotonic = 0.0
        self.execution_deadline_ros_ns = 0
        self.execution_watchdog_deadline_monotonic = 0.0
        self.queue.clear()
        if self.attachment_active:
            self.get_logger().error(
                f"attachment for {detection.object_id} remains active because controller "
                "termination was not confirmed"
            )
        self.get_logger().error(
            "coordinator fault latched; no further goals will be sent until the node is restarted"
        )

    def _record_active_failure(self, reason: str) -> Detection | None:
        detection = self.active_detection
        if detection is None or self.terminal_published:
            return None
        self.failed += 1
        elapsed_s = time.monotonic() - self.cycle_started_monotonic
        self._publish_terminal(
            object_id=detection.object_id,
            success=False,
            elapsed_s=elapsed_s,
            reason=reason or "cycle failed",
        )
        self.terminal_published = True
        self.get_logger().error(
            f"cycle finished: object={detection.object_id}, state=FAILED, "
            f"elapsed={elapsed_s:.2f}s, reason={reason}"
        )
        return detection

    def _publish_terminal(
        self, *, object_id: str, success: bool, elapsed_s: float, reason: str
    ) -> None:
        message = CellStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.state = CycleState.COMPLETE.value if success else CycleState.FAILED.value
        message.active_object_id = object_id
        message.completed_cycles = self.completed
        message.failed_cycles = self.failed
        message.current_cycle_time_s = float(max(0.0, elapsed_s))
        message.message = reason
        self.status_publisher.publish(message)

    def _clear_active_cycle(self) -> None:
        self.active_detection = None
        self.active_plan = None
        self.segments = []
        self.segment_index = 0
        self.phase = "idle"
        self.phase_deadline_monotonic = 0.0
        self.execution_deadline_ros_ns = 0
        self.execution_watchdog_deadline_monotonic = 0.0
        self.active_goal_handle = None
        self.pending_failure_reason = ""
        self.pending_attachment_action = ""

    def _callback_is_current(self, token: int, expected_phase: str) -> bool:
        return (
            self.active_detection is not None
            and token == self.cycle_token
            and self.phase == expected_phase
            and not self.terminal_published
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PickCoordinatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # rclpy can raise from take_message while SIGINT is tearing down DDS.
        # Preserve genuine runtime failures that occur while the context is live.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
