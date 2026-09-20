from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import yaml

from .kinematics import ArmKinematics, IKError
from .models import CellFrame, CyclePlan, CycleReport, CycleState, Detection, Vector3
from .safety import SafetyError, SafetyMonitor, WorkspaceLimits
from .trajectory import generate_quintic


@dataclass(frozen=True)
class CellLayout:
    destinations: dict[str, Vector3]
    class_to_destination: dict[str, str]

    @classmethod
    def default(cls) -> "CellLayout":
        return cls(
            destinations={
                "accepted_bin": Vector3(0.30, 0.30, 0.075),
                "reject_bin": Vector3(0.30, -0.30, 0.075),
                "rework_bin": Vector3(0.48, 0.25, 0.075),
            },
            class_to_destination={
                "accepted": "accepted_bin",
                "scratch": "reject_bin",
                "dent": "reject_bin",
                "unknown": "rework_bin",
            },
        )


class WorkflowPlanner:
    def __init__(
        self,
        kinematics: ArmKinematics | None = None,
        safety: SafetyMonitor | None = None,
        layout: CellLayout | None = None,
        sample_period: float = 0.04,
    ) -> None:
        self.kinematics = kinematics or ArmKinematics()
        self.safety = safety or SafetyMonitor()
        self.layout = layout or CellLayout.default()
        self.sample_period = sample_period
        self.open_gripper = 0.032
        self.closed_gripper = 0.004
        self.grasp_offset = 0.040

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        kinematics: ArmKinematics | None = None,
        sample_period: float | None = None,
    ) -> "WorkflowPlanner":
        """Build a planner from the version-controlled cell configuration."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cell = payload["cell"]
        workspace = cell["workspace"]
        safety = SafetyMonitor(
            WorkspaceLimits(
                min_radius=float(workspace["min_radius_m"]),
                max_radius=float(workspace["max_radius_m"]),
                min_z=float(workspace["min_z_m"]),
                max_z=float(workspace["max_z_m"]),
                max_abs_y=float(workspace["max_abs_y_m"]),
                minimum_confidence=float(workspace["minimum_confidence"]),
            )
        )
        destinations = {
            name: Vector3(*(float(value) for value in coordinates))
            for name, coordinates in cell["destinations"].items()
        }
        layout = CellLayout(
            destinations=destinations,
            class_to_destination={
                str(name): str(destination)
                for name, destination in cell["routing"].items()
            },
        )
        planner = cls(
            kinematics=kinematics,
            safety=safety,
            layout=layout,
            sample_period=(
                float(cell["command_period_s"])
                if sample_period is None
                else float(sample_period)
            ),
        )
        planner.open_gripper = float(cell["gripper"]["open_m"])
        planner.closed_gripper = float(cell["gripper"]["closed_m"])
        return planner

    def plan_cycle(
        self,
        detection: Detection,
        start_joints: tuple[float, float, float, float, float] | None = None,
        start_gripper: float | None = None,
        object_positions: dict[str, Vector3] | None = None,
        start_time_s: float = 0.0,
    ) -> CyclePlan:
        current_joints = start_joints or self.kinematics.home
        current_gripper = self.open_gripper if start_gripper is None else start_gripper
        positions = dict(object_positions or {detection.object_id: detection.position})
        frames: list[CellFrame] = []
        time_s = start_time_s
        attached = False

        def append_frame(state: CycleState) -> None:
            tool_position, _ = self.kinematics.forward(current_joints)
            if attached:
                positions[detection.object_id] = Vector3(
                    tool_position.x,
                    tool_position.y,
                    max(0.065, tool_position.z - self.grasp_offset),
                )
            frames.append(
                CellFrame(
                    time_s=time_s,
                    state=state,
                    joints=current_joints,
                    gripper_m=current_gripper,
                    tool_position=tool_position,
                    active_object_id=detection.object_id,
                    object_positions=dict(positions),
                )
            )

        def move_to(state: CycleState, target: Vector3, pitch: float = -pi / 2.0) -> None:
            nonlocal current_joints, time_s
            self.safety.validate_pose(target, label=state.value.lower())
            goal = self.kinematics.inverse(target, tool_pitch=pitch, seed=current_joints)
            samples = generate_quintic(
                current_joints,
                goal,
                max_velocity=(1.0, 0.9, 1.0, 1.2, 1.4),
                sample_period=self.sample_period,
            )
            segment_start = time_s
            for sample in samples[1:]:
                current_joints = tuple(sample.positions)  # type: ignore[assignment]
                time_s = segment_start + sample.time_s
                append_frame(state)

        def set_gripper(state: CycleState, goal: float, duration: float = 0.45) -> None:
            nonlocal current_gripper, time_s
            start = current_gripper
            steps = max(2, round(duration / self.sample_period))
            for index in range(1, steps + 1):
                tau = index / steps
                smooth = 3.0 * tau**2 - 2.0 * tau**3
                current_gripper = start + (goal - start) * smooth
                time_s += self.sample_period
                append_frame(state)

        append_frame(CycleState.IDLE)
        try:
            self.safety.validate_detection(detection)
            append_frame(CycleState.VALIDATING)
            destination_name = self.layout.class_to_destination.get(detection.class_name, "rework_bin")
            destination = self.layout.destinations[destination_name]
            self.safety.validate_pose(destination, label=destination_name)

            pregrasp = detection.position.with_z(max(0.25, detection.position.z + 0.18))
            approach = detection.position.with_z(detection.position.z + 0.045)
            lift = detection.position.with_z(0.245)
            transfer = destination.with_z(0.245)
            place = destination.with_z(0.12)
            retreat = destination.with_z(0.245)

            move_to(CycleState.PREGRASP, pregrasp)
            move_to(CycleState.APPROACH, approach)
            set_gripper(CycleState.GRASP, self.closed_gripper)
            attached = True
            move_to(CycleState.LIFT, lift)
            move_to(CycleState.TRANSFER, transfer)
            move_to(CycleState.PLACE, place)
            set_gripper(CycleState.RELEASE, self.open_gripper)
            attached = False
            positions[detection.object_id] = destination
            move_to(CycleState.RETREAT, retreat)

            home_samples = generate_quintic(
                current_joints,
                self.kinematics.home,
                max_velocity=(1.0, 0.9, 1.0, 1.2, 1.4),
                sample_period=self.sample_period,
            )
            segment_start = time_s
            for sample in home_samples[1:]:
                current_joints = tuple(sample.positions)  # type: ignore[assignment]
                time_s = segment_start + sample.time_s
                append_frame(CycleState.HOME)
            append_frame(CycleState.COMPLETE)
            report = CycleReport(
                object_id=detection.object_id,
                class_name=detection.class_name,
                destination=destination_name,
                success=True,
                duration_s=time_s - start_time_s,
            )
        except (IKError, SafetyError, ValueError) as error:
            append_frame(CycleState.FAILED)
            report = CycleReport(
                object_id=detection.object_id,
                class_name=detection.class_name,
                destination="",
                success=False,
                duration_s=time_s - start_time_s,
                reason=str(error),
            )
        return CyclePlan(frames=frames, report=report)
