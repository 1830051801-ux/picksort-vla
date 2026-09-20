from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .kinematics import ArmKinematics
from .models import CellFrame, CycleReport, Part, RunSummary, Vector3
from .workflow import WorkflowPlanner


DEFAULT_PARTS = (
    Part("part_blue", "accepted", Vector3(0.47, -0.13, 0.075), 0.98),
    Part("part_red", "scratch", Vector3(0.50, 0.02, 0.075), 0.96),
    Part("part_orange", "unknown", Vector3(0.43, 0.15, 0.075), 0.91),
)


class HeadlessCell:
    def __init__(self, planner: WorkflowPlanner | None = None) -> None:
        self.planner = planner or WorkflowPlanner()

    def run(self, parts: tuple[Part, ...] = DEFAULT_PARTS) -> tuple[list[CellFrame], list[CycleReport], RunSummary]:
        object_positions = {part.object_id: part.position for part in parts}
        frames: list[CellFrame] = []
        reports: list[CycleReport] = []
        joints = self.planner.kinematics.home
        gripper = self.planner.open_gripper
        time_s = 0.0

        for part in parts:
            plan = self.planner.plan_cycle(
                part.detection(),
                start_joints=joints,
                start_gripper=gripper,
                object_positions=object_positions,
                start_time_s=time_s,
            )
            if plan.frames:
                frames.extend(plan.frames)
                last = plan.frames[-1]
                joints = last.joints
                gripper = last.gripper_m
                object_positions = dict(last.object_positions)
                time_s = last.time_s + 0.4
            if plan.report is not None:
                reports.append(plan.report)

        completed = sum(report.success for report in reports)
        failed = len(reports) - completed
        cycle_times = [report.duration_s for report in reports if report.success]
        summary = RunSummary(
            total=len(reports),
            completed=completed,
            failed=failed,
            success_rate=completed / len(reports) if reports else 0.0,
            average_cycle_s=sum(cycle_times) / len(cycle_times) if cycle_times else 0.0,
            total_runtime_s=frames[-1].time_s if frames else 0.0,
        )
        return frames, reports, summary


def write_results(output_dir: Path, reports: list[CycleReport], summary: RunSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "cycles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(reports[0]).keys()) if reports else [])
        if reports:
            writer.writeheader()
            writer.writerows(asdict(report) for report in reports)


def serialize_frames(frames: list[CellFrame], output_path: Path, stride: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for frame in frames[:: max(1, stride)]:
        payload.append(
            {
                "time_s": round(frame.time_s, 3),
                "state": frame.state.value,
                "joints": [round(value, 5) for value in frame.joints],
                "gripper_m": round(frame.gripper_m, 5),
                "tool": asdict(frame.tool_position),
                "active_object_id": frame.active_object_id,
                "objects": {key: asdict(value) for key, value in frame.object_positions.items()},
            }
        )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
