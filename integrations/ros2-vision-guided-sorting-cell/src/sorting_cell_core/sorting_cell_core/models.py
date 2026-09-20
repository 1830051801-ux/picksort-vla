from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Mapping


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def distance_to(self, other: "Vector3") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def with_z(self, z: float) -> "Vector3":
        return Vector3(self.x, self.y, z)


@dataclass(frozen=True)
class Detection:
    object_id: str
    class_name: str
    position: Vector3
    confidence: float = 1.0


@dataclass(frozen=True)
class Part:
    object_id: str
    class_name: str
    position: Vector3
    confidence: float = 0.98

    def detection(self) -> Detection:
        return Detection(self.object_id, self.class_name, self.position, self.confidence)


class CycleState(str, Enum):
    IDLE = "IDLE"
    VALIDATING = "VALIDATING"
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    HOME = "HOME"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CellFrame:
    time_s: float
    state: CycleState
    joints: tuple[float, float, float, float, float]
    gripper_m: float
    tool_position: Vector3
    active_object_id: str
    object_positions: Mapping[str, Vector3]


@dataclass(frozen=True)
class CycleReport:
    object_id: str
    class_name: str
    destination: str
    success: bool
    duration_s: float
    reason: str = ""


@dataclass
class CyclePlan:
    frames: list[CellFrame] = field(default_factory=list)
    report: CycleReport | None = None


@dataclass(frozen=True)
class RunSummary:
    total: int
    completed: int
    failed: int
    success_rate: float
    average_cycle_s: float
    total_runtime_s: float
