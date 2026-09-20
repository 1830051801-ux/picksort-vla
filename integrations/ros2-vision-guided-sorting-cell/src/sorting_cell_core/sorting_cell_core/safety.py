from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .models import Detection, Vector3


class SafetyError(ValueError):
    """Raised when a target violates a cell safety constraint."""


@dataclass(frozen=True)
class WorkspaceLimits:
    min_radius: float = 0.18
    max_radius: float = 0.69
    min_z: float = 0.055
    max_z: float = 0.56
    max_abs_y: float = 0.40
    minimum_confidence: float = 0.70


class SafetyMonitor:
    def __init__(self, limits: WorkspaceLimits | None = None) -> None:
        self.limits = limits or WorkspaceLimits()

    def validate_detection(self, detection: Detection) -> None:
        if detection.confidence < self.limits.minimum_confidence:
            raise SafetyError(
                f"confidence {detection.confidence:.2f} is below {self.limits.minimum_confidence:.2f}"
            )
        self.validate_pose(detection.position, label=detection.object_id)

    def validate_pose(self, pose: Vector3, label: str = "target") -> None:
        radius = hypot(pose.x, pose.y)
        if not self.limits.min_radius <= radius <= self.limits.max_radius:
            raise SafetyError(f"{label} radial distance {radius:.3f} m is outside workspace")
        if not self.limits.min_z <= pose.z <= self.limits.max_z:
            raise SafetyError(f"{label} height {pose.z:.3f} m is outside workspace")
        if abs(pose.y) > self.limits.max_abs_y:
            raise SafetyError(f"{label} lateral coordinate {pose.y:.3f} m is outside workspace")
