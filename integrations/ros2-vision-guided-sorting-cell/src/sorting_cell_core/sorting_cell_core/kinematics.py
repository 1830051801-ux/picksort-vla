from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, hypot, pi, sin
from typing import Iterable

from .models import Vector3


class IKError(ValueError):
    """Raised when a Cartesian target has no valid joint solution."""


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float

    def contains(self, value: float, tolerance: float = 1e-8) -> bool:
        return self.lower - tolerance <= value <= self.upper + tolerance


class ArmKinematics:
    """Analytical yaw-plus-planar IK for the five-axis sorting arm."""

    joint_names = (
        "base_yaw_joint",
        "shoulder_joint",
        "elbow_joint",
        "wrist_pitch_joint",
        "wrist_roll_joint",
    )

    def __init__(
        self,
        base_height: float = 0.21,
        upper_arm: float = 0.32,
        forearm: float = 0.28,
        tool_length: float = 0.20,
    ) -> None:
        self.base_height = base_height
        self.upper_arm = upper_arm
        self.forearm = forearm
        self.tool_length = tool_length
        self.limits = (
            JointLimit(-2.967, 2.967),
            JointLimit(-1.40, 1.92),
            JointLimit(-2.62, 2.62),
            JointLimit(-2.88, 2.88),
            JointLimit(-3.14, 3.14),
        )

    @property
    def home(self) -> tuple[float, float, float, float, float]:
        return (0.0, 0.80, -1.50, 0.70, 0.0)

    def inverse(
        self,
        target: Vector3,
        tool_pitch: float = -pi / 2.0,
        wrist_roll: float = 0.0,
        seed: Iterable[float] | None = None,
    ) -> tuple[float, float, float, float, float]:
        yaw = atan2(target.y, target.x)
        radial = hypot(target.x, target.y)
        wrist_r = radial - self.tool_length * cos(tool_pitch)
        wrist_z = target.z - self.base_height - self.tool_length * sin(tool_pitch)

        numerator = wrist_r**2 + wrist_z**2 - self.upper_arm**2 - self.forearm**2
        denominator = 2.0 * self.upper_arm * self.forearm
        cosine_elbow = numerator / denominator
        if cosine_elbow < -1.000001 or cosine_elbow > 1.000001:
            raise IKError(f"target outside arm reach: {target}")
        cosine_elbow = max(-1.0, min(1.0, cosine_elbow))

        candidates: list[tuple[float, float, float, float, float]] = []
        for elbow in (-acos(cosine_elbow), acos(cosine_elbow)):
            shoulder = atan2(wrist_z, wrist_r) - atan2(
                self.forearm * sin(elbow),
                self.upper_arm + self.forearm * cos(elbow),
            )
            wrist_pitch = tool_pitch - shoulder - elbow
            candidate = (yaw, shoulder, elbow, wrist_pitch, wrist_roll)
            if self.within_limits(candidate):
                candidates.append(candidate)

        if not candidates:
            raise IKError(f"target reachable geometrically but violates joint limits: {target}")

        reference = tuple(seed) if seed is not None else self.home
        return min(candidates, key=lambda q: sum((a - b) ** 2 for a, b in zip(q, reference)))

    def forward(self, joints: Iterable[float]) -> tuple[Vector3, float]:
        q1, q2, q3, q4, _ = tuple(joints)
        q23 = q2 + q3
        pitch = q23 + q4
        radial = (
            self.upper_arm * cos(q2)
            + self.forearm * cos(q23)
            + self.tool_length * cos(pitch)
        )
        z = (
            self.base_height
            + self.upper_arm * sin(q2)
            + self.forearm * sin(q23)
            + self.tool_length * sin(pitch)
        )
        return Vector3(radial * cos(q1), radial * sin(q1), z), pitch

    def joint_positions(self, joints: Iterable[float]) -> tuple[Vector3, ...]:
        q1, q2, q3, q4, _ = tuple(joints)
        radial_1 = self.upper_arm * cos(q2)
        z_1 = self.base_height + self.upper_arm * sin(q2)
        radial_2 = radial_1 + self.forearm * cos(q2 + q3)
        z_2 = z_1 + self.forearm * sin(q2 + q3)
        radial_3 = radial_2 + self.tool_length * cos(q2 + q3 + q4)
        z_3 = z_2 + self.tool_length * sin(q2 + q3 + q4)
        return (
            Vector3(0.0, 0.0, 0.04),
            Vector3(0.0, 0.0, self.base_height),
            Vector3(radial_1 * cos(q1), radial_1 * sin(q1), z_1),
            Vector3(radial_2 * cos(q1), radial_2 * sin(q1), z_2),
            Vector3(radial_3 * cos(q1), radial_3 * sin(q1), z_3),
        )

    def within_limits(self, joints: Iterable[float]) -> bool:
        values = tuple(joints)
        return len(values) == len(self.limits) and all(
            limit.contains(value) for value, limit in zip(values, self.limits)
        )
