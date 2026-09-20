from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...]


def _smoothstep5(tau: float) -> tuple[float, float]:
    position_scale = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    velocity_scale = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    return position_scale, velocity_scale


def generate_quintic(
    start: Iterable[float],
    goal: Iterable[float],
    max_velocity: Iterable[float] | float = 1.1,
    sample_period: float = 0.04,
    minimum_duration: float = 0.6,
) -> list[TrajectorySample]:
    start_values = tuple(float(value) for value in start)
    goal_values = tuple(float(value) for value in goal)
    if len(start_values) != len(goal_values) or not start_values:
        raise ValueError("start and goal must have the same non-zero length")
    if sample_period <= 0.0:
        raise ValueError("sample_period must be positive")

    if isinstance(max_velocity, (float, int)):
        velocity_limits = (float(max_velocity),) * len(start_values)
    else:
        velocity_limits = tuple(float(value) for value in max_velocity)
    if len(velocity_limits) != len(start_values) or any(value <= 0.0 for value in velocity_limits):
        raise ValueError("max_velocity must contain one positive value per joint")

    deltas = tuple(goal_value - start_value for start_value, goal_value in zip(start_values, goal_values))
    duration = max(
        minimum_duration,
        max(1.875 * abs(delta) / limit for delta, limit in zip(deltas, velocity_limits)),
    )
    step_count = max(2, int(ceil(duration / sample_period)))
    duration = step_count * sample_period

    samples: list[TrajectorySample] = []
    for index in range(step_count + 1):
        time_s = min(duration, index * sample_period)
        tau = time_s / duration
        position_scale, velocity_scale = _smoothstep5(tau)
        positions = tuple(start_value + delta * position_scale for start_value, delta in zip(start_values, deltas))
        velocities = tuple(delta * velocity_scale / duration for delta in deltas)
        samples.append(TrajectorySample(time_s, positions, velocities))
    return samples
