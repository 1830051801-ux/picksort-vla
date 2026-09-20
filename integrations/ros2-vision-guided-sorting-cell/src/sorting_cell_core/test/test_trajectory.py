import pytest

from sorting_cell_core.trajectory import generate_quintic


def test_quintic_trajectory_has_exact_endpoints_and_zero_endpoint_velocity() -> None:
    trajectory = generate_quintic((0.0, -0.2), (1.0, 0.4), max_velocity=(0.8, 1.0))
    assert trajectory[0].positions == pytest.approx((0.0, -0.2))
    assert trajectory[-1].positions == pytest.approx((1.0, 0.4))
    assert trajectory[0].velocities == pytest.approx((0.0, 0.0))
    assert trajectory[-1].velocities == pytest.approx((0.0, 0.0), abs=1e-10)
    assert all(later.time_s > earlier.time_s for earlier, later in zip(trajectory, trajectory[1:]))


def test_quintic_respects_velocity_limits() -> None:
    limits = (0.7, 0.4, 1.2)
    trajectory = generate_quintic((0.0, 0.0, 0.0), (1.2, -0.8, 0.2), max_velocity=limits)
    for sample in trajectory:
        assert all(abs(value) <= limit + 1e-8 for value, limit in zip(sample.velocities, limits))
