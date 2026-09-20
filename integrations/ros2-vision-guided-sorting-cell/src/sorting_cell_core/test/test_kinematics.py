from math import pi

import pytest

from sorting_cell_core.kinematics import ArmKinematics, IKError
from sorting_cell_core.models import Vector3


@pytest.mark.parametrize(
    "target",
    [
        Vector3(0.47, -0.13, 0.12),
        Vector3(0.30, 0.30, 0.12),
        Vector3(0.48, 0.25, 0.245),
        Vector3(0.43, 0.15, 0.25),
    ],
)
def test_inverse_forward_round_trip(target: Vector3) -> None:
    model = ArmKinematics()
    joints = model.inverse(target, tool_pitch=-pi / 2.0)
    solved, pitch = model.forward(joints)
    assert solved.distance_to(target) < 1e-7
    assert pitch == pytest.approx(-pi / 2.0, abs=1e-7)
    assert model.within_limits(joints)


def test_unreachable_target_is_rejected() -> None:
    with pytest.raises(IKError):
        ArmKinematics().inverse(Vector3(1.2, 0.0, 0.1))
