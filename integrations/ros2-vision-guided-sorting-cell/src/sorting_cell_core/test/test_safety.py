import pytest

from sorting_cell_core.models import Detection, Vector3
from sorting_cell_core.safety import SafetyError, SafetyMonitor


def test_valid_detection_passes() -> None:
    SafetyMonitor().validate_detection(Detection("part", "accepted", Vector3(0.45, 0.1, 0.075), 0.95))


@pytest.mark.parametrize(
    "detection",
    [
        Detection("too_far", "accepted", Vector3(0.9, 0.0, 0.075), 0.95),
        Detection("below_table", "accepted", Vector3(0.4, 0.0, 0.01), 0.95),
        Detection("low_confidence", "accepted", Vector3(0.4, 0.0, 0.075), 0.4),
    ],
)
def test_invalid_detection_is_rejected(detection: Detection) -> None:
    with pytest.raises(SafetyError):
        SafetyMonitor().validate_detection(detection)
