import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sorting_cell_core.nodes.color_perception_node import (  # noqa: E402
    CameraIntrinsics,
    MultiFrameStabilizer,
    ProjectedDetection,
    decode_ros_image,
    detect_projected_parts,
    pixel_to_base,
)


INTRINSICS = CameraIntrinsics(fx=550.0, fy=550.0, cx=320.0, cy=240.0)


def _pixel_for_base(x: float, y: float) -> tuple[int, int]:
    distance = 0.95 - 0.075
    u = INTRINSICS.cx - y * INTRINSICS.fx / distance
    v = INTRINSICS.cy - (x - 0.45) * INTRINSICS.fy / distance
    return round(u), round(v)


def _draw_centered_rectangle(
    image: np.ndarray,
    x: float,
    y: float,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    u, v = _pixel_for_base(x, y)
    cv2.rectangle(
        image,
        (u - width // 2, v - height // 2),
        (u + width // 2, v + height // 2),
        color,
        thickness=-1,
    )


def test_decode_rgb_image_honors_row_padding() -> None:
    rows = np.array(
        [
            [255, 0, 0, 0, 255, 0, 99, 99],
            [0, 0, 255, 255, 255, 255, 99, 99],
        ],
        dtype=np.uint8,
    )

    decoded = decode_ros_image(rows.tobytes(), width=2, height=2, encoding="rgb8", step=8)

    assert decoded.shape == (2, 2, 3)
    assert decoded[0, 0].tolist() == [0, 0, 255]
    assert decoded[0, 1].tolist() == [0, 255, 0]
    assert decoded[1, 0].tolist() == [255, 0, 0]


def test_pixel_projection_matches_overhead_camera_convention() -> None:
    assert pixel_to_base(320.0, 240.0, INTRINSICS) == pytest.approx((0.45, 0.0, 0.075))

    u, v = _pixel_for_base(0.54, -0.13)
    x, y, z = pixel_to_base(float(u), float(v), INTRINSICS)

    assert (x, y, z) == pytest.approx((0.54, -0.13, 0.075), abs=0.001)


def test_detects_three_parts_and_filters_same_color_bins_and_robot() -> None:
    image = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Workpieces use the rendered Gazebo colors and projected pixel sizes.
    _draw_centered_rectangle(image, 0.47, -0.13, 34, 28, (220, 120, 20))
    _draw_centered_rectangle(image, 0.50, 0.02, 34, 34, (20, 20, 220))
    _draw_centered_rectangle(image, 0.43, 0.15, 40, 22, (10, 120, 245))

    # Large same-color bins and an elongated blue robot link are distractors.
    _draw_centered_rectangle(image, 0.30, -0.30, 100, 90, (20, 20, 220))
    _draw_centered_rectangle(image, 0.48, 0.25, 100, 90, (10, 120, 245))
    _draw_centered_rectangle(image, 0.20, 0.00, 160, 35, (220, 120, 20))

    detections = detect_projected_parts(image, INTRINSICS)

    assert [(item.color_name, item.class_name) for item in detections] == [
        ("blue", "accepted"),
        ("red", "scratch"),
        ("orange", "unknown"),
    ]
    positions = {item.color_name: (item.x, item.y) for item in detections}
    assert positions["blue"] == pytest.approx((0.47, -0.13), abs=0.003)
    assert positions["red"] == pytest.approx((0.50, 0.02), abs=0.003)
    # The orange ROI intentionally clips the adjacent same-color bin edge.
    assert positions["orange"] == pytest.approx((0.43, 0.15), abs=0.008)
    assert all(item.confidence >= 0.70 for item in detections)


def test_stabilizer_requires_consecutive_low_jitter_observations() -> None:
    gate = MultiFrameStabilizer(required_frames=3, max_jitter_m=0.01)

    def observation(x: float, y: float) -> ProjectedDetection:
        return ProjectedDetection("blue", "accepted", 400.0, 225.0, x, y, 0.075, 0.95)

    assert gate.update([observation(0.470, -0.130)]) == []
    assert gate.update([observation(0.472, -0.129)]) == []
    stable = gate.update([observation(0.469, -0.131)])
    assert len(stable) == 1
    assert (stable[0].x, stable[0].y) == pytest.approx((0.470, -0.130))

    # A missing frame breaks consecutiveness, so two new frames are not enough.
    assert gate.update([]) == []
    assert gate.update([observation(0.470, -0.130)]) == []
    assert gate.update([observation(0.471, -0.130)]) == []
