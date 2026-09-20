from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from math import hypot
from typing import Iterable, Sequence

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import CameraInfo, Image

    from sorting_cell_interfaces.msg import CellStatus, DetectedObject
except ImportError as exc:  # Keep the image-processing helpers usable without ROS 2.
    rclpy = None
    Node = object  # type: ignore[assignment,misc]
    qos_profile_sensor_data = None
    CameraInfo = Image = CellStatus = DetectedObject = object  # type: ignore[misc,assignment]
    _ROS_IMPORT_ERROR: ImportError | None = exc
else:
    _ROS_IMPORT_ERROR = None


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class ColorModel:
    color_name: str
    class_name: str
    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]


@dataclass(frozen=True)
class ColorCandidate:
    color_name: str
    class_name: str
    u: float
    v: float
    area_px: float
    width_px: int
    height_px: int
    confidence: float


@dataclass(frozen=True)
class ProjectedDetection:
    color_name: str
    class_name: str
    u: float
    v: float
    x: float
    y: float
    z: float
    confidence: float

    @property
    def object_id(self) -> str:
        return f"part_{self.color_name}"


COLOR_MODELS = (
    ColorModel("blue", "accepted", (((94, 95, 55), (128, 255, 255)),)),
    ColorModel(
        "red",
        "scratch",
        (
            ((0, 110, 65), (7, 255, 255)),
            ((170, 110, 65), (179, 255, 255)),
        ),
    ),
    ColorModel("orange", "unknown", (((8, 120, 75), (19, 255, 255)),)),
)


def decode_ros_image(
    data: bytes | bytearray | memoryview | Sequence[int],
    width: int,
    height: int,
    encoding: str,
    step: int = 0,
) -> np.ndarray:
    """Decode common 8-bit sensor_msgs/Image layouts into a BGR array."""

    encoding = encoding.lower()
    channels_by_encoding = {
        "mono8": 1,
        "8uc1": 1,
        "rgb8": 3,
        "bgr8": 3,
        "8uc3": 3,
        "rgba8": 4,
        "bgra8": 4,
        "8uc4": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding: {encoding}")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    channels = channels_by_encoding[encoding]
    packed_step = width * channels
    row_step = step or packed_step
    if row_step < packed_step:
        raise ValueError(f"image step {row_step} is smaller than packed row {packed_step}")

    try:
        raw = np.frombuffer(data, dtype=np.uint8)
    except (TypeError, ValueError):
        raw = np.asarray(data, dtype=np.uint8)
    required = row_step * height
    if raw.size < required:
        raise ValueError(f"image has {raw.size} bytes but {required} are required")

    packed = raw[:required].reshape(height, row_step)[:, :packed_step]
    if channels == 1:
        mono = packed.reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    image = packed.reshape(height, width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding in ("bgra8", "8uc4"):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def pixel_to_base(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
    *,
    camera_x: float = 0.45,
    camera_z: float = 0.95,
    plane_z: float = 0.075,
) -> tuple[float, float, float]:
    """Project a pixel onto the horizontal workpiece plane in base_link."""

    distance = camera_z - plane_z
    if distance <= 0.0:
        raise ValueError("the camera must be above the projection plane")
    x = camera_x - (v - intrinsics.cy) * distance / intrinsics.fy
    y = -(u - intrinsics.cx) * distance / intrinsics.fx
    return x, y, plane_z


def find_color_candidates(
    bgr_image: np.ndarray,
    *,
    min_area_px: float = 120.0,
    max_area_px: float = 4000.0,
    max_aspect_ratio: float = 2.6,
) -> list[ColorCandidate]:
    """Segment supported colors and reject very small, large, or slender blobs."""

    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR image")
    if bgr_image.dtype != np.uint8:
        raise ValueError("expected an 8-bit BGR image")

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidates: list[ColorCandidate] = []

    for model in COLOR_MODELS:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in model.hsv_ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8)),
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not min_area_px <= area <= max_area_px:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if min(width, height) <= 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > max_aspect_ratio:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                continue

            u = float(moments["m10"] / moments["m00"])
            v = float(moments["m01"] / moments["m00"])
            fill_ratio = min(1.0, area / float(width * height))
            saturation = float(hsv[int(round(v)), int(round(u)), 1]) / 255.0
            area_quality = min(1.0, area / 600.0)
            confidence = min(0.99, 0.72 + 0.10 * fill_ratio + 0.09 * saturation + 0.08 * area_quality)
            candidates.append(
                ColorCandidate(
                    color_name=model.color_name,
                    class_name=model.class_name,
                    u=u,
                    v=v,
                    area_px=area,
                    width_px=width,
                    height_px=height,
                    confidence=confidence,
                )
            )

    color_order = {model.color_name: index for index, model in enumerate(COLOR_MODELS)}
    return sorted(candidates, key=lambda item: (color_order[item.color_name], -item.confidence))


def detect_projected_parts(
    bgr_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    camera_x: float = 0.45,
    camera_z: float = 0.95,
    plane_z: float = 0.075,
    pickup_x_bounds: tuple[float, float] = (0.36, 0.62),
    pickup_y_bounds: tuple[float, float] = (-0.20, 0.17),
    min_part_extent_m: float = 0.018,
    max_part_extent_m: float = 0.090,
    min_area_px: float = 120.0,
    max_area_px: float = 4000.0,
) -> list[ProjectedDetection]:
    """Detect workpieces, excluding bins and robot geometry by size and workspace."""

    distance = camera_z - plane_z
    if distance <= 0.0:
        raise ValueError("the camera must be above the projection plane")

    # Mask before contour extraction.  In this scene the orange part is close
    # enough to the orange rework bin for their silhouettes to touch; filtering
    # only the final centroid would therefore see one oversized contour.
    image_height, image_width = bgr_image.shape[:2]
    u_values = [intrinsics.cx - y * intrinsics.fx / distance for y in pickup_y_bounds]
    v_values = [
        intrinsics.cy - (x - camera_x) * intrinsics.fy / distance for x in pickup_x_bounds
    ]
    u_min = max(0, int(np.floor(min(u_values))))
    u_max = min(image_width - 1, int(np.ceil(max(u_values))))
    v_min = max(0, int(np.floor(min(v_values))))
    v_max = min(image_height - 1, int(np.ceil(max(v_values))))
    if u_min > u_max or v_min > v_max:
        return []
    roi_image = np.zeros_like(bgr_image)
    roi_image[v_min : v_max + 1, u_min : u_max + 1] = bgr_image[
        v_min : v_max + 1, u_min : u_max + 1
    ]

    detections: list[ProjectedDetection] = []
    for candidate in find_color_candidates(
        roi_image,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
    ):
        width_m = candidate.width_px * distance / intrinsics.fx
        height_m = candidate.height_px * distance / intrinsics.fy
        if min(width_m, height_m) < min_part_extent_m:
            continue
        if max(width_m, height_m) > max_part_extent_m:
            continue
        x, y, z = pixel_to_base(
            candidate.u,
            candidate.v,
            intrinsics,
            camera_x=camera_x,
            camera_z=camera_z,
            plane_z=plane_z,
        )
        if not pickup_x_bounds[0] <= x <= pickup_x_bounds[1]:
            continue
        if not pickup_y_bounds[0] <= y <= pickup_y_bounds[1]:
            continue
        detections.append(
            ProjectedDetection(
                color_name=candidate.color_name,
                class_name=candidate.class_name,
                u=candidate.u,
                v=candidate.v,
                x=x,
                y=y,
                z=z,
                confidence=candidate.confidence,
            )
        )
    return detections


def stable_median_detection(
    history: Sequence[ProjectedDetection],
    *,
    required_frames: int,
    max_jitter_m: float,
) -> ProjectedDetection | None:
    """Return a median observation only when a full history is spatially stable."""

    if required_frames < 1:
        raise ValueError("required_frames must be at least one")
    if len(history) < required_frames:
        return None
    window = history[-required_frames:]
    if len({item.color_name for item in window}) != 1:
        return None

    x = float(np.median([item.x for item in window]))
    y = float(np.median([item.y for item in window]))
    if any(hypot(item.x - x, item.y - y) > max_jitter_m for item in window):
        return None
    representative = window[-1]
    return replace(
        representative,
        u=float(np.median([item.u for item in window])),
        v=float(np.median([item.v for item in window])),
        x=x,
        y=y,
        confidence=float(np.mean([item.confidence for item in window])),
    )


class MultiFrameStabilizer:
    """Small testable state holder for consecutive-frame stability gating."""

    def __init__(self, required_frames: int = 4, max_jitter_m: float = 0.012) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be at least one")
        self.required_frames = required_frames
        self.max_jitter_m = max_jitter_m
        self._history: defaultdict[str, deque[ProjectedDetection]] = defaultdict(
            lambda: deque(maxlen=required_frames)
        )

    def update(self, detections: Iterable[ProjectedDetection]) -> list[ProjectedDetection]:
        best_by_color: dict[str, ProjectedDetection] = {}
        for detection in detections:
            previous = best_by_color.get(detection.color_name)
            if previous is None or detection.confidence > previous.confidence:
                best_by_color[detection.color_name] = detection

        for color_name in tuple(self._history):
            if color_name not in best_by_color:
                self._history[color_name].clear()

        stable: list[ProjectedDetection] = []
        for color_name, detection in best_by_color.items():
            history = self._history[color_name]
            history.append(detection)
            result = stable_median_detection(
                tuple(history),
                required_frames=self.required_frames,
                max_jitter_m=self.max_jitter_m,
            )
            if result is not None:
                stable.append(result)
        return stable


class ColorPerceptionNode(Node):
    """Detect Gazebo RGB workpieces and release one stable detection per cycle."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("ROS 2 Python dependencies are unavailable") from _ROS_IMPORT_ERROR
        super().__init__("color_perception")
        self.declare_parameter("image_topic", "/sorting_cell/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/sorting_cell/camera/camera_info")
        self.declare_parameter("stable_frames", 4)
        self.declare_parameter("stability_radius_m", 0.012)
        self.declare_parameter("inter_cycle_delay_s", 0.5)
        # A completed pick returns the arm through the camera view.  Keep the
        # already-stabilized batch snapshot long enough to release later parts
        # even when the parked arm occludes them temporarily.
        self.declare_parameter("cached_detection_max_age_s", 60.0)

        stable_frames = int(self.get_parameter("stable_frames").value)
        stability_radius = float(self.get_parameter("stability_radius_m").value)
        self._stabilizer = MultiFrameStabilizer(stable_frames, stability_radius)
        self._intrinsics: CameraIntrinsics | None = None
        self._processed_colors: set[str] = set()
        self._stable_by_color: dict[str, tuple[ProjectedDetection, float]] = {}
        self._waiting_for_cycle = False
        self._active_object_id: str | None = None
        self._active_color: str | None = None
        self._active_detection: ProjectedDetection | None = None
        self._last_publish_time_s = 0.0
        self._next_publish_time_s = 0.0
        self._camera_ready_logged = False

        self._publisher = self.create_publisher(DetectedObject, "/sorting_cell/detections", 10)
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        status_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            CellStatus, "/sorting_cell/status", self._on_status, status_qos
        )
        self.create_timer(0.05, self._publish_if_ready)
        self.get_logger().info("RGB color perception ready; waiting for camera calibration")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_camera_info(self, message: CameraInfo) -> None:
        try:
            self._intrinsics = CameraIntrinsics(
                fx=float(message.k[0]),
                fy=float(message.k[4]),
                cx=float(message.k[2]),
                cy=float(message.k[5]),
            )
        except (IndexError, ValueError) as exc:
            self.get_logger().warning(f"ignoring invalid CameraInfo: {exc}")
            return
        if not self._camera_ready_logged:
            self.get_logger().info(
                f"camera calibrated: fx={self._intrinsics.fx:.1f}, fy={self._intrinsics.fy:.1f}, "
                f"cx={self._intrinsics.cx:.1f}, cy={self._intrinsics.cy:.1f}"
            )
            self._camera_ready_logged = True

    def _on_image(self, message: Image) -> None:
        if self._intrinsics is None or len(self._processed_colors) == len(COLOR_MODELS):
            return
        try:
            image = decode_ros_image(
                message.data,
                int(message.width),
                int(message.height),
                str(message.encoding),
                int(message.step),
            )
            detections = detect_projected_parts(image, self._intrinsics)
        except (ValueError, cv2.error) as exc:
            self.get_logger().warning(f"camera frame rejected: {exc}")
            return

        for detection in self._stabilizer.update(detections):
            if detection.color_name in self._processed_colors:
                continue
            previous = self._stable_by_color.get(detection.color_name)
            self._stable_by_color[detection.color_name] = (detection, self._now_s())
            if previous is None:
                self.get_logger().info(
                    f"stable {detection.color_name} workpiece at "
                    f"({detection.x:.3f}, {detection.y:.3f}, {detection.z:.3f})"
                )
        self._publish_if_ready()

    def _on_status(self, message: CellStatus) -> None:
        if not self._waiting_for_cycle or str(message.state).upper() not in ("COMPLETE", "FAILED"):
            return
        if self._active_object_id and message.active_object_id not in ("", self._active_object_id):
            return
        if self._active_color:
            self._processed_colors.add(self._active_color)
            self._stable_by_color.pop(self._active_color, None)
        self._waiting_for_cycle = False
        self._active_object_id = None
        self._active_color = None
        self._active_detection = None
        self._next_publish_time_s = self._now_s() + float(
            self.get_parameter("inter_cycle_delay_s").value
        )

    def _publish_if_ready(self) -> None:
        now_s = self._now_s()
        if self._waiting_for_cycle:
            if (
                self._active_detection is not None
                and self._publisher.get_subscription_count() > 0
                and now_s - self._last_publish_time_s >= 1.0
            ):
                self._publish_detection(self._active_detection, retry=True)
            return
        if now_s < self._next_publish_time_s:
            return
        if self._publisher.get_subscription_count() == 0:
            return
        next_color = next(
            (
                model.color_name
                for model in COLOR_MODELS
                if model.color_name not in self._processed_colors
            ),
            None,
        )
        if next_color is None:
            return
        stable = self._stable_by_color.get(next_color)
        max_age_s = max(
            0.0, float(self.get_parameter("cached_detection_max_age_s").value)
        )
        if stable is None or now_s - stable[1] > max_age_s:
            return
        detection = stable[0]
        self._waiting_for_cycle = True
        self._active_object_id = detection.object_id
        self._active_color = detection.color_name
        self._active_detection = detection
        self._publish_detection(detection, retry=False)

    def _publish_detection(self, detection: ProjectedDetection, *, retry: bool) -> None:
        message = DetectedObject()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.object_id = detection.object_id
        message.class_name = detection.class_name
        message.pose.position.x = detection.x
        message.pose.position.y = detection.y
        message.pose.position.z = detection.z
        message.pose.orientation.w = 1.0
        message.confidence = detection.confidence

        self._publisher.publish(message)
        self._last_publish_time_s = self._now_s()
        self.get_logger().info(
            f"{'republished' if retry else 'published'} {detection.object_id}: "
            f"class={detection.class_name}, "
            f"confidence={detection.confidence:.2f}"
        )


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python dependencies are unavailable") from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node = ColorPerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
