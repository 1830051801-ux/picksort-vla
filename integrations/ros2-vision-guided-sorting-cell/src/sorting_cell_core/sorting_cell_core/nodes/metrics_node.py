from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from sorting_cell_interfaces.msg import CellStatus


class MetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("cell_metrics")
        self.declare_parameter("output_csv", "")
        configured_path = str(self.get_parameter("output_csv").value).strip()
        if configured_path:
            self.output_path = Path(configured_path).expanduser()
        else:
            run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.output_path = Path("/tmp") / f"sorting_cell_metrics_{run_stamp}.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.seen_cycles: set[tuple[str, str, int, int]] = set()
        self.create_subscription(CellStatus, "/sorting_cell/status", self._on_status, 20)
        self.get_logger().info(f"writing cycle metrics to {self.output_path}")

    def _on_status(self, message: CellStatus) -> None:
        if message.state not in ("COMPLETE", "FAILED"):
            return
        key = (
            message.active_object_id,
            message.state,
            int(message.completed_cycles),
            int(message.failed_cycles),
        )
        if key in self.seen_cycles:
            return
        self.seen_cycles.add(key)
        exists = self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(
                    ["timestamp_ns", "object_id", "state", "cycle_time_s", "completed", "failed", "message"]
                )
            writer.writerow(
                [
                    self.get_clock().now().nanoseconds,
                    message.active_object_id,
                    message.state,
                    f"{message.current_cycle_time_s:.3f}",
                    message.completed_cycles,
                    message.failed_cycles,
                    message.message,
                ]
            )
        self.get_logger().info(
            f"metrics: complete={message.completed_cycles}, failed={message.failed_cycles}, "
            f"last_cycle={message.current_cycle_time_s:.2f}s"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MetricsNode()
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
