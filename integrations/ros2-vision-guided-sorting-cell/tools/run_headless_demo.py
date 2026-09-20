from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = PROJECT_ROOT / "src" / "sorting_cell_core"
sys.path.insert(0, str(CORE_SOURCE))

from sorting_cell_core.headless import HeadlessCell, serialize_frames, write_results
from sorting_cell_core.visualization import render_animation, render_dashboard
from sorting_cell_core.workflow import WorkflowPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sorting workcell without a ROS 2 installation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for metrics and visual artifacts.",
    )
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = CORE_SOURCE / "config" / "cell.yaml"
    frames, reports, summary = HeadlessCell(WorkflowPlanner.from_yaml(config_path)).run()
    write_results(args.output, reports, summary)
    serialize_frames(frames, args.output / "sampled_frames.json")
    render_dashboard(frames, reports, summary, args.output / "headless_dashboard.png")
    if not args.no_animation:
        render_animation(frames, args.output / "sorting_cell_demo.gif")

    print("ROS 2 vision-guided sorting workcell - headless verification")
    print(f"  cycles:       {summary.completed}/{summary.total}")
    print(f"  success rate: {summary.success_rate * 100:.1f}%")
    print(f"  average:      {summary.average_cycle_s:.2f} s")
    print(f"  runtime:      {summary.total_runtime_s:.2f} s")
    for report in reports:
        print(
            f"  {report.object_id:<12} {report.class_name:<9} -> "
            f"{report.destination:<12} {report.duration_s:.2f} s"
        )
    print(f"  artifacts:    {args.output.resolve()}")
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
