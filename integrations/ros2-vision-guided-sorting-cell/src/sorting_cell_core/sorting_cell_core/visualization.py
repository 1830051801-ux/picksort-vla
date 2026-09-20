from __future__ import annotations

from pathlib import Path

from .kinematics import ArmKinematics
from .models import CellFrame, CycleReport, RunSummary
from .workflow import CellLayout


OBJECT_COLORS = {
    "part_blue": "#168AAD",
    "part_red": "#D1495B",
    "part_orange": "#F28E2B",
}


def render_dashboard(
    frames: list[CellFrame],
    reports: list[CycleReport],
    summary: RunSummary,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))

    ax_path = fig.add_subplot(grid[:, 0])
    xs = [frame.tool_position.x for frame in frames]
    ys = [frame.tool_position.y for frame in frames]
    ax_path.plot(xs, ys, color="#274C77", linewidth=1.6, alpha=0.85, label="tool path")
    layout = CellLayout.default()
    for name, pose in layout.destinations.items():
        ax_path.scatter(pose.x, pose.y, marker="s", s=180, label=name)
        ax_path.text(pose.x + 0.012, pose.y + 0.012, name, fontsize=9)
    first = frames[0].object_positions if frames else {}
    for object_id, pose in first.items():
        ax_path.scatter(pose.x, pose.y, s=90, color=OBJECT_COLORS.get(object_id, "#666666"))
        ax_path.text(pose.x + 0.012, pose.y - 0.025, object_id, fontsize=9)
    ax_path.set_title("Tool trajectory and workcell layout")
    ax_path.set_xlabel("x / m")
    ax_path.set_ylabel("y / m")
    ax_path.set_xlim(-0.05, 0.75)
    ax_path.set_ylim(-0.43, 0.43)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.25)

    ax_z = fig.add_subplot(grid[0, 1])
    ax_z.plot([frame.time_s for frame in frames], [frame.tool_position.z for frame in frames], color="#2A9D8F")
    ax_z.axhline(0.055, color="#D1495B", linestyle="--", linewidth=1.0, label="table clearance")
    ax_z.set_title("End-effector height")
    ax_z.set_xlabel("time / s")
    ax_z.set_ylabel("z / m")
    ax_z.grid(True, alpha=0.25)
    ax_z.legend(loc="upper right", fontsize=8)

    ax_metrics = fig.add_subplot(grid[1, 1])
    ax_metrics.axis("off")
    lines = [
        "RUN SUMMARY",
        f"Completed cycles     {summary.completed}/{summary.total}",
        f"Success rate        {summary.success_rate * 100:.1f}%",
        f"Average cycle       {summary.average_cycle_s:.2f} s",
        f"Total runtime       {summary.total_runtime_s:.2f} s",
        "",
        "CLASS ROUTING",
    ]
    lines.extend(f"{report.class_name:<12} -> {report.destination}" for report in reports)
    ax_metrics.text(
        0.03,
        0.96,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=11,
        color="#1D2D35",
        bbox={"facecolor": "#F5F7F8", "edgecolor": "#B8C4CC", "boxstyle": "round,pad=0.7"},
    )
    fig.suptitle("ROS 2 Vision-Guided Robot Sorting Workcell - Headless Verification", fontsize=15, weight="bold")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_animation(frames: list[CellFrame], output_path: Path, max_frames: int = 140) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not frames:
        raise ValueError("cannot animate an empty frame list")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, len(frames) // max_frames)
    selected = frames[::stride]
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])

    kinematics = ArmKinematics()
    layout = CellLayout.default()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame: CellFrame) -> None:
        ax.clear()
        ax.set_xlim(-0.05, 0.78)
        ax.set_ylim(-0.44, 0.44)
        ax.set_zlim(0.0, 0.70)
        ax.set_box_aspect((0.83, 0.88, 0.70))
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_zlabel("z / m")
        ax.set_title(f"{frame.state.value} | {frame.active_object_id} | t={frame.time_s:.1f}s")

        ax.plot_surface(
            np.array([[-0.05, 0.78], [-0.05, 0.78]]),
            np.array([[-0.42, -0.42], [0.42, 0.42]]),
            np.array([[0.04, 0.04], [0.04, 0.04]]),
            color="#AAB7B8",
            alpha=0.28,
            linewidth=0,
        )
        arm = kinematics.joint_positions(frame.joints)
        ax.plot(
            [point.x for point in arm],
            [point.y for point in arm],
            [point.z for point in arm],
            color="#274C77",
            linewidth=6,
            marker="o",
            markersize=6,
        )
        ax.scatter(frame.tool_position.x, frame.tool_position.y, frame.tool_position.z, color="#111111", s=45)

        for name, pose in layout.destinations.items():
            ax.scatter(pose.x, pose.y, pose.z, marker="s", s=220, alpha=0.35)
            ax.text(pose.x, pose.y, pose.z + 0.035, name.replace("_bin", ""), fontsize=8)
        for object_id, pose in frame.object_positions.items():
            ax.scatter(
                pose.x,
                pose.y,
                pose.z,
                marker="s",
                s=90,
                color=OBJECT_COLORS.get(object_id, "#555555"),
                edgecolor="white",
                linewidth=0.6,
            )

    animation = FuncAnimation(fig, draw, frames=selected, interval=70, repeat=True)
    animation.save(output_path, writer=PillowWriter(fps=14))
    plt.close(fig)
