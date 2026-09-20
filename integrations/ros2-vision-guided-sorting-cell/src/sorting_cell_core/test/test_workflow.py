from pathlib import Path

from sorting_cell_core.headless import DEFAULT_PARTS, HeadlessCell
from sorting_cell_core.models import CycleState
from sorting_cell_core.workflow import WorkflowPlanner


def test_single_cycle_contains_complete_industrial_sequence() -> None:
    plan = WorkflowPlanner().plan_cycle(DEFAULT_PARTS[0].detection())
    assert plan.report is not None
    assert plan.report.success
    states = {frame.state for frame in plan.frames}
    assert {
        CycleState.PREGRASP,
        CycleState.APPROACH,
        CycleState.GRASP,
        CycleState.LIFT,
        CycleState.TRANSFER,
        CycleState.PLACE,
        CycleState.RELEASE,
        CycleState.HOME,
        CycleState.COMPLETE,
    }.issubset(states)
    assert plan.frames[-1].object_positions["part_blue"].y > 0.25


def test_headless_cell_sorts_all_default_parts() -> None:
    frames, reports, summary = HeadlessCell().run()
    assert frames
    assert len(reports) == 3
    assert summary.completed == 3
    assert summary.failed == 0
    assert summary.success_rate == 1.0
    assert {report.destination for report in reports} == {
        "accepted_bin",
        "reject_bin",
        "rework_bin",
    }


def test_versioned_cell_configuration_drives_the_planner() -> None:
    config = Path(__file__).parents[1] / "config" / "cell.yaml"
    planner = WorkflowPlanner.from_yaml(config)
    assert planner.open_gripper == 0.032
    assert planner.closed_gripper == 0.004
    assert planner.layout.destinations["rework_bin"].x == 0.48
    assert planner.safety.limits.minimum_confidence == 0.70
