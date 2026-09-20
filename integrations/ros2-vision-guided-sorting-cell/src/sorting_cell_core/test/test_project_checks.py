from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECK_PROJECT_PATH = PROJECT_ROOT / "tools" / "check_project.py"
SPEC = importlib.util.spec_from_file_location("sorting_cell_project_checks", CHECK_PROJECT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


def test_committed_launch_urdf_and_sdf_contracts_pass() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "src/sorting_cell_core/config/cell.yaml").read_text(encoding="utf-8")
    )
    assert CHECKS.validate_launch_contract(
        PROJECT_ROOT / "src/sorting_cell_bringup/launch/sorting_cell.launch.py"
    ) == []
    assert CHECKS.validate_urdf_pitch_axes(
        PROJECT_ROOT / "src/sorting_cell_description/urdf/sorting_arm.urdf.xacro"
    ) == []
    assert CHECKS.validate_sdf_semantics(
        PROJECT_ROOT / "src/sorting_cell_gazebo/worlds/sorting_cell.sdf", config
    ) == []
    assert CHECKS.validate_release_metadata() == []


def test_launch_contract_rejects_an_unhandled_perception_mode(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT / "src/sorting_cell_bringup/launch/sorting_cell.launch.py"
    ).read_text(encoding="utf-8")
    changed = source.replace(
        'choices=["color", "synthetic"]',
        'choices=["color", "synthetic", "unhandled"]',
        1,
    )
    launch_path = tmp_path / "invalid.launch.py"
    launch_path.write_text(changed, encoding="utf-8")

    failures = CHECKS.validate_launch_contract(launch_path)

    assert any("restrict choices" in failure for failure in failures)


def test_pitch_axis_sign_is_checked_against_analytical_motion(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT / "src/sorting_cell_description/urdf/sorting_arm.urdf.xacro"
    ).read_text(encoding="utf-8")
    changed = source.replace('<axis xyz="0 -1 0"/>', '<axis xyz="0 1 0"/>', 1)
    urdf_path = tmp_path / "wrong_axis.urdf.xacro"
    urdf_path.write_text(changed, encoding="utf-8")

    failures = CHECKS.validate_urdf_pitch_axes(urdf_path)

    assert any("shoulder_joint axis" in failure for failure in failures)


def test_sdf_semantics_require_attachment_and_physics_plugins(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT / "src/sorting_cell_gazebo/worlds/sorting_cell.sdf"
    ).read_text(encoding="utf-8")
    changed = source.replace("gz-sim-user-commands-system", "missing-user-commands", 1)
    world_path = tmp_path / "missing_plugin.sdf"
    world_path.write_text(changed, encoding="utf-8")
    config = yaml.safe_load(
        (PROJECT_ROOT / "src/sorting_cell_core/config/cell.yaml").read_text(encoding="utf-8")
    )

    failures = CHECKS.validate_sdf_semantics(world_path, config)

    assert any("gz-sim-user-commands-system" in failure for failure in failures)


def test_sdf_semantics_require_real_bin_collision_walls(tmp_path: Path) -> None:
    tree = ET.parse(PROJECT_ROOT / "src/sorting_cell_gazebo/worlds/sorting_cell.sdf")
    accepted_link = tree.find("./world/model[@name='accepted_bin']/link")
    assert accepted_link is not None
    front_wall = accepted_link.find("collision[@name='front_collision']")
    assert front_wall is not None
    accepted_link.remove(front_wall)
    world_path = tmp_path / "open_bin.sdf"
    tree.write(world_path, encoding="unicode")
    config = yaml.safe_load(
        (PROJECT_ROOT / "src/sorting_cell_core/config/cell.yaml").read_text(encoding="utf-8")
    )

    failures = CHECKS.validate_sdf_semantics(world_path, config)

    assert any("base and four collision walls" in failure for failure in failures)


def test_external_sdformat_checker_hook_receives_the_world_path(
    tmp_path: Path, monkeypatch
) -> None:
    checker = tmp_path / "sdf checker.py"
    checker.write_text(
        "import pathlib, sys\n"
        "raise SystemExit(0 if pathlib.Path(sys.argv[1]).suffix == '.sdf' else 2)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SORTING_CELL_SDF_CHECKER",
        f'"{sys.executable}" "{checker}" {{sdf}}',
    )

    failures = CHECKS.validate_with_sdformat(
        PROJECT_ROOT / "src/sorting_cell_gazebo/worlds/sorting_cell.sdf"
    )

    assert failures == []


def test_wsl_setup_executes_as_the_selected_linux_user() -> None:
    source = (PROJECT_ROOT / "tools/setup_wsl_after_reboot.ps1").read_text(
        encoding="utf-8"
    )

    assert "wsl.exe -d $Distro -u $LinuxUser -- bash -lc" in source
    assert 'Write-Host "Launch: wsl -d $Distro -u $LinuxUser' in source


def test_action_timeout_blocks_new_goals_until_controller_is_terminal() -> None:
    source = (
        PROJECT_ROOT
        / "src/sorting_cell_core/sorting_cell_core/nodes/pick_coordinator_node.py"
    ).read_text(encoding="utf-8")

    assert 'self.phase = "cancelling_goal"' in source
    assert 'self.phase not in {"executing_goal", "cancelling_goal"}' in source
    assert "self._fail_active(f\"{reason}; {status_detail}\")" in source
    assert '"goal ownership is unknown"' in source
    assert "terminal state is unknown: {error}" in source
    assert 'self.phase = "faulted"' in source
    assert "self.queue.clear()" in source
