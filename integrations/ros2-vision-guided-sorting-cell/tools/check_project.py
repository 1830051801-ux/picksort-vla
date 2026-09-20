from __future__ import annotations

import ast
import compileall
import os
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from math import sqrt
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = PROJECT_ROOT / "src" / "sorting_cell_core"
REPOSITORY_URL = "https://github.com/1830051801-ux/ros2-vision-guided-sorting-cell"
RELEASE_VERSION = "1.0.0"
if str(CORE_SOURCE) not in sys.path:
    sys.path.insert(0, str(CORE_SOURCE))

from sorting_cell_core.kinematics import ArmKinematics  # noqa: E402


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _parse_vector(text: str | None, length: int) -> tuple[float, ...] | None:
    if text is None:
        return None
    try:
        values = tuple(float(value) for value in text.split())
    except ValueError:
        return None
    return values if len(values) == length else None


def validate_urdf_pitch_axes(path: Path) -> list[str]:
    """Ensure URDF right-hand rotations match positive analytical pitch."""

    failures: list[str] = []
    root = ET.parse(path).getroot()
    joints = {joint.get("name", ""): joint for joint in root.findall("joint")}
    model = ArmKinematics()
    zero = (0.0, 0.0, 0.0, 0.0, 0.0)
    base_z = model.forward(zero)[0].z

    for index, joint_name in enumerate(
        ("shoulder_joint", "elbow_joint", "wrist_pitch_joint"), start=1
    ):
        perturbed = list(zero)
        perturbed[index] = 1e-4
        analytical_delta_z = model.forward(perturbed)[0].z - base_z
        expected_y_sign = -1.0 if analytical_delta_z > 0.0 else 1.0

        joint = joints.get(joint_name)
        axis = _parse_vector(
            joint.find("axis").get("xyz")
            if joint is not None and joint.find("axis") is not None
            else None,
            3,
        )
        if axis is None:
            failures.append(f"{_label(path)}: {joint_name} has no valid xyz axis")
            continue
        norm = sqrt(sum(component**2 for component in axis))
        normalized = tuple(component / norm for component in axis) if norm else axis
        expected = (0.0, expected_y_sign, 0.0)
        if any(abs(actual - wanted) > 1e-6 for actual, wanted in zip(normalized, expected)):
            failures.append(
                f"{_label(path)}: {joint_name} axis {axis} disagrees with analytical "
                f"positive-pitch direction; expected {expected}"
            )
    return failures


def validate_sdf_semantics(path: Path, config: dict) -> list[str]:
    """Check the world contracts required by perception and attachment."""

    failures: list[str] = []
    root = ET.parse(path).getroot()
    worlds = root.findall("world")
    if len(worlds) != 1 or worlds[0].get("name") != "sorting_cell":
        return [f"{_label(path)}: expected exactly one world named 'sorting_cell'"]
    world = worlds[0]

    plugin_files = {plugin.get("filename", "") for plugin in world.findall("plugin")}
    required_plugins = {
        "gz-sim-physics-system",
        "gz-sim-user-commands-system",
        "gz-sim-scene-broadcaster-system",
        "gz-sim-sensors-system",
    }
    missing_plugins = sorted(required_plugins - plugin_files)
    if missing_plugins:
        failures.append(
            f"{_label(path)}: missing required Gazebo system plugins: "
            + ", ".join(missing_plugins)
        )

    models = world.findall("model")
    model_names = [model.get("name", "") for model in models]
    duplicates = sorted({name for name in model_names if model_names.count(name) > 1})
    if duplicates:
        failures.append(f"{_label(path)}: duplicate model names: {', '.join(duplicates)}")
    model_by_name = {model.get("name", ""): model for model in models}

    destinations = config.get("cell", {}).get("destinations", {})
    if not isinstance(destinations, dict) or not destinations:
        failures.append(
            "src/sorting_cell_core/config/cell.yaml: no cell destinations configured"
        )
        destinations = {}
    for name, target in destinations.items():
        model = model_by_name.get(name)
        if model is None:
            failures.append(f"{_label(path)}: configured destination model {name!r} is missing")
            continue
        if model.findtext("static", default="false").strip().lower() != "true":
            failures.append(f"{_label(path)}: destination {name!r} must be static")
        collisions = model.findall("./link/collision")
        if len(collisions) < 5:
            failures.append(
                f"{_label(path)}: destination {name!r} needs a base and four collision walls"
            )
        pose = _parse_vector(model.findtext("pose"), 6)
        try:
            target_xy = (
                (float(target[0]), float(target[1]))
                if isinstance(target, list) and len(target) == 3
                else None
            )
        except (TypeError, ValueError):
            target_xy = None
        if (
            pose is None
            or target_xy is None
            or abs(pose[0] - target_xy[0]) > 0.02
            or abs(pose[1] - target_xy[1]) > 0.02
        ):
            failures.append(
                f"{_label(path)}: destination {name!r} pose does not match cell.yaml XY"
            )

    for part_name in ("part_blue", "part_red", "part_orange"):
        part = model_by_name.get(part_name)
        if part is None:
            failures.append(f"{_label(path)}: required workpiece {part_name!r} is missing")
            continue
        if part.findtext("static", default="false").strip().lower() == "true":
            failures.append(f"{_label(path)}: workpiece {part_name!r} must be dynamic")
        if part.find("./link/collision") is None:
            failures.append(f"{_label(path)}: workpiece {part_name!r} has no collision")
        try:
            mass = float(part.findtext("./link/inertial/mass", default="0"))
        except ValueError:
            mass = 0.0
        if mass <= 0.0:
            failures.append(f"{_label(path)}: workpiece {part_name!r} needs positive mass")

    camera_model = model_by_name.get("overhead_camera")
    sensor = camera_model.find("./link/sensor") if camera_model is not None else None
    if camera_model is None or sensor is None or sensor.get("type") != "camera":
        failures.append(f"{_label(path)}: overhead_camera camera sensor is missing")
    else:
        expected_values = {
            "topic": "/sorting_cell/camera/image_raw",
            "camera/camera_info_topic": "/sorting_cell/camera/camera_info",
        }
        for field, expected in expected_values.items():
            if sensor.findtext(field, default="").strip() != expected:
                failures.append(
                    f"{_label(path)}: camera {field} must be {expected!r}"
                )
        for field in (
            "update_rate",
            "camera/horizontal_fov",
            "camera/image/width",
            "camera/image/height",
        ):
            try:
                value = float(sensor.findtext(field, default="0"))
            except ValueError:
                value = 0.0
            if value <= 0.0:
                failures.append(f"{_label(path)}: camera {field} must be positive")
    return failures


def validate_with_sdformat(path: Path) -> list[str]:
    """Run sdformat when available; allow CI to inject an explicit checker."""

    configured = os.environ.get("SORTING_CELL_SDF_CHECKER", "").strip()
    if configured:
        command = shlex.split(configured, posix=os.name != "nt")
        if os.name == "nt":
            command = [
                argument[1:-1]
                if len(argument) >= 2 and argument[0] == argument[-1] == '"'
                else argument
                for argument in command
            ]
        if not command:
            return [f"{_label(path)}: SORTING_CELL_SDF_CHECKER is empty"]
        replaced = [argument.replace("{sdf}", str(path)) for argument in command]
        command = (
            replaced
            if any("{sdf}" in argument for argument in command)
            else [*replaced, str(path)]
        )
    elif shutil.which("gz"):
        command = ["gz", "sdf", "-k", str(path)]
    elif shutil.which("ign"):
        command = ["ign", "sdf", "-k", str(path)]
    else:
        return []

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"{_label(path)}: sdformat validation could not run: {error}"]
    if result.returncode == 0:
        return []
    detail = (result.stderr or result.stdout).strip().replace("\n", " | ")
    return [
        f"{_label(path)}: sdformat validation failed with exit code "
        f"{result.returncode}: {detail or 'no diagnostic output'}"
    ]


def validate_launch_contract(path: Path) -> list[str]:
    """Keep invalid modes from silently launching without perception."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    perception_choices: list[str] | None = None
    create_arguments: list[str] | None = None
    startup_order_valid = False
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        function_name = getattr(call.func, "id", getattr(call.func, "attr", ""))
        if function_name == "DeclareLaunchArgument" and call.args:
            try:
                argument_name = ast.literal_eval(call.args[0])
            except (ValueError, TypeError):
                continue
            if argument_name == "perception":
                for keyword in call.keywords:
                    if keyword.arg == "choices":
                        try:
                            perception_choices = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            perception_choices = None
        if function_name == "Node":
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            try:
                package = ast.literal_eval(keywords.get("package"))
                executable = ast.literal_eval(keywords.get("executable"))
            except (ValueError, TypeError):
                continue
            if package == "ros_gz_sim" and executable == "create":
                try:
                    create_arguments = ast.literal_eval(keywords.get("arguments"))
                except (ValueError, TypeError):
                    create_arguments = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        if getattr(node.value.func, "id", "") != "LaunchDescription" or not node.value.args:
            continue
        entities = node.value.args[0]
        if not isinstance(entities, (ast.List, ast.Tuple)):
            continue
        handler_indices = [
            index
            for index, entity in enumerate(entities.elts)
            if isinstance(entity, ast.Call)
            and getattr(entity.func, "id", "") == "RegisterEventHandler"
        ]
        process_indices = [
            index
            for index, entity in enumerate(entities.elts)
            if isinstance(entity, ast.Name)
            and entity.id
            in {"gazebo", "state_publisher", "clock_and_camera_bridge", "rviz_node"}
        ]
        startup_order_valid = bool(
            handler_indices
            and process_indices
            and max(handler_indices) < min(process_indices)
        )
        break

    if perception_choices != ["color", "synthetic"]:
        return [
            f"{_label(path)}: perception launch argument must restrict choices to "
            "['color', 'synthetic']"
        ]
    if create_arguments is None:
        return [f"{_label(path)}: could not inspect ros_gz_sim create arguments"]
    try:
        world_index = create_arguments.index("-world")
    except ValueError:
        world_index = -1
    if world_index < 0 or create_arguments[world_index + 1 : world_index + 2] != [
        "sorting_cell"
    ]:
        return [f"{_label(path)}: robot spawn must explicitly target world 'sorting_cell'"]
    if not startup_order_valid:
        return [
            f"{_label(path)}: process-exit handlers must be registered before "
            "starting Gazebo and critical ROS nodes"
        ]
    return []


def validate_release_metadata() -> list[str]:
    """Keep public package and repository identity complete and consistent."""

    failures: list[str] = []
    for relative_path in ("README.md", "LICENSE", "CHANGELOG.md", "CITATION.cff"):
        if not (PROJECT_ROOT / relative_path).is_file():
            failures.append(f"missing release file: {relative_path}")

    for package_xml in (PROJECT_ROOT / "src").glob("*/package.xml"):
        root = ET.parse(package_xml).getroot()
        label = _label(package_xml)
        if root.findtext("version", default="").strip() != RELEASE_VERSION:
            failures.append(f"{label}: version must be {RELEASE_VERSION}")
        if root.findtext("license", default="").strip() != "MIT":
            failures.append(f"{label}: license must be MIT")
        repository_url = root.find("url[@type='repository']")
        if repository_url is None or (repository_url.text or "").strip() != REPOSITORY_URL:
            failures.append(f"{label}: repository URL must be {REPOSITORY_URL}")
        maintainer = root.find("maintainer")
        email = maintainer.get("email", "").strip() if maintainer is not None else ""
        if not email or "example" in email.lower() or "invalid" in email.lower():
            failures.append(f"{label}: maintainer email is missing or placeholder text")

    citation_path = PROJECT_ROOT / "CITATION.cff"
    if citation_path.is_file():
        try:
            citation = yaml.safe_load(citation_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            failures.append(f"CITATION.cff: {error}")
        else:
            expected_citation = {
                "repository-code": REPOSITORY_URL,
                "version": RELEASE_VERSION,
                "license": "MIT",
            }
            for key, expected in expected_citation.items():
                if str(citation.get(key, "")).strip() != expected:
                    failures.append(f"CITATION.cff: {key} must be {expected!r}")

    setup_path = PROJECT_ROOT / "src" / "sorting_cell_core" / "setup.py"
    try:
        setup_tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    except (OSError, SyntaxError) as error:
        failures.append(f"{_label(setup_path)}: metadata could not be parsed: {error}")
    else:
        setup_call = next(
            (
                node
                for node in ast.walk(setup_tree)
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup"
            ),
            None,
        )
        if setup_call is None:
            failures.append(f"{_label(setup_path)}: setup() call is missing")
        else:
            keywords = {keyword.arg: keyword.value for keyword in setup_call.keywords}
            for key, expected in {"version": RELEASE_VERSION, "url": REPOSITORY_URL}.items():
                try:
                    actual = ast.literal_eval(keywords.get(key))
                except (ValueError, TypeError):
                    actual = None
                if actual != expected:
                    failures.append(f"{_label(setup_path)}: {key} must be {expected!r}")
            try:
                project_urls = ast.literal_eval(keywords.get("project_urls"))
            except (ValueError, TypeError):
                project_urls = {}
            if not isinstance(project_urls, dict) or project_urls.get("Source") != REPOSITORY_URL:
                failures.append(
                    f"{_label(setup_path)}: project_urls['Source'] must be {REPOSITORY_URL!r}"
                )

    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        if REPOSITORY_URL not in readme:
            failures.append("README.md: repository identity is missing")
        if "Simulation only" not in readme and "simulation workflow" not in readme:
            failures.append("README.md: simulation-only scope must be stated plainly")

    return failures


def main() -> int:
    failures: list[str] = []
    if not compileall.compile_dir(PROJECT_ROOT / "src", quiet=1):
        failures.append("Python compilation failed")

    for pattern in ("*.xml", "*.xacro", "*.sdf"):
        for path in PROJECT_ROOT.rglob(pattern):
            try:
                ET.parse(path)
            except ET.ParseError as error:
                failures.append(f"{path.relative_to(PROJECT_ROOT)}: {error}")
    for pattern in ("*.yaml", "*.yml"):
        for path in PROJECT_ROOT.rglob(pattern):
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as error:
                failures.append(f"{path.relative_to(PROJECT_ROOT)}: {error}")

    config_path = PROJECT_ROOT / "src" / "sorting_cell_core" / "config" / "cell.yaml"
    world_path = (
        PROJECT_ROOT / "src" / "sorting_cell_gazebo" / "worlds" / "sorting_cell.sdf"
    )
    urdf_path = (
        PROJECT_ROOT
        / "src"
        / "sorting_cell_description"
        / "urdf"
        / "sorting_arm.urdf.xacro"
    )
    launch_path = (
        PROJECT_ROOT
        / "src"
        / "sorting_cell_bringup"
        / "launch"
        / "sorting_cell.launch.py"
    )
    try:
        cell_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        failures.extend(validate_sdf_semantics(world_path, cell_config))
        failures.extend(validate_with_sdformat(world_path))
    except (OSError, ET.ParseError, yaml.YAMLError) as error:
        failures.append(f"{_label(world_path)}: semantic validation failed: {error}")
    try:
        failures.extend(validate_urdf_pitch_axes(urdf_path))
    except (OSError, ET.ParseError) as error:
        failures.append(f"{_label(urdf_path)}: kinematic validation failed: {error}")
    try:
        failures.extend(validate_launch_contract(launch_path))
    except (OSError, SyntaxError) as error:
        failures.append(f"{_label(launch_path)}: launch validation failed: {error}")
    failures.extend(validate_release_metadata())

    for package_xml in (PROJECT_ROOT / "src").glob("*/package.xml"):
        package_root = ET.parse(package_xml).getroot()
        name = package_root.findtext("name", default="").strip()
        if name != package_xml.parent.name:
            failures.append(
                f"{package_xml.relative_to(PROJECT_ROOT)}: package name {name!r} "
                f"does not match directory {package_xml.parent.name!r}"
            )

    stale_names = []
    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "tools", PROJECT_ROOT / "docs"):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".xml", ".xacro", ".sdf", ".yaml", ".yml", ".md", ".sh", ".ps1"}:
                if path.resolve() == Path(__file__).resolve():
                    continue
                if "smartpick" in path.read_text(encoding="utf-8").lower():
                    stale_names.append(str(path.relative_to(PROJECT_ROOT)))
    if stale_names:
        failures.append("stale SmartPick references: " + ", ".join(stale_names))

    for asset in ("docs/assets/headless_dashboard.png", "docs/assets/sorting_cell_demo.gif"):
        if not (PROJECT_ROOT / asset).is_file():
            failures.append(f"missing README asset: {asset}")

    if failures:
        print("Project checks failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "Project checks passed: source, launch contracts, analytical/URDF kinematics, "
        "SDF semantics, package/release identities, configuration and README assets are valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
