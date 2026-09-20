#!/usr/bin/env python3
"""Verify that the archived ROS 2 workcell import is complete and source-only.

The check is intentionally offline. It does not start ROS 2, Gazebo, a camera,
or any hardware transport.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "docs/CONSOLIDATION.md",
    "docs/CONSOLIDATION_MANIFEST.json",
    "docs/assets/repository-map.svg",
    "integrations/ros2-vision-guided-sorting-cell/UPSTREAM.md",
    "integrations/ros2-vision-guided-sorting-cell/LICENSE",
    "integrations/ros2-vision-guided-sorting-cell/docs/assets/sorting_cell_demo.gif",
    "integrations/ros2-vision-guided-sorting-cell/tools/check_project.py",
    "integrations/ros2-vision-guided-sorting-cell/src/sorting_cell_core/sorting_cell_core/workflow.py",
)


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    manifest_path = ROOT / "docs/CONSOLIDATION_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        missing.append(f"unreadable manifest: {exc}")
        manifest = {}
    imports = manifest.get("imports", []) if isinstance(manifest, dict) else []
    imported = any(
        isinstance(item, dict)
        and item.get("destination") == "integrations/ros2-vision-guided-sorting-cell"
        and item.get("commit") == "a7c53de176abd791802da2a6b0ebfc12b077cb4b"
        for item in imports
    )
    if not imported:
        missing.append("manifest does not describe the pinned ROS 2 import")
    report = {
        "passed": not missing,
        "required_paths": len(REQUIRED),
        "missing": missing,
        "hardware_transport_opened": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
