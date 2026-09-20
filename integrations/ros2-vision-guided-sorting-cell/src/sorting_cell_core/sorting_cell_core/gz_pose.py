from __future__ import annotations

import re
from collections.abc import Iterable


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_gz_pose_vector(
    payload: str, expected_names: Iterable[str] | None = None
) -> dict[str, tuple[float, float, float]]:
    """Extract named positions from ``gz topic``'s Pose_V text output.

    Gazebo omits scalar fields whose value is zero, so each coordinate defaults
    to 0.0 when it is absent from an otherwise valid position block.
    """

    wanted = set(expected_names) if expected_names is not None else None
    blocks: list[str] = []
    current: list[str] = []
    depth = 0
    for line in payload.splitlines():
        if depth == 0:
            if line.strip() != "pose {":
                continue
            current = [line]
            depth = 1
            continue
        current.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            blocks.append("\n".join(current))
            current = []

    positions: dict[str, tuple[float, float, float]] = {}
    for block in blocks:
        name_match = re.search(r'\bname\s*:\s*"([^"]+)"', block)
        position_match = re.search(r"\bposition\s*\{(.*?)\}", block, re.DOTALL)
        if name_match is None or position_match is None:
            continue
        name = name_match.group(1)
        if wanted is not None and name not in wanted:
            continue
        position_body = position_match.group(1)
        coordinates: list[float] = []
        for axis in ("x", "y", "z"):
            match = re.search(
                rf"\b{axis}\s*:\s*({_FLOAT_PATTERN})", position_body
            )
            coordinates.append(float(match.group(1)) if match is not None else 0.0)
        positions[name] = tuple(coordinates)  # type: ignore[assignment]
    return positions
