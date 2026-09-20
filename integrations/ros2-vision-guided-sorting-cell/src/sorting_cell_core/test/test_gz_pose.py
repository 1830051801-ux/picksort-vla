from sorting_cell_core.gz_pose import parse_gz_pose_vector


def test_parse_gz_pose_vector_preserves_names_and_default_zero_fields() -> None:
    payload = """
header { stamp { sec: 12 } }
pose {
  name: "part_blue"
  id: 51
  position {
    x: 0.301
    y: 0.299
    z: 8.0e-2
  }
  orientation { w: 1 }
}
pose {
  name: "part_red"
  id: 55
  position {
    x: 0.3
    z: 0.081
  }
  orientation { w: 1 }
}
pose {
  name: "unrelated"
  position { x: 9 }
}
"""

    positions = parse_gz_pose_vector(payload, {"part_blue", "part_red"})

    assert positions == {
        "part_blue": (0.301, 0.299, 0.08),
        "part_red": (0.3, 0.0, 0.081),
    }
