# ROS 2 Runtime Evidence

This directory records the server-only reference run made on
2026-07-15 (Asia/Shanghai) with:

- Windows 11 + WSL 2, Ubuntu 24.04;
- ROS 2 Jazzy;
- Gazebo Sim 8.11.0 (Harmonic);
- Python 3.12.3;
- default `perception:=color` image processing.

The run was started from Windows with:

```powershell
.\tools\run_ros2_demo.ps1 `
  -SkipSyncBuild `
  -TimeoutSeconds 240 `
  -OutputDir .\results\final-ros2-runtime
```

The same wrapper without `-SkipSyncBuild` performs the Windows-to-WSL sync and
ROS workspace build first.

## Result

| Check | Result |
| --- | ---: |
| Wall time | 52.208 s |
| Camera frames received | 585 |
| Joint-state messages received | 10,587 |
| Vision detections | 3 / 3 in `base_link` |
| Unique successful cycles | 3 / 3 |
| Failed cycles | 0 |
| Gazebo bin-pose checks | 3 / 3 |

Final Gazebo model positions were:

| Part | Position `[x, y, z]` m |
| --- | --- |
| `part_blue` | `[0.30000, 0.30000, 0.08550]` |
| `part_red` | `[0.30000, -0.30000, 0.08550]` |
| `part_orange` | `[0.48000, 0.25000, 0.08549]` |

## Files

- [`ros2_runtime_verification.json`](ros2_runtime_verification.json) is the
  machine-readable run summary, including detections, controller
  states, terminal counts and named Gazebo poses.
- [`ros2_runtime_metrics.csv`](ros2_runtime_metrics.csv) contains the three
  terminal cycle records written by the metrics node.
- [`ros2_runtime_launch.log`](ros2_runtime_launch.log) contains the launch,
  camera, controller, attachment and completion trace. Its error scan is empty.

Local absolute paths in the committed launch log are replaced with descriptive
placeholders; timestamps, process output and acceptance events are unchanged.

This evidence verifies the simulated vision-to-coordinate-to-grasp/control
loop. It is not physical-robot safety validation or production throughput data.
