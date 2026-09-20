# Repository consolidation

`PickSort-VLA` is the public research and simulation repository for language-conditioned sorting, MuJoCo evaluation and ROS 2 workcell integration.

| Layer | Entry point | Scope |
| --- | --- | --- |
| Policy and evaluation | `src/smartpick_vla/`, `configs/`, `results/` | MuJoCo/Gymnasium data, compact VLA variants, evaluation and replay |
| ROS 2 workcell | `integrations/ros2-vision-guided-sorting-cell/` | ROS 2 Jazzy, Gazebo Harmonic, color perception, planning and headless verification |
| Visual evidence | root `results/**/media/`, `docs/assets/repository-map.svg` | Checked-in simulation figures and GIFs; not physical-camera recordings |
| Contract boundary | `docs/` and integration docs | Hardware bridge remains an explicit dry-run / authorization boundary |

## Imported source

| Repository | Frozen commit | Destination | License handling |
| --- | --- | --- | --- |
| `1830051801-ux/ros2-vision-guided-sorting-cell` | `a7c53de176abd791802da2a6b0ebfc12b077cb4b` | `integrations/ros2-vision-guided-sorting-cell/` | Original MIT `LICENSE` is preserved in the imported directory |

Run the root tests for the learning stack, then validate the import without starting ROS, Gazebo, cameras or a hardware transport:

```bash
python tools/verify_consolidation.py
python integrations/ros2-vision-guided-sorting-cell/tools/check_project.py
```

The old standalone workcell repository remains as a migration pointer so existing links remain usable. The machine-readable companion is [CONSOLIDATION_MANIFEST.json](CONSOLIDATION_MANIFEST.json).
