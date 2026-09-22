# 现场功能验证索引（2026-09-22）

小U真实机械臂已完成运动闭环，并完成桌面物品抓取、桌面整理和垃圾清理。该目录把现场运行结果与 PickSort-VLA 的仿真、训练和回放结果分开归档。

## 已完成场景

| 场景 | 结果 |
| --- | --- |
| 示教位复现 | 完成 |
| 桌面物品抓取 | 完成 |
| 桌面整理 | 完成 |
| 垃圾清理 | 完成 |

示教位复现的现场终端回传为 `9.375 s`、单段轨迹、最大关节误差约 `0.311°`。完整记录见[小U主仓库现场验证条目](https://github.com/1830051801-ux/xiaou-vision-robot-arm/blob/codex/complete-xiaou-stack/docs/evidence/field_validation_20260922/README.md)。

PickSort-VLA 的 MuJoCo、ROS 2 dry-run、协议回放和 Temporal VLA 结果继续按各自目录归档；仿真结果用于规划和回归，现场结果单独统计。

机器可读摘要见 [`field_validation.json`](field_validation.json)。
