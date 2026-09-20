# PickSort-VLA 成果图册

[项目入口](../README.md) · [仓库归并说明](CONSOLIDATION.md) · [成果下载](https://github.com/1830051801-ux/picksort-vla/releases/tag/consolidation-20260920)

## 六轴语言条件分拣

![六轴三物体仿真](../results/upgrade_20260815/mission_six_axis_showcase/media/ik_expert/id-seed-9600.gif)

保留原有 MuJoCo/Gymnasium、紧凑 VLA、动作块、时序输入、域随机化和回放工具。GIF 来自仓库已有的仿真专家演示；不是本轮新训练出来的策略，也不是真机视频。

## ROS 2 工作单元

![本轮工作单元运动学演示](evidence/consolidation_20260920/sorting_cell_demo.gif)

本轮在归并后的目录重新运行纯 Python 运动学演示：3 个物体分别进入合格、拒收和返工路线，完成 3/3；总仿真时间 37.60 s，平均单件 12.27 s。本轮没有启动 Gazebo 或 ROS 2 进程。

![本轮轨迹、高度和周期统计](evidence/consolidation_20260920/headless_dashboard.png)

数据：[运行摘要](evidence/consolidation_20260920/run_summary.json)、[逐件结果](evidence/consolidation_20260920/cycles.csv)、[轨迹采样](evidence/consolidation_20260920/sampled_frames.json)。已有的 ROS 2/Gazebo 运行记录仍在 [导入工程的历史证据目录](../integrations/ros2-vision-guided-sorting-cell/docs/evidence/README.md)。

## 本轮验证

| 范围 | 结果 |
| --- | --- |
| VLA 平台安全测试 | 57 通过；19 项慢测试或 MuJoCo 标记测试未在该命令运行 |
| 导入的工作单元测试 | 27 通过 |
| 发布结构与导入完整性 | 通过 |
| Ruff 检查与格式 | 通过 |
| 新的运动学演示 | 3/3 路由通过，PNG/GIF/JSON/CSV 已归档 |

配套机械 CAD、Pi/F407、六轴数字孪生和策略量化资产归入 [小U主仓库](https://github.com/1830051801-ux/xiaou-vision-robot-arm)。两个项目通过文档和接口互相引用，保留各自清晰的运行入口。
