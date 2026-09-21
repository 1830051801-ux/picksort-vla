# 2026-09-22 六轴辅助监督回放

本目录保存本轮固定种子回放的机器可读摘要。运行由 `configs/eval/temporal_vla_upgrade_long.yaml` 定义，未连接真实硬件。

- `temporal_vla_mission_six_axis_aux_eval/summary.json`：按套件汇总；
- `temporal_vla_mission_six_axis_aux_eval/eval_episodes.csv`：逐回合记录；
- 训练曲线与训练 manifest 位于 `checkpoints/upgrade_20260922/temporal_vla_mission_six_axis_aux/`。

生成的 NPZ 数据和权重默认留在本地工作区，不作为源代码提交。需要复现时，先按报告中的命令生成数据，再运行训练与评测配置。
