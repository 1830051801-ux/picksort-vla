# 离线验证记录（2026-09-22）

本记录对应当前 `codex/consolidate-ros2-sorting-cell` 分支。验证在 Windows 本机的 MuJoCo 与 CUDA 环境完成；没有打开树莓派、串口、CAN、相机或真实执行接口。

## 1. 数据生成

配置：`configs/train/data_mission_six_axis_upgrade.yaml`

- 六轴 MuJoCo 场景，任务长度 3，启用域随机化与 privileged IK expert；
- 24 个回合，21 个专家回合成功；
- 7,780 条 transition，图像尺寸 64×64，机器人状态 29 维，动作 6 维；
- 监督标签包含动作、目标像素（`goal_xy`）和操作阶段（9 类）；
- 数据为合成数据，不含真实机械臂采样。

数据清单与哈希保存在本地生成目录：
`datasets/generated/upgrade_20260922/mission_six_axis_aux.manifest.json`。
原始 NPZ 不纳入发布提交，便于控制仓库体积并避免把训练输入误当作实机证据。

## 2. 辅助监督训练

配置：`configs/train/temporal_vla_mission_six_axis_aux.yaml`

| 项目 | 结果 |
| --- | --- |
| 设备 | CUDA（NVIDIA GeForce RTX 3050 Laptop GPU） |
| epoch / optimizer steps | 12 / 1,140 |
| 可训练参数 | 1,830,481 |
| 最佳 epoch | 8 |
| 最佳验证损失 | 0.1163348714 |
| 辅助项 | 目标像素 Smooth L1 + 阶段 Cross Entropy |

训练 manifest：
`checkpoints/upgrade_20260922/temporal_vla_mission_six_axis_aux/manifest.json`。
权重文件保留在本地生成目录；发布提交只保留配置、训练历史和可审计摘要。

## 3. 固定种子回放

配置：`configs/eval/temporal_vla_upgrade_long.yaml`，五套件（ID、paraphrase、OOD、physics、perception），每套件 4 回合。

| 方法 | 回合 | 成功 | 碰撞记录 | 超时 | 错误抓取 | 错误分箱 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IK expert | 20 | 17 | 20 | 3 | 0 | 0 |
| temporal_vla_aux | 20 | 0 | 14 | 19 | 9 | 1 |

详细逐回合数据：
`results/upgrade_20260922/temporal_vla_mission_six_axis_aux_eval/eval_episodes.csv`；
汇总：
`results/upgrade_20260922/temporal_vla_mission_six_axis_aux_eval/summary.json`。

Temporal VLA 推理延迟约为 4.1–4.3 ms 平均值、约 10 ms 级 p95；当前短周期训练尚未形成可靠的闭环抓取策略。该结果证明了多任务标签、训练、checkpoint 加载和回放链路可复现，但不代表真实机械臂可用。

## 4. 代码质量门

```text
ruff check src tests : passed
pytest -q            : 80 passed
```

## 5. 结论与下一步

本轮完成了六轴目标/阶段辅助监督链路、完整数据生成、CUDA 训练和固定种子回放。下一轮应优先：

1. 用成功回合与失败回合分层采样，修正阶段标签和终止条件；
2. 增加行为克隆 warm-start 或 DAgger 回收，而不是直接把当前 checkpoint 当作部署模型；
3. 对碰撞/超时回合逐步调试 action horizon、replan interval 与动作尺度；
4. 在真实硬件回来后，另行完成六轴反馈新鲜度、TCP/标定和低速跟踪验收。
