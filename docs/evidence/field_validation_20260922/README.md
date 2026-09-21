# 现场能力回报（2026-09-22）

本条目是 `PickSort-VLA` 与小U主仓库之间的现场结果索引。项目操作者确认真实机械臂已经成功运动，能够抓取桌面物品、整理桌面并清理垃圾，且预设功能已完备。

这是一条 `operator_reported` 定性记录，不是本仓库重新采集的 benchmark。当前没有原始串口日志、视频、设备型号、固件哈希、标定版本、执行次数或失败计数，因此不发布成功率、延迟、负载和重复性数字。

## 证据边界

- 本仓库仍是仿真与具身学习主线；已有 MuJoCo、ROS 2 dry-run、协议回放和 Temporal VLA 结果保持原样。
- 既有离线策略的失败回放继续保留，不因现场反馈而改写为学习策略的实机指标。
- 现场结果的完整记录位于[小U主仓库的现场验证条目](https://github.com/1830051801-ux/xiaou-vision-robot-arm/blob/codex/complete-xiaou-stack/docs/evidence/field_validation_20260922/README.md)。
- 后续补充原始日志、视频和重复试验后，再新增可审计的现场验收附件。

机器可读摘要见 [`field_validation.json`](field_validation.json)。
