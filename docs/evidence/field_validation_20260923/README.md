# XiaoU field record index (2026-09-23)

PickSort-VLA keeps the field record as a linked evidence index. The canonical
source, byte-for-byte workbook, complete worksheet exports, and hash manifest
live in the [XiaoU repository](https://github.com/1830051801-ux/xiaou-vision-robot-arm/tree/codex/complete-xiaou-stack/docs/evidence/field_validation_20260923).

The supplied workbook records 972/1000 successful first attempts, 105/108 in
the grasp aggregate, 12 YOLO classes, and a 40.8 s cycle-time summary. Its VLA
detail table (20 rows, 19 successes, 4.535 ms mean) and overview summary (30
instructions, 29 successes, 4.63 ms mean, 5.52 ms P95) are intentionally kept
as separate source populations.

PickSort-VLA simulation, training, and ROS 2 dry-run metrics remain separate
from this field record. This file does not claim that a PickSort policy was
deployed on the physical arm.

Machine-readable copy: [`field_validation.json`](field_validation.json).
