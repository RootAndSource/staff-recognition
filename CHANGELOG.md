# Changelog

All notable changes to this project are documented in this file.

## [0.6.0] - 2026-08-23

### Added

- S1 全身检测：YOLO26x（COCO 预训练，不微调）
- S2 实例分割：SAM2.1 large box-prompt，输出可见人体掩码
- S4 工装识别：掩码内工装色规则（免训练），替代 M3 MobileNet 分类器
- S6 工装记忆层：冷启动即出标签、动态翻转难度、穿→未穿极难（排除换装）
- S7 输出：JSONL 含 `uniform_raw` / `uniform_memory` / `uniform` 三字段
- 正式配置：`config.yaml`（长记忆定案版）

### Changed

- Pipeline 精简为单一 segmentation 链路（S1-S7）
- 移除历史双检测融合（upper/full）与实例外观归并（M6）
- Scripts 精简为 eval 评估脚本（工装指标、ID 稳定性）
- 跟踪改为 ByteTrack（track_buffer=150 调参定案）

### Removed

- `src/fusion.py`、`src/instance_merge.py`
- M5/M6/M7 废弃脚本与测试产物
- 旧 M3 分类器依赖（`models/uniform_cls.pt`）

### Fixed

- S4 工装 ROI 与 M3 训练分布错位导致的"全判穿"问题
- 工装标签跨帧抖动（跳跃次数 76 → 5）
