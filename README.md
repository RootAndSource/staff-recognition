# 任务 A：员工检测（人员检测 + 工装识别）

对固定机位、斜俯视角的教室/会议室本地视频做离线批处理：检测画面中所有人员，识别是否穿红色工装（背心/短袖），为每个人维护跨帧稳定的 person ID，供任务 B/C/D 按人统计使用。

## 主链路（S1-S7，已定稿 v0.6）

```
视频文件（离线批处理）
│
└── S1 全身检测：YOLO26x 出全身框
    │
    ├── S2 实例分割：SAM2.1 对框内出可见块掩码 1~N 块
    │   │
    │   ├── S3 上半身切割：弃用
    │   │
    │   ├── S4 工装识别：掩码内工装色规则（免训练）
    │   │
    │   └── S6 身份与状态：person_id + 工装记忆（动态翻转难度）
    │       └── S7 输出：JSONL + 叠加视频
    │
    └── S5 跟踪（并行支线）：YOLO26 + ByteTrack → track_id → 汇入 S6
```

## 技术栈

| 环节 | 方案 | 状态 |
| --- | --- | --- |
| 人员检测（S1） | YOLO26x（COCO 预训练，不微调） | ✅ 定案 |
| 实例分割（S2） | SAM2.1 large box-prompt | ✅ 定案 |
| 工装识别（S4） | 掩码内工装色规则（红色占比 + 连通块） | ✅ 定案（免训练） |
| 跟踪（S5） | ByteTrack（track_buffer=150 等调参） | ✅ 定案 |
| 身份与状态（S6） | IdentityTracker + 工装记忆层 | ✅ 定案（长记忆） |
| 输出（S7） | JSONL + 叠加视频 | ✅ 定案 |

## 目录结构

```
src/            核心代码（检测/跟踪/身份/工装/输出）
scripts/
  pipeline/     主线组合脚本（S1/S2/S5、对比、渲染、扫描）
  train/        数据准备与训练（M2/M3）
  eval/         评估脚本
  tools/        辅助工具
tests/          单元测试
config.yaml     默认全链路配置（最终定案）
configs/        s1246.yaml（正式配置，入库；bytetrack.yaml 运行时生成，不入库）
data/
  preprocessed/ 批量输入视频（隐私，不入库）
  finetune/     检测微调数据（M2/M4，不入库）
  uniform_crops/工装分类数据（M3，不入库）
models/         模型权重（不入库）
outputs/
  json/         批量 JSON 输出
  mp4/          批量视频输出
docs/           方案/需求/调研/标注指南（本地，不入库）
```

## 安装

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

GPU 版 torch（RTX 4070 实测，YOLO26x + SAM2.1 离线约 1.4fps）：

```bash
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 运行

```bash
# 单视频（输出 JSONL + 视频）
python main.py run data/preprocessed/005.mp4 \
  --json outputs/json/005.jsonl --save-video outputs/mp4/005.mp4 --no-show

# 批量处理 data/preprocessed/ 下所有视频
python main.py batch

# 性能分析
python main.py run data/preprocessed/005.mp4 --no-show --profile
```

config.yaml 即最终定案配置（长记忆），直接 `python main.py run ...` 即可。

## JSONL 输出字段

每条人员记录包含：`person_id`、`bt_id`（ByteTrack ID）、`bbox`、`uniform`（最终标签）、
`uniform_raw`（当前帧掩码检测）、`uniform_memory`（记忆判断）、`uniform_state`、
`uniform_prob`、`mask_polygon`、`confidence`、`restored`、`split_relation`。

## 关键设计说明

- **工装识别免训练**：M3 分类器未穿样本仅 12 张，改用 SAM2 人体掩码内的红色规则，
  排除桌面/墙壁背景，不依赖标注数据。
- **工装记忆**：`uniform_memory` 只由当前帧掩码检测结果更新（不被记忆输出污染）；
  翻转难度随状态持续时间动态增长；穿→未穿极难（排除换装）、未穿→穿较易。
- **隐私**：视频数据、模型权重、输出产物、docs 均不入 git，全程本地处理。

## 文档

- 最终实现方案：`docs/plan/task-a-person-detection-v0.6.md`
- 要求文档：`docs/requirements/task-a-person-detection.md`（导师侧）
- 问题反馈：`docs/feedback/问题反馈-任务A员工检测.md`
- 调研笔记：`docs/research/notes/`
