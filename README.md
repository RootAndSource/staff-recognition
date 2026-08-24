# StaffRecognition - 员工检测与工装识别

对固定机位、斜俯视角的教室/会议室本地视频做**离线批处理**：检测画面中所有人员，识别是否穿红色工装（背心/短袖），为每个人维护跨帧稳定的 person ID。为任务 B/C/D 的按人统计提供基础（人员框、工装标签、身份 ID）。

## 主链路（S1-S7）

```
视频文件（离线批处理）
│
└── S1 全身检测：YOLO26x 出全身框
    │
    ├── S2 实例分割：SAM2.1 对框内出可见块掩码 1~N 块
    │   │
    │   └── S4 工装识别：掩码内工装色规则（免训练）
    │       └── S6 身份与状态：person_id + 工装记忆
    │           └── S7 输出：JSONL + 叠加视频
    │
    └── S5 跟踪（并行支线）：YOLO26 + ByteTrack → track_id → 汇入 S6
```

> 注：S3 上半身切割已弃用；S4 直接基于 S2 可见掩码做工装色规则判定，不依赖标注数据训练分类器。

## 技术栈

| 环节 | 方案 |
| --- | --- |
| 人员检测（S1） | YOLO26x（COCO 预训练，不微调） |
| 实例分割（S2） | SAM2.1 large box-prompt |
| 工装识别（S4） | 掩码内工装色规则（红色占比 + 最大连通块） |
| 跟踪（S5） | ByteTrack（track_buffer=150 等） |
| 身份与状态（S6） | IdentityTracker + 工装记忆层（动态翻转难度） |
| 输出（S7） | JSONL + 叠加视频 |

## 目录结构

```
src/            核心代码（检测/跟踪/身份/工装/输出）
scripts/
  eval/         评估脚本（工装指标、ID 稳定性）
tests/          单元测试
config.yaml     全链路配置（最终定案）
main.py         CLI 入口
data/
  preprocessed/ 批量输入视频（隐私，不入库）
models/         模型权重（不入库，见下方获取方式）
outputs/
  json/         批量 JSON 输出
  mp4/          批量视频输出
```

## 安装

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

GPU 版 torch（RTX 4070 实测，YOLO26x + SAM2.1 离线约 1.4 fps）：

```bash
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 模型获取

本项目使用两个官方预训练模型，不提供模型权重文件：

| 模型 | 用途 | 获取方式 |
| --- | --- | --- |
| YOLO26x | S1 全身检测 | `pip install ultralytics` 后首次运行自动下载，或从 [Ultralytics Assets](https://github.com/ultralytics/assets/releases) 下载 `yolo26x.pt` 放入 `models/` |
| SAM2.1 large | S2 实例分割 | 从 [Hugging Face: facebook/sam2.1](https://huggingface.co/facebook/sam2.1) 下载 `sam2.1_l.pt` 放入 `models/` |

`models/` 目录无需手动创建，运行前保证两个权重文件存在即可。

## 数据说明

输入为本地录制的教室/会议室视频（固定机位、斜俯视角，每画面 5-15 人）。视频数据涉及隐私，**不随本仓库公开**。复现需自行准备视频，放入 `data/preprocessed/`（支持 `.mp4/.avi/.mov`）。

## 运行

```bash
# 单个视频（输出 JSONL + 视频）
python main.py run data/preprocessed/005.mp4 \
  --json outputs/json/005.jsonl --save-video outputs/mp4/005.mp4 --no-show

# 批量处理 data/preprocessed/ 下所有视频
python main.py batch

# 性能分析
python main.py run data/preprocessed/005.mp4 --no-show --profile
```

`config.yaml` 即最终定案配置（长记忆），直接运行即可；如需调节阈值可修改其中 `identity` / `uniform` 段。

## JSONL 输出格式

每帧一条 JSON，`people` 数组含每位人员的记录：

```json
{
  "frame_id": 123,
  "timestamp_ms": 4100.0,
  "people": [
    {
      "person_id": 3,
      "bt_id": "s7",
      "bbox": [120.5, 80.2, 240.1, 420.8],
      "uniform": true,
      "uniform_raw": true,
      "uniform_memory": true,
      "uniform_state": "confirmed",
      "uniform_prob": 0.95,
      "mask_polygon": [[0.12, 0.08], [0.20, 0.35]],
      "confidence": 0.87,
      "restored": false
    }
  ]
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `person_id` | 稳定身份 ID（跨帧、跨遮挡恢复） |
| `bt_id` | 底层 ByteTrack ID |
| `bbox` | 全身检测框（像素） |
| `uniform` | 最终工装标签 |
| `uniform_raw` | 当前帧掩码检测结果（S4 原始） |
| `uniform_memory` | 长期记忆判断结果（S6） |
| `uniform_state` | pending / active / confirmed |
| `uniform_prob` | 工装证据强度 0~1 |
| `mask_polygon` | 人体掩码简化多边形（归一化坐标） |
| `restored` | 是否从消失恢复 |

## 关键设计

- **工装识别免训练**：在 SAM2 人体掩码内统计红色占比与最大红色连通块占比，排除桌面/墙壁背景干扰，不依赖标注数据。
- **工装记忆**：`uniform_memory` 只由当前帧掩码检测结果更新（不被记忆输出污染）；翻转难度随状态持续时间动态增长；穿→未穿极难（排除换装）、未穿→穿较易（见工装色即翻）。
- **隐私**：视频数据、模型权重、输出产物均不入 git，全程本地处理。

## 许可证

本项目采用 [MIT License](LICENSE)。

## 引用

如论文或工作中使用本项目，请引用：

```bibtex
@software{staffrecognition2026,
  author = {Dai, Benyuan},
  title  = {StaffRecognition: Person Detection and Uniform Recognition},
  year   = {2026},
  url    = {https://gitee.com/dai-benyuan/staff-recognition}
}
```
