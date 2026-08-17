# 任务 A：员工检测（人员检测 + 工装识别）

单摄像头（固定机位，斜俯视）实时检测会议/教室场景中的所有人员，
判定其是否穿着红色工装（背心/短袖），为每个人员维护持续 ID，
供任务 B/C/D 按人统计使用。

## 目录结构

```
docs/           文档（plan 方案 / requirements 需求 / research 调研）
src/            核心代码（检测/跟踪/工装/输出）
scripts/        标定与评估脚本
tests/          单元测试
config.yaml     全部可标定参数
data/           现场/测试视频与截图（隐私，不入库）
models/         本地模型权重（不入库）
outputs/run/    运行产物（JSON/输出视频/评估 CSV，不入库）
```

## 安装

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

首次运行会自动下载 YOLO 预训练权重（`yolo11n.pt`）到本地缓存；
权重路径可在 `config.yaml` 的 `model.weights` 指定本地文件。

## GPU 加速

检测到 NVIDIA GPU 时（如 RTX 4070），ultralytics 会自动使用 CUDA；
`config.yaml` 的 `model.device` 留空表示自动选择，也可显式指定 `"cpu"` 或 `"0"`。
安装 CUDA 版 torch：

```bash
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 运行

```bash
# 本地视频
python main.py run path/to/video.mp4 --json outputs/run/out.jsonl --save-video outputs/run/out.mp4

# RTSP 流（无显示器环境加 --no-show）
python main.py run rtsp://user:pass@host:554/stream --no-show --json outputs/run/out.jsonl

# 说明：--save-video 时输出视频始终叠加检测框/标签/统计；
#       --show 仅控制是否弹出实时预览窗口，--no-show 不影响视频与 JSON 输出。

# 交互标定 HSV / ROI（需要现场截图）
python main.py calibrate data/frame.jpg

# 抽帧 + 生成人工标注模板
python main.py evaluate --video data/videos/005.mp4 --frames 100
```

## 现场标定流程

1. 从现场视频抽若干帧（`evaluate` 或 `video_input` 均可）。
2. `python main.py calibrate <截图>`，拖动滑块找到红色工装的分界。
3. 把最终参数写回 `config.yaml` 的 `uniform` 段。

## 工装判定逻辑（v0.2）

- 双段 ROI：主段（肩胸，y 0.15-0.55）+ 兜底段（腹腰，y 0.45-0.85），
  任一段红色占比达标即判工装，缓解斜俯视/桌子遮挡导致的躯干外移。
- 时间平滑：按 `track_id` 维护 5 帧滑动窗口，取多数票，减少单帧抖动。
- 工装特征为红色背心/短袖（上半身），HSV 阈值与占比阈值集中在 `config.yaml`。

## 实时性与性能

- 首帧推理较慢（模型加载 + CUDA 预热，约 1-2 秒），之后单帧约 15-20ms（GPU）。
- 偶发卡顿排查：`python main.py run <视频> --no-show --profile` 打印各环节耗时。
- 实时预览（`--show`）在高分辨率下会占用显示开销；批处理建议 `--no-show --save-video`。

## 隐私说明

全部处理在本机完成，视频与画面不出本机、不联网、不上传任何平台。
`data/`、`models/`、`outputs/` 均已加入 `.gitignore`，不入库。
