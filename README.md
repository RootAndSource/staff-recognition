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

默认使用 `yolo11m.pt`（遮挡/坐姿场景检出率显著优于 11n，RTX 4070 下约 12-23ms/帧）；
权重路径可在 `config.yaml` 的 `model.weights` 指定本地文件。
模型下载：`curl.exe --ssl-no-revoke -L -o models/yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt`

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

# 批量处理（输入目录 → outputs/json、outputs/mp4 分类输出）
python main.py batch
# 自定义目录
python main.py batch --input-dir data/preprocessed --json-dir outputs/json --mp4-dir outputs/mp4

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

## 身份与工装状态管理（v0.4）

- **工装判定（单帧）**：双段 ROI（肩胸 + 腹腰）HSV 红色检测 +
  最大红色连通域占比（工装成块先验）+ 场景红色基线对比
  （ROI 红色需显著高于全框红色，过滤背景红干扰）。
- **延迟态状态机（跨帧）**：`pending`（观察期不输出标签）→
  `active`（滑动窗口 + 翻转惯性：连续 5 帧反向证据才翻转）→
  `confirmed`（长时间高一致标记高置信，但**不阻止翻转**）。
  标签可翻转、不固化，避免误判被永久锁定。
- **轨迹暂存恢复**：目标消失（遮挡等）后暂存 3 秒并做位置外推；
  重新出现且位置匹配时复用原 `person_id` 与工装状态，避免 ID 膨胀。
- **同帧重复框合并**：趴姿/斜俯视下 YOLO 会对同一人输出“小框+大框”，
  后置合并（IoU ≥0.3 保留高置信度框）避免一个员工出现两个检测框。
- JSON 输出：`uniform`、`uniform_state`（pending/active/confirmed）、
  `uniform_prob`、`red_ratio_connected`（最大红色连通域占比）、
  `baseline_ratio`（场景红色基线），便于排查误判。

## 身份与工装状态管理（v0.5，ID 生命周期）

- **ByteTrack 参数真正生效**：自定义 `configs/bytetrack.yaml`（track_buffer 等），
  目标短暂漏检时 ByteTrack 自己保持 track_id。
- **画面内绝不分裂 ID**：画面内消失的目标暂存 10 秒（位置外推 + HSV 外观直方图匹配），
  重新出现复用原 `person_id` 与工装状态；JSON 输出 `person_restored`/`restored_from`。
- **只有真正走出画面才释放**：检测框可见面积 <40% 视为离开（被画面裁切的目标不算），
  释放后再次进入按新 ID 处理。
- **新 ID 确认期**：连续 3 帧稳定才分配正式 `person_id`，短命误检不再消耗 ID。

## v0.6：BoT-SORT + ReID（按方案 v0.2 M1）

- 跟踪器升级为 **BoT-SORT（运动 + ReID 外观融合关联）**（ultralytics 内置，
  `configs/botsort.yaml`，`with_reid: true`），替换纯运动 ByteTrack；
  ReID 额外开销实测仅 ~0.4ms/帧。
- 固定机位关闭相机运动补偿（`gmc_method: none`）；
  `appearance_thresh=0.8` 防 ID 交换，`track_buffer=150`（5 秒）保持轨迹。
- 完整视频（28182 帧）实测：唯一 person_id 30 → 24，工装翻转 532 → 489。
- 验收口径（已确定，2026-08-19，见 docs/requirements v0.3）：
  短时遮挡（≤3 秒）恢复率 ≥80%；连续可见期间 ID 切换为 0（目标值）。
- ReID 当前用检测器特征（`reid_model: auto`，零额外开销）；
  `scripts/calibrate_reid.py` 可标定阈值。实测 yolo26n-cls 通用特征
  在统一工装场景下同人/异人区分度有限（p5=0.940 vs p95=0.943 重叠），
  专用 ReID 模型（OSNet 级）是后续升级点。
- `scripts/evaluate_id.py <jsonl>` 自动评估短时遮挡恢复率（验收口径 ≥80%）。

## 实时性与性能

- 首帧推理较慢（模型加载 + CUDA 预热，约 1-2 秒），之后单帧约 15-20ms（GPU）。
- 偶发卡顿排查：`python main.py run <视频> --no-show --profile` 打印各环节耗时。
- 实时预览（`--show`）在高分辨率下会占用显示开销；批处理建议 `--no-show --save-video`。

## 隐私说明

全部处理在本机完成，视频与画面不出本机、不联网、不上传任何平台。
`data/`、`models/`、`outputs/` 均已加入 `.gitignore`，不入库。
