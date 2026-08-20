# 任务 A：员工检测（人员检测 + 工装识别）

单摄像头（固定机位，斜俯视）检测会议/教室场景中的所有人员，识别是否穿红色工装
（背心/短袖），为每个人员维护跨帧持续 ID，供任务 B/C/D 按人统计使用。

## 当前技术栈

| 环节 | 方案 |
| --- | --- |
| 人员检测 | YOLO11x 现场微调版（`models/yolo11x_finetuned.pt`，mAP50=0.965） |
| 多目标跟踪 | BoT-SORT（运动 + ReID 外观融合，`reid_model: auto` 用检测器特征） |
| ID 管理 | IdentityTracker：画面内消失 10 秒暂存恢复、边界离开释放、新 ID 3 帧确认 |
| 工装判定 | HSV 规则（当前过渡）+ MobileNetV3-Small 分类器（`models/uniform_cls.pt`，已训练待接入） |
| 工装状态 | 延迟态状态机（pending/active/confirmed，翻转需连续反向证据） |
| 输出 | JSONL（person_id/bbox/uniform/state/置信度）+ 叠加视频 |

## 目录结构

```
src/            核心代码（检测/跟踪/ID/工装/输出）
scripts/        数据准备与训练脚本（M2/M3）
tests/          单元测试
config.yaml     全部可标定参数
data/
  preprocessed/   批量输入视频（001.mp4 ...）
  finetune/       检测微调数据（frames + annotations）
  uniform_crops/  工装分类数据（sample/positive|negative）
models/         模型权重（检测器/分类器，不入库）
outputs/
  json/         批量 JSON 输出
  mp4/          批量视频输出
  evaluate/     评估产物
docs/           方案/需求/调研/标注指南（本地，不入库）
```

## 安装

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

GPU 版 torch（RTX 4070 实测约 20 FPS @ yolo11x-960）：

```bash
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

标注工具（M2 用）：

```bash
.\.venv\Scripts\pip install "PyQt5==5.15.9" "PyQt5-Qt5==5.15.2" labelImg==1.8.6
```

> labelImg 1.8.6 与新版 PyQt5 不兼容，需固定 5.15.9 并打 7 处 float 兼容补丁
> （`labelImg.py` 的 scroll/setValue、`libs/canvas.py` 的 drawRect/drawLine、
>  `libs/shape.py` 的 drawText，坐标转 int）。

## 运行

```bash
# 单视频（输出到 outputs/json、outputs/mp4）
python main.py run data/preprocessed/005.mp4 --no-show --json outputs/json/005.jsonl --save-video outputs/mp4/005.mp4

# 批量处理 data/preprocessed/ 下所有视频
python main.py batch

# 性能分析
python main.py run data/preprocessed/005.mp4 --no-show --profile

# ID 稳定性评估（短时遮挡恢复率，验收口径 ≥80%）
python scripts/evaluate_id.py outputs/json/005.jsonl
```

`--show` 显示实时预览；`--no-show` 不影响 JSON/视频输出。

## 数据与模型（隐私，全部不入库）

- 源视频：`data/preprocessed/`（15 个现场视频，16GB）
- 检测微调：375 帧人工标注（`data/finetune/`）→ 训练出 `models/yolo11x_finetuned.pt`
- 工装分类：480 张上半身 crop 人工标注（`data/uniform_crops/sample/`）→
  `models/uniform_cls.pt`（MobileNetV3-Small，逐类准确率 100%；
  未穿样本仅 12 张，代表性有限，待后续补标）

## 里程碑状态

- M1 ✅ BoT-SORT + ReID：完整视频唯一 ID 30→24，短时遮挡恢复率 98.4%
- M2 ✅ 检测微调：人工标注 375 帧，mAP50 0.941→0.965，已切换为默认权重
- M3 ✅ 分类器训练：crop 480 张标注完成，模型已训练（推理接入待做）

## 验收口径（已确定，2026-08-19）

- 工装：抽查 100 帧准确率 ≥95%；crop 验证集准确率 ≥95%
- ID：短时遮挡（≤3 秒）恢复率 ≥80%；连续可见期间 ID 切换为 0（目标值）
- 检测：现场验证集统计检出率/漏检率
- 性能：FPS ≥15、连续运行 ≥1 小时不崩溃

## 隐私说明

全部处理在本机完成，视频与画面不出本机、不联网、不上传任何平台。
