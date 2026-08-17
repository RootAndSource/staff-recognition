# 分析笔记：人员检测与跟踪

> 服务于任务 A（员工检测：人员检测 + 是否穿工装），并为 B/C/D 的“人员持续 ID”打基础。
> 检索日期：2026-08-17；数据源：OpenAlex、arXiv、Crossref + 网页核验（Semantic Scholar 限流未取全）。
> 引用约定：文献原件在 Zotero；下方 “Zotero key” 待导入后回填。

## 1. 人员检测

### You Only Look Once: Unified, Real-Time Object Detection（2016·CVPR）

- 作者：Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- DOI：10.1109/CVPR.2016.91 ｜ arXiv：1506.02640
- 要点：单阶段检测开山之作，实时性与精度平衡，是 YOLO 系列的事实起点。
- 用途：理解检测器基础；任务 A 用 YOLO 系列预训练模型作为人员检测主干。
- Zotero：00-共享基础/人员检测与跟踪 ｜ 标签：方法
- Zotero key：RNACS4BD

### YOLOv10: Real-Time End-to-End Object Detection（2024·arXiv）

- 作者：Ao Wang, Hui Chen, Lihao Liu, et al.（清华大学）
- arXiv：2405.14458
- 要点：去 NMS 的端到端实时检测，效率和精度均优，工程集成简单。
- 用途：任务 A 的候选人员检测器之一（配合 ultralytics 工具链）。
- Zotero：00-共享基础/人员检测与跟踪 ｜ 标签：方法
- Zotero key：UW4MF6WP

### RTMDet: An Empirical Study of Designing Real-Time Object Detectors（2023·CVPR）

- 作者：Chengqi Lyu, Wenwei Zhang, Haian Huang, et al.
- DOI：10.1109/CVPR52729.2023.01169 ｜ arXiv：2212.07784
- 要点：面向实时检测的架构设计经验总结，速度/精度权衡出色，MM 工具链完善。
- 用途：YOLO 之外的备选检测器，便于在 mmdet/mmyolo 生态中与后续姿态模型打通。
- Zotero：00-共享基础/人员检测与跟踪 ｜ 标签：方法
- Zotero key：GIKCUUJN

### Object Detection Using YOLO: Challenges, Architectural Successors, Datasets and Applications（2023·MTA）

- 作者：Tausif Diwan, G. Anirudh, Jitendra V. Tembhurne
- DOI：10.1007/s11042-022-13644-y
- 要点：YOLO 系列演进综述，含数据集与评测口径对比。
- 用途：检测器选型与指标口径参考。
- Zotero：00-共享基础/人员检测与跟踪 ｜ 标签：综述
- Zotero key：ATMZQ3BM

### DETRs Beat YOLOs on Real-time Object Detection（2024·CVPR）

- 作者：Yian Zhao, Wenyu Lv, Shangliang Xu, et al.
- DOI：10.1109/CVPR52733.2024.01605
- 要点：DETR 类模型在实时检测上反超 YOLO，说明检测器选择有持续演进空间。
- 用途：背景参考，避免选型视野过窄。
- Zotero：00-共享基础/人员检测与跟踪 ｜ 标签：综述
- Zotero key：I99KJVPX

## 2. 多目标跟踪与人员关联

### ByteTrack: Multi-Object Tracking by Associating Every Detection Box（2022·ECCV）

- 作者：Yifu Zhang, Peize Sun, Yi Jiang, et al.
- DOI：10.1007/978-3-031-20047-2_1 ｜ arXiv：2110.06864
- 要点：保留低分检测框参与关联，密集/遮挡场景 ID 稳定性好，30 FPS 可实时运行。
- 用途：任务 A 人员持续 ID 的首选跟踪器（对应 Kimi 方案选型）。
- Zotero：00-共享基础/多目标跟踪与人员关联 ｜ 标签：方法
- Zotero key：8HSPQ4HT

### Simple Online and Realtime Tracking with a Deep Association Metric（2017·ICIP）

- 作者：Nicolai Wojke, Alex Bewley, Dietrich Paulus
- DOI：10.1109/ICIP.2017.8296962 ｜ arXiv：1703.07402
- 要点：DeepSORT 基线：运动（卡尔曼）+外观（ReID embedding）级联匹配。
- 用途：跟踪算法基线对比，理解 ByteTrack 的改进点。
- Zotero：00-共享基础/多目标跟踪与人员关联 ｜ 标签：方法
- Zotero key：TXHKP6Q8

### StrongSORT: Make DeepSORT Great Again（2023·IEEE TMM）

- 作者：Yunhao Du, Zhicheng Zhao, Yang Song, et al.
- DOI：10.1109/TMM.2023.3240881 ｜ arXiv：2202.13514
- 要点：在 DeepSORT 上系统补强（检测、外观、AFLink/GSI 后处理），指标优于原版。
- 用途：DeepSORT 路线的升级备选。
- Zotero：00-共享基础/多目标跟踪与人员关联 ｜ 标签：方法
- Zotero key：2KRKVTSM

### Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking（2023·CVPR）

- 作者：Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, et al.
- DOI：10.1109/CVPR52729.2023.01362 ｜ arXiv：2203.14360
- 要点：OC-SORT 缓解遮挡/非线性运动下的 ID 切换，适合人员密集与短暂遮挡场景。
- 用途：会议场景人员起身、入座等动作下的 ID 稳定性对比候选。
- Zotero：00-共享基础/多目标跟踪与人员关联 ｜ 标签：方法
- Zotero key：5W57RFUI

### Deep Learning in Video Multi-Object Tracking: A Survey（2020·Neurocomputing）

- 作者：Gioele Ciaparrone, Francisco Luque Sánchez, Siham Tabik, et al.
- DOI：10.1016/j.neucom.2019.11.023
- 要点：MOT 分类体系、评测指标（MOTA/IDF1/HOTA）与深度方法全景。
- 用途：指标口径与算法术语基准，后续验收指标（ID 稳定性）对齐用。
- Zotero：00-共享基础/多目标跟踪与人员关联 ｜ 标签：综述
- Zotero key：ITNB8829

## 3. 视角适配（斜俯视/顶视）

### Open-World Person Re-Identification With RGBD Camera in Top-View Configuration for Retail Applications（2020·IEEE Access）

- 作者：Massimo Martini, Marina Paolanti, Emanuele Frontoni
- DOI：10.1109/ACCESS.2020.2985985
- 要点：零售场景顶视摄像头下的开集人员重识别，与本项目“天花板-墙壁交叉处俯视”配置最接近的公开研究。
- 用途：当前单摄像头阶段作背景参考；多摄像头扩展时用于跨摄像头身份关联。
- Zotero：00-共享基础/多目标跟踪与人员关联 ｜ 标签：方法、数据集
- Zotero key：QWV8F5XJ

> 说明：斜俯视/顶视下“人员检测+工装分类”的专门文献较少，工程上多以预训练检测器 + 颜色规则实现（对应 Kimi 方案）。POC 先直接验证，再按实测结果决定是否微调。
