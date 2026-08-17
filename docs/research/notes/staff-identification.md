# 分析笔记：服装与工装识别

> 服务于任务 A 的“是否穿工装（红色背心/短袖）”二分类。
> 检索日期：2026-08-17；数据源：OpenAlex、arXiv、Crossref + 网页核验。
> 引用约定：文献原件在 Zotero；下方 “Zotero key” 待导入后回填。

## 1. 服装/工装识别

### Pedestrian Attribute Recognition: A Survey（2022·Pattern Recognition）

- 作者：Xiao Wang, Shaofei Zheng, Rui Yang, et al.
- DOI：10.1016/j.patcog.2021.108220 ｜ arXiv：1901.07474
- 要点：行人属性识别综述，含服装颜色/款式等属性分类的方法、数据集与评测口径。
- 用途：为“工装=红色”这类属性识别提供方法论与数据集参考；任务 A 的学术依据。
- Zotero：00-共享基础/服装与属性识别（工服） ｜ 标签：综述、方法
- Zotero key：IEFVK5G5

### Open-World Person Re-Identification With RGBD Camera in Top-View Configuration for Retail Applications（2020·IEEE Access）

- 作者：Massimo Martini, Marina Paolanti, Emanuele Frontoni
- DOI：10.1109/ACCESS.2020.2985985
- 要点：零售场景顶视摄像头下以服装/外观特征关联人员，证明俯视角度下颜色-外观特征可用。
- 用途：佐证“颜色/外观特征在俯视视角下仍稳定”这一工程假设。
- Zotero：00-共享基础/服装与属性识别（工服） ｜ 标签：方法
- Zotero key：QWV8F5XJ

> 说明：专门的“工服检测/工装识别”文献很少，通常归属行人属性识别（attribute recognition）或颜色直方图匹配。任务 A POC 采用预训练人员检测 + 上半身 ROI 的 HSV 红色匹配（对应 Kimi 方案）；若现场误报率高，再评估引入属性识别模型或采集少量数据微调。
