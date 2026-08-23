"""StaffRecognition - S1 人员检测：ultralytics YOLO，推理时仅保留 person 类别。"""

from __future__ import annotations

from typing import Any

class PersonDetector:
    """YOLO person 检测器封装。

    模型在首次调用时惰性加载，避免导入阶段联网/占显存。
    """

    def __init__(
        self,
        weights: str = "models/yolo26x.pt",
        conf_thres: float = 0.35,
        iou_thres: float = 0.5,
        imgsz: int = 640,
        device: str = "",
    ) -> None:
        """初始化检测器参数（模型懒加载）。"""
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = device
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        """返回已加载的 YOLO 模型，未加载则先 load。"""
        if self._model is None:
            self.load()
        return self._model

    def load(self) -> Any:
        """显式加载模型；在 pipeline 启动时调用，让加载失败尽早暴露。"""
        if self._model is None:
            from ultralytics import YOLO  # 延迟导入，保持启动轻量

            self._model = YOLO(self.weights)
        return self._model

