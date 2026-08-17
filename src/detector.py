"""人员检测：ultralytics YOLO，推理时只保留 person 类别。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    """单个检测框（像素坐标，xyxy）。"""

    xyxy: np.ndarray        # [x1, y1, x2, y2]
    confidence: float


class PersonDetector:
    """YOLO person 检测器封装。

    模型在首次调用时惰性加载，避免导入阶段联网/占显存。
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf_thres: float = 0.35,
        iou_thres: float = 0.5,
        imgsz: int = 640,
        device: str = "",
    ) -> None:
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = device
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    def load(self) -> Any:
        """显式加载模型；在 pipeline 启动时调用，让加载失败尽早暴露。"""
        if self._model is None:
            from ultralytics import YOLO  # 延迟导入，保持启动轻量

            self._model = YOLO(self.weights)
        return self._model

    def detect(self, frame_bgr: cv2.Mat) -> list[Detection]:
        """对一帧 BGR 图像做推理，只返回 person(类别0) 检测。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        model = self._ensure_model()
        results = model.predict(
            frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device or None,
            classes=[0],          # COCO 中 person=0
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                detections.append(
                    Detection(xyxy=box.astype(np.float32), confidence=float(conf))
                )
        return detections
