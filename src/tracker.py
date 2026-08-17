"""多目标跟踪：ByteTrack 封装，维护跨帧 track_id。

设计为可替换接口：新增跟踪器时实现同一 update 签名即可
（预留 OC-SORT / StrongSORT 切换位）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .detector import Detection, PersonDetector


@dataclass
class TrackedObject:
    """跟踪器输出的一个目标。"""

    track_id: int
    xyxy: np.ndarray        # [x1, y1, x2, y2]
    confidence: float
    is_confirmed: bool      # 轨迹是否已确认（过滤冷启动阶段的闪烁轨迹）


@dataclass
class ByteTrackTracker:
    """基于 ultralytics 内置 ByteTrack（persist=True）的轻量封装。

    与 PersonDetector 共享同一个 YOLO 实例：track() 本身包含检测，
    避免同一帧重复推理（这是实时性 ≥15 FPS 的关键）。
    """

    lost_buffer: int = 30
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    detector: PersonDetector | None = field(default=None, repr=False)

    def _ensure_model(self) -> Any:
        if self.detector is None:
            raise RuntimeError(
                "ByteTrackTracker 需要先与 PersonDetector 关联："
                "在 DetectionPipeline 中把 detector 传入 tracker。"
            )
        return self.detector.load()

    def update(self, frame_bgr: cv2.Mat, detections: list[Detection]) -> list[TrackedObject]:
        """更新轨迹并返回带 track_id 的目标列表。

        track() 在内部完成 person 检测 + ByteTrack 关联，
        detections 参数保留用于未来接入独立检测器（如 RTMDet）的扩展位。
        """
        model = self._ensure_model()
        results = model.track(
            frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.detector.conf_thres,
            iou=self.detector.iou_thres,
            imgsz=self.detector.imgsz,
            device=self.detector.device or None,
            classes=[0],
            verbose=False,
        )
        tracked: list[TrackedObject] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for tid, box, conf in zip(ids, xyxy, confs):
                # ultralytics 无公开 is_confirmed 标志，简化：高分框视为确认轨迹
                tracked.append(
                    TrackedObject(
                        track_id=int(tid),
                        xyxy=box.astype(np.float32),
                        confidence=float(conf),
                        is_confirmed=bool(conf >= self.track_high_thresh),
                    )
                )
        return tracked
