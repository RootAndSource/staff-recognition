"""多目标跟踪：ByteTrack 封装，维护跨帧 track_id。

设计为可替换接口：新增跟踪器时实现同一 update 签名即可
（预留 OC-SORT / StrongSORT 切换位）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

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
    merge_overlap_iou: float = 0.3   # 同帧重复框合并阈值（趴姿多框场景）
    tracker_yaml: str | None = field(default=None, repr=False)  # 自定义 ByteTrack 配置

    @staticmethod
    def _merge_overlapping(tracked: list[TrackedObject], iou_thres: float) -> list[TrackedObject]:
        """合并同一帧内高度重叠的检测框，保留高置信度者。

        背景：斜俯视/趴姿下 YOLO 常对同一人输出“上半身小框 + 全身大框”，
        IoU 不足 NMS 阈值而被当成两个目标。这里按置信度降序贪心合并。
        """
        if len(tracked) < 2:
            return tracked
        ordered = sorted(tracked, key=lambda o: -o.confidence)
        kept: list[TrackedObject] = []
        for obj in ordered:
            overlap = False
            for k in kept:
                a, b = obj.xyxy, k.xyxy
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                x2, y2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
                area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
                iou = inter / max(area_a + area_b - inter, 1e-6)
                if iou >= iou_thres:
                    overlap = True
                    break
            if not overlap:
                kept.append(obj)
        return kept

    def _ensure_model(self) -> Any:
        if self.detector is None:
            raise RuntimeError(
                "ByteTrackTracker 需要先与 PersonDetector 关联："
                "在 DetectionPipeline 中把 detector 传入 tracker。"
            )
        return self.detector.load()

    @staticmethod
    def write_tracker_yaml(
        path: str | Path,
        *,
        track_buffer: int,
        track_high_thresh: float,
        track_low_thresh: float,
        new_track_thresh: float,
        tracker_type: str = "bytetrack",
        with_reid: bool = False,
        appearance_thresh: float = 0.8,
    ) -> Path:
        """写出自定义跟踪器配置（track_buffer 等参数真正生效）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "tracker_type": tracker_type,
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": 0.8,
            "fuse_score": True,
        }
        if tracker_type == "botsort":
            cfg.update(
                {
                    "gmc_method": "none",  # 固定机位，关闭相机运动补偿
                    "proximity_thresh": 0.5,
                    "appearance_thresh": appearance_thresh,
                    "with_reid": with_reid,
                    "model": "auto",
                }
            )
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return p

    def update(self, frame_bgr: cv2.Mat, detections: list[Detection]) -> list[TrackedObject]:
        """更新轨迹并返回带 track_id 的目标列表。

        track() 在内部完成 person 检测 + ByteTrack 关联，
        detections 参数保留用于未来接入独立检测器（如 RTMDet）的扩展位。
        """
        model = self._ensure_model()
        results = model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker_yaml or "bytetrack.yaml",
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
        return self._merge_overlapping(tracked, self.merge_overlap_iou)
