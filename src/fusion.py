"""检测融合：上半身主检 + 全身兜底的"可见部分框"融合（v0.3 拍板）。

策略：
- full 框与 upper 框位置匹配（upper 中心落在 full 上部区域内）→ 同一人，保留 upper 为主框；
- 仅 upper / 仅 full → 直接保留；
- 同类框间做 NMS 合并；
- 输出每人一个主框 + box_type（upper/full）+ 推导的上半身 ROI（工装输入）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusedDetection:
    xyxy: np.ndarray
    confidence: float
    box_type: str                 # "upper" / "full"
    upper_roi: tuple[int, int, int, int]  # 推导的上半身 ROI（像素）


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    return inter / max(area_a + area_b - inter, 1e-6)


def _upper_roi_from_full(box: np.ndarray, frame_w: int, frame_h: int,
                         ratio: float = 0.55) -> tuple[int, int, int, int]:
    """从全身框推导上半身 ROI（顶部到顶部 + ratio*高度）。"""
    x1, y1, x2, y2 = box
    by2 = y1 + ratio * (y2 - y1)
    return (
        int(max(0, x1)), int(max(0, y1)),
        int(min(frame_w, x2)), int(min(frame_h, by2)),
    )


class DetectionFusion:
    """上半身/全身框融合。"""

    def __init__(
        self,
        *,
        upper_inside_ratio: float = 0.6,   # upper 中心需落在 full 上部该比例内
        upper_iou_thresh: float = 0.15,    # upper 与 full 的 IoU 匹配阈值
        same_class_nms: float = 0.5,       # 同类框 NMS 阈值
        upper_ratio: float = 0.55,         # full → upper ROI 高度比例
    ) -> None:
        self.upper_inside_ratio = upper_inside_ratio
        self.upper_iou_thresh = upper_iou_thresh
        self.same_class_nms = same_class_nms
        self.upper_ratio = upper_ratio

    @staticmethod
    def _nms(boxes, confs, iou_thresh: float):
        """按置信度降序贪心 NMS。"""
        order = np.argsort(-np.asarray(confs))
        kept = []
        boxes = list(boxes)
        while order.size:
            i = order[0]
            kept.append(i)
            rest = order[1:]
            if rest.size == 0:
                break
            ious = np.array([_iou(boxes[i], boxes[j]) for j in rest])
            order = rest[ious <= iou_thresh]
        return kept

    def fuse(
        self,
        upper_dets: list[tuple[np.ndarray, float]],
        full_dets: list[tuple[np.ndarray, float]],
        frame_w: int,
        frame_h: int,
    ) -> list[FusedDetection]:
        """融合两类检测框，输出每人一个主框。"""
        # 1. 同类 NMS
        upper_kept = self._nms([d[0] for d in upper_dets], [d[1] for d in upper_dets], self.same_class_nms)
        full_kept = self._nms([d[0] for d in full_dets], [d[1] for d in full_dets], self.same_class_nms)
        uppers = [upper_dets[i] for i in upper_kept]
        fulls = [full_dets[i] for i in full_kept]

        used_full = [False] * len(fulls)
        results: list[FusedDetection] = []

        # 2. upper 优先：与 full 匹配 → 用 upper
        for ubox, uconf in uppers:
            uc = np.array([(ubox[0] + ubox[2]) / 2, (ubox[1] + ubox[3]) / 2])
            best = None
            best_iou = 0.0
            for j, (fbox, _) in enumerate(fulls):
                if used_full[j]:
                    continue
                fb = np.array(fbox)
                # upper 中心需落在 full 上部区域
                in_upper_zone = (
                    fb[0] <= uc[0] <= fb[2]
                    and fb[1] <= uc[1] <= fb[1] + self.upper_inside_ratio * (fb[3] - fb[1])
                )
                iou = _iou(ubox, fbox)
                if in_upper_zone and iou >= self.upper_iou_thresh and iou > best_iou:
                    best, best_iou = j, iou
            if best is not None:
                used_full[best] = True
            results.append(
                FusedDetection(
                    xyxy=np.array(ubox, dtype=float),
                    confidence=uconf,
                    box_type="upper",
                    upper_roi=_upper_roi_from_full(ubox, frame_w, frame_h, 1.0),
                )
            )

        # 3. 未匹配的 full 兜底保留
        for j, (fbox, fconf) in enumerate(fulls):
            if not used_full[j]:
                results.append(
                    FusedDetection(
                        xyxy=np.array(fbox, dtype=float),
                        confidence=fconf,
                        box_type="full",
                        upper_roi=_upper_roi_from_full(fbox, frame_w, frame_h, self.upper_ratio),
                    )
                )

        results.sort(key=lambda r: -r.confidence)
        return results
