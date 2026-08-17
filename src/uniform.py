"""工装识别：上半身 ROI 裁剪 + HSV 红色判定（红色背心/短袖）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection


@dataclass
class UniformResult:
    """单个目标的工装判定结果。"""

    is_uniform: bool
    red_ratio: float        # ROI 内红色像素占比
    roi: tuple[int, int, int, int]  # 实际裁剪的上半身 ROI (x1, y1, x2, y2)


@dataclass
class UniformClassifier:
    """基于 HSV 颜色规则的红色工装分类器。

    现场标定路径：scripts/calibrate.py 调整 HSV 阈值与 ROI 区间。
    """

    roi: tuple[float, float, float, float] = (0.10, 0.20, 0.90, 0.60)  # x_begin, y_begin, x_end, y_end
    hsv_ranges: tuple[tuple[int, int], tuple[int, int]] = ((0, 10), (170, 180))
    s_min: int = 80
    v_min: int = 80
    red_ratio_threshold: float = 0.10
    normalize: bool = True

    @staticmethod
    def _clip_roi(
        frame_h: int, frame_w: int, box: np.ndarray, roi: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        """把相对 ROI 换算成检测框内像素坐标并裁剪到画面内。"""
        x1, y1, x2, y2 = box
        bx, by, bw, bh = x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)
        rx1 = int(round(bx + roi[0] * bw))
        ry1 = int(round(by + roi[1] * bh))
        rx2 = int(round(bx + roi[2] * bw))
        ry2 = int(round(by + roi[3] * bh))
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(frame_w, rx2), min(frame_h, ry2)
        if rx2 <= rx1 or ry2 <= ry1:
            return (rx1, ry1, rx2, ry2)
        return (rx1, ry1, rx2, ry2)

    @staticmethod
    def _is_red_pixel(h: int, s: int, v: int, ranges, s_min: int, v_min: int) -> bool:
        """HSV 单像素红色判定（OpenCV H 0-180，红色在 0-10 ∪ 170-180）。"""
        if s < s_min or v < v_min:
            return False
        (h1_begin, h1_end), (h2_begin, h2_end) = ranges
        return (h1_begin <= h <= h1_end) or (h2_begin <= h <= h2_end)

    def classify(self, frame_bgr: cv2.Mat, det: Detection) -> UniformResult:
        """对检测框裁剪上半身 ROI，返回红色占比与判定结果。"""
        frame_h, frame_w = frame_bgr.shape[:2]
        roi_px = self._clip_roi(frame_h, frame_w, det.xyxy, self.roi)
        rx1, ry1, rx2, ry2 = roi_px
        if rx2 <= rx1 or ry2 <= ry1:
            return UniformResult(is_uniform=False, red_ratio=0.0, roi=roi_px)

        roi_img = frame_bgr[ry1:ry2, rx1:rx2]
        if self.normalize:
            # 光照归一化：LAB 的 L 通道直方图均衡化，降低明暗影响
            lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
            l_plane, a_plane, b_plane = cv2.split(lab)
            l_plane = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_plane)
            lab = cv2.merge((l_plane, a_plane, b_plane))
            roi_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # 向量化：避免逐像素 Python 循环（性能关键路径）
        red_mask = (
            ((h >= self.hsv_ranges[0][0]) & (h <= self.hsv_ranges[0][1]))
            | ((h >= self.hsv_ranges[1][0]) & (h <= self.hsv_ranges[1][1]))
        ) & (s >= self.s_min) & (v >= self.v_min)
        red_ratio = float(red_mask.mean())
        is_uniform = red_ratio >= self.red_ratio_threshold
        return UniformResult(is_uniform=is_uniform, red_ratio=red_ratio, roi=roi_px)
