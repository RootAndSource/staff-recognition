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
    red_ratio: float        # 主 ROI 红色像素占比（兼容旧字段，取主段）
    red_ratio_main: float   # 主 ROI（肩胸）红色占比
    red_ratio_lower: float  # 兜底 ROI（腹腰）红色占比
    roi_main: tuple[int, int, int, int]   # 主 ROI 像素坐标
    roi_lower: tuple[int, int, int, int]  # 兜底 ROI 像素坐标


@dataclass
class _RoiSpec:
    """相对 ROI 定义（0-1 区间）。"""

    x_begin: float
    y_begin: float
    x_end: float
    y_end: float


@dataclass
class UniformClassifier:
    """基于 HSV 颜色规则的红色工装分类器。

    双段 ROI：主段（肩胸）+ 兜底段（腹腰），任一段红色占比达标即判工装，
    用于缓解斜俯视/桌子遮挡导致躯干落在单一 ROI 之外的问题。
    """

    roi_main: _RoiSpec = _RoiSpec(0.10, 0.15, 0.90, 0.55)
    roi_lower: _RoiSpec = _RoiSpec(0.10, 0.45, 0.90, 0.85)
    hsv_ranges: tuple[tuple[int, int], tuple[int, int]] = ((0, 10), (170, 180))
    s_min: int = 70
    v_min: int = 70
    red_ratio_threshold: float = 0.08
    normalize: bool = True

    @staticmethod
    def _clip_roi(
        frame_h: int, frame_w: int, box: np.ndarray, roi: _RoiSpec
    ) -> tuple[int, int, int, int]:
        """把相对 ROI 换算成检测框内像素坐标并裁剪到画面内。"""
        x1, y1, x2, y2 = box
        bx, by, bw, bh = x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)
        rx1 = int(round(bx + roi.x_begin * bw))
        ry1 = int(round(by + roi.y_begin * bh))
        rx2 = int(round(bx + roi.x_end * bw))
        ry2 = int(round(by + roi.y_end * bh))
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(frame_w, rx2), min(frame_h, ry2)
        return (rx1, ry1, rx2, ry2)

    def _roi_red_ratio(self, frame_bgr: cv2.Mat, roi_px: tuple[int, int, int, int]) -> float:
        """计算单个 ROI 内的红色像素占比；ROI 为空时返回 0.0。"""
        rx1, ry1, rx2, ry2 = roi_px
        if rx2 <= rx1 or ry2 <= ry1:
            return 0.0
        roi_img = frame_bgr[ry1:ry2, rx1:rx2]
        if self.normalize:
            # 光照归一化：LAB 的 L 通道 CLAHE，降低明暗影响
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
        return float(red_mask.mean())

    def classify(self, frame_bgr: cv2.Mat, det: Detection) -> UniformResult:
        """对检测框裁剪主/兜底双 ROI，任一红色占比达标即判工装。"""
        frame_h, frame_w = frame_bgr.shape[:2]
        roi_main = self._clip_roi(frame_h, frame_w, det.xyxy, self.roi_main)
        roi_lower = self._clip_roi(frame_h, frame_w, det.xyxy, self.roi_lower)
        ratio_main = self._roi_red_ratio(frame_bgr, roi_main)
        ratio_lower = self._roi_red_ratio(frame_bgr, roi_lower)
        is_uniform = (
            ratio_main >= self.red_ratio_threshold
            or ratio_lower >= self.red_ratio_threshold
        )
        return UniformResult(
            is_uniform=is_uniform,
            red_ratio=ratio_main,
            red_ratio_main=ratio_main,
            red_ratio_lower=ratio_lower,
            roi_main=roi_main,
            roi_lower=roi_lower,
        )
