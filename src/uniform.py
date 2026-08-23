"""StaffRecognition - S4 工装识别：SAM2 人体掩码内的工装色规则（免训练）。
最终方案：在 S2 输出的可见人体掩码内统计红色占比与最大红色连通块占比，
排除桌面/墙壁等背景干扰；任一块见工装色即判穿（宽松），全部无工装色且
可见面积足够才判未穿（谨慎），掩码缺失/过小则标记 valid=False 由 S6 维持前态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class UniformResult:
    """单个目标的工装判定结果。"""

    is_uniform: bool
    red_ratio: float            # 掩码内红色像素占比
    red_ratio_main: float       # 兼容字段：等于 red_ratio
    red_ratio_lower: float      # 兼容字段：等于 red_ratio
    red_ratio_connected: float  # 掩码内最大红色连通块占比（工装成块先验）
    baseline_ratio: float       # 检测框内（含背景）红色占比，用于诊断
    roi_main: tuple[int, int, int, int]
    roi_lower: tuple[int, int, int, int]
    uniform_prob: float = 0.5   # 工装证据强度 0~1
    valid: bool = True          # False=证据不足，S6 不更新状态


@dataclass
class UniformClassifier:
    """基于 SAM2 人体掩码的工装色规则分类器（免训练）。

    判定规则：
    - 有工装色：red_ratio >= red_ratio_threshold 且 connected >= connected_min_ratio，
      或 red_ratio >= 0.30（强红色兜底）；
    - 未穿：全部可见块无工装色，且可见面积 >= min_visible_area_ratio；
    - 证据不足：掩码缺失或可见面积过小 -> valid=False。
    """

    hsv_ranges: tuple[tuple[int, int], tuple[int, int]] = ((0, 10), (170, 180))
    s_min: int = 70
    v_min: int = 70
    red_ratio_threshold: float = 0.08
    connected_min_ratio: float = 0.05
    min_visible_area_ratio: float = 0.08  # 判定"未穿"所需最小可见面积（相对检测框）
    @staticmethod
    def _largest_component_ratio(red_mask: np.ndarray, area: int) -> float:
        """最大红色连通块面积 / 给定面积（人体掩码面积），工装成块先验。"""
        if area <= 0 or not red_mask.any():
            return 0.0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            red_mask.astype(np.uint8), connectivity=8
        )
        if num <= 1:
            return 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return float(areas.max()) / area

    def classify_masks(self, frame_bgr: cv2.Mat, dets: list[Any]) -> list[UniformResult]:
        """对每个带人体掩码的目标做工装判定（S4 主路径）。

        dets 需具备 .xyxy（检测框）与 .mask（HxW bool 人体掩码）。
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red_full = (
            ((h >= self.hsv_ranges[0][0]) & (h <= self.hsv_ranges[0][1]))
            | ((h >= self.hsv_ranges[1][0]) & (h <= self.hsv_ranges[1][1]))
        ) & (s >= self.s_min) & (v >= self.v_min)

        results: list[UniformResult] = []
        for d in dets:
            mask = getattr(d, "mask", None)
            x1, y1, x2, y2 = [int(v) for v in d.xyxy]
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            box_area = bw * bh
            if mask is None or mask.sum() == 0:
                results.append(self._make_result(False, 0.0, 0.0, 0.0, (x1, y1, x2, y2), valid=False))
                continue

            mask_bool = mask > 0.5 if mask.dtype != np.bool_ else mask
            visible_ratio = float(mask_bool.sum()) / max(box_area, 1)
            # 可见面积过小 -> 证据不足，直接 valid=False（避免小掩码误判）
            if visible_ratio < self.min_visible_area_ratio:
                results.append(
                    self._make_result(False, 0.0, 0.0, 0.0, (x1, y1, x2, y2), 0.5, valid=False)
                )
                continue
            red_in_mask = red_full & mask_bool
            red_ratio = float(red_in_mask.sum()) / max(mask_bool.sum(), 1)
            connected = self._largest_component_ratio(red_in_mask, int(mask_bool.sum()))
            box_px = (max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2))
            box_red = red_full[box_px[1]:box_px[3], box_px[0]:box_px[2]]
            baseline = float(box_red.mean()) if box_red.size else 0.0

            has_uniform_color = (
                red_ratio >= self.red_ratio_threshold
                and connected >= self.connected_min_ratio
            ) or red_ratio >= 0.30
            if has_uniform_color:
                evidence = min(1.0, red_ratio * 2.0 + connected * 2.0)
                results.append(
                    self._make_result(True, red_ratio, connected, baseline, (x1, y1, x2, y2), evidence)
                )
                continue

            results.append(
                self._make_result(False, red_ratio, connected, baseline, (x1, y1, x2, y2), 0.0)
            )
        return results

    @staticmethod
    def _make_result(
        is_uniform: bool,
        red_ratio: float,
        connected: float,
        baseline: float,
        box: tuple[int, int, int, int],
        prob: float = 0.5,
        valid: bool = True,
    ) -> UniformResult:
        return UniformResult(
            is_uniform=is_uniform,
            red_ratio=red_ratio,
            red_ratio_main=red_ratio,
            red_ratio_lower=red_ratio,
            red_ratio_connected=connected,
            baseline_ratio=baseline,
            roi_main=box,
            roi_lower=box,
            uniform_prob=round(prob, 4),
            valid=valid,
        )