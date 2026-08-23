"""StaffRecognition - S4 掩码工装规则单元测试。"""

from __future__ import annotations

import numpy as np

from src.uniform import UniformClassifier


class _FakeDet:
    """带人体掩码的测试目标（模拟 pipeline FusedDetection）。"""

    def __init__(self, xyxy, mask):
        self.xyxy = np.array(xyxy, dtype=float)
        self.mask = mask


def _frame(width: int, height: int, bgr) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _mask_from_roi(h: int, w: int, roi) -> np.ndarray:
    """在 ROI 内生成人体掩码（覆盖整个框，模拟可见部分充足）。"""
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = roi
    mask[y1:y2, x1:x2] = True
    return mask


def test_red_uniform_detected_via_mask() -> None:
    # 检测框内人体掩码覆盖区域为红色 -> 判穿
    frame = _frame(100, 200, (255, 0, 0))          # 全蓝背景
    frame[0:120, 0:100] = (0, 0, 255)              # 人体区域红色
    mask = _mask_from_roi(200, 100, (0, 0, 100, 120))
    det = _FakeDet([0, 0, 100, 200], mask)
    res = UniformClassifier().classify_masks(frame, [det])[0]
    assert res.is_uniform is True
    assert res.red_ratio > 0.8
    assert res.red_ratio_connected > 0.3
    assert res.valid is True


def test_blue_not_uniform_via_mask() -> None:
    # 人体掩码覆盖区域为蓝色 -> 判未穿
    frame = _frame(100, 200, (255, 0, 0))          # 全蓝
    mask = _mask_from_roi(200, 100, (0, 0, 100, 200))
    det = _FakeDet([0, 0, 100, 200], mask)
    res = UniformClassifier().classify_masks(frame, [det])[0]
    assert res.is_uniform is False
    assert res.red_ratio < 0.01
    assert res.valid is True


def test_background_red_ignored_when_mask_clean() -> None:
    # 背景红色（掩码外）不影响判定：掩码内蓝色 -> 未穿
    frame = _frame(100, 200, (0, 0, 255))          # 全红背景
    frame[0:100, 0:100] = (255, 0, 0)              # 人体区域蓝色
    mask = _mask_from_roi(200, 100, (0, 0, 100, 100))
    det = _FakeDet([0, 0, 100, 200], mask)
    res = UniformClassifier().classify_masks(frame, [det])[0]
    assert res.is_uniform is False
    assert res.baseline_ratio >= 0.5               # 背景确为红色


def test_missing_mask_marks_invalid() -> None:
    # 无掩码 -> valid=False，S6 维持前态
    frame = _frame(100, 200, (0, 0, 255))
    det = _FakeDet([0, 0, 100, 200], None)
    res = UniformClassifier().classify_masks(frame, [det])[0]
    assert res.valid is False


def test_tiny_mask_marks_invalid() -> None:
    # 掩码面积过小（证据不足）-> valid=False
    frame = _frame(100, 200, (0, 0, 255))
    mask = np.zeros((200, 100), dtype=bool)
    mask[0:2, 0:2] = True                          # 仅 4 像素
    det = _FakeDet([0, 0, 100, 200], mask)
    res = UniformClassifier().classify_masks(frame, [det])[0]
    assert res.valid is False


def test_red_ratio_strong_fallback() -> None:
    # 红色占比极高（>=0.30）即使连通块略低也判穿
    frame = _frame(100, 200, (0, 0, 255))
    mask = _mask_from_roi(200, 100, (0, 0, 100, 120))
    det = _FakeDet([0, 0, 100, 200], mask)
    clf = UniformClassifier(red_ratio_threshold=0.5, connected_min_ratio=0.5)
    res = clf.classify_masks(frame, [det])[0]
    assert res.is_uniform is True                   # red_ratio=1.0 >= 0.30 兜底
