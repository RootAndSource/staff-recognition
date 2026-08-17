"""uniform.py 纯逻辑单测（不依赖模型/视频）。"""

from __future__ import annotations

import numpy as np

from src.detector import Detection
from src.uniform import UniformClassifier, _RoiSpec


def _make_frame(width: int, height: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_red_uniform_detected() -> None:
    # 纯红色画面（BGR 红 = (0, 0, 255)）应判为工装
    frame = _make_frame(100, 200, (0, 0, 255))
    det = Detection(xyxy=np.array([0, 0, 100, 200]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.is_uniform is True
    assert res.red_ratio > 0.9


def test_blue_not_uniform() -> None:
    # 纯蓝色画面不应误判
    frame = _make_frame(100, 200, (255, 0, 0))
    det = Detection(xyxy=np.array([0, 0, 100, 200]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.is_uniform is False
    assert res.red_ratio < 0.01


def test_roi_clipping_out_of_frame() -> None:
    # 检测框超出画面时 ROI 裁剪不应崩溃
    frame = _make_frame(50, 50, (0, 0, 0))
    det = Detection(xyxy=np.array([-10, -10, 200, 200]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.roi_main[0] >= 0 and res.roi_main[1] >= 0
    assert res.roi_main[2] <= 50 and res.roi_main[3] <= 50
    assert res.roi_lower[0] >= 0 and res.roi_lower[1] >= 0
    assert res.roi_lower[2] <= 50 and res.roi_lower[3] <= 50


def test_invalid_roi_returns_false() -> None:
    # 检测框过小导致 ROI 为空时，安全返回 False 而非崩溃
    frame = _make_frame(100, 100, (0, 0, 255))
    det = Detection(xyxy=np.array([50, 50, 50, 50]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.is_uniform is False
    assert res.red_ratio == 0.0


def test_dual_roi_any_band_wins() -> None:
    # 主 ROI 蓝色、兜底 ROI 内红色集中：任一达标即判工装
    frame = _make_frame(100, 200, (255, 0, 0))  # 全蓝
    frame[100:160, 0:100] = (0, 0, 255)         # 红色涂在兜底 ROI(y 90-170)内部
    det = Detection(xyxy=np.array([0, 0, 100, 200]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.red_ratio_lower > 0.4
    assert res.is_uniform is True
