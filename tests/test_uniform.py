"""uniform.py 纯逻辑单测（不依赖模型/视频）。"""

from __future__ import annotations

import numpy as np

from src.detector import Detection
from src.uniform import UniformClassifier


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
    assert res.roi[0] >= 0 and res.roi[1] >= 0
    assert res.roi[2] <= 50 and res.roi[3] <= 50


def test_invalid_roi_returns_false() -> None:
    # 检测框过小导致 ROI 为空时，安全返回 False 而非崩溃
    frame = _make_frame(100, 100, (0, 0, 255))
    det = Detection(xyxy=np.array([50, 50, 50, 50]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.is_uniform is False
    assert res.red_ratio == 0.0
