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
    # 检测框内：上部红色（工装）、下部蓝色（裤子）→ 应判为工装
    frame = _make_frame(100, 200, (255, 0, 0))  # 全蓝
    frame[0:120, 0:100] = (0, 0, 255)           # 上部（含主/兜底 ROI）红色
    det = Detection(xyxy=np.array([0, 0, 100, 200]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.is_uniform is True
    assert res.red_ratio > 0.8
    assert res.red_ratio_connected > 0.3
    assert res.baseline_ratio < 0.7


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
    # 主 ROI 蓝色、兜底 ROI 内红色集中成块：判为工装
    frame = _make_frame(100, 200, (255, 0, 0))  # 全蓝
    frame[100:160, 0:100] = (0, 0, 255)         # 红色涂在兜底 ROI(y 90-170)内部
    det = Detection(xyxy=np.array([0, 0, 100, 200]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.red_ratio_lower > 0.4
    assert res.red_ratio_connected > 0.5
    assert res.is_uniform is True


def test_scattered_red_not_uniform() -> None:
    # 红色零散分布（背景/桌面干扰）→ 最大连通域占比低，不应判工装
    frame = _make_frame(100, 200, (255, 0, 0))
    rng = np.random.default_rng(42)
    for _ in range(200):
        x, y = rng.integers(0, 100), rng.integers(0, 200)
        frame[y:y + 2, x:x + 2] = (0, 0, 255)
    det = Detection(xyxy=np.array([0, 0, 100, 200]), confidence=1.0)
    res = UniformClassifier().classify(frame, det)
    assert res.red_ratio_connected < 0.05
    assert res.is_uniform is False
