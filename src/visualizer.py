"""实时画面叠加：检测框 + 置信度 + 工装标签。"""

from __future__ import annotations

import cv2

from .identity import PersonObject

COLOR_UNIFORM = (0, 200, 0)      # BGR 绿色：穿工装
COLOR_NO_UNIFORM = (0, 0, 255)   # BGR 红色：未穿工装


def draw_tracked(frame_bgr, person: PersonObject) -> None:
    """在帧上就地绘制单个人员的框与标签（显示置信度，不显示 ID）。"""
    x1, y1, x2, y2 = [int(v) for v in person.xyxy]
    if person.uniform is None:
        color = (200, 200, 200)  # 灰色：待定
        label = f"{person.confidence:.2f} 待定"
    else:
        color = COLOR_UNIFORM if person.uniform else COLOR_NO_UNIFORM
        label = f"{person.confidence:.2f} {'工装' if person.uniform else '未工装'}"
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 4, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_stats(frame_bgr, fps: float, person_count: int) -> None:
    """绘制帧率与人数统计。"""
    text = f"FPS {fps:.1f} | People {person_count}"
    cv2.putText(
        frame_bgr,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
