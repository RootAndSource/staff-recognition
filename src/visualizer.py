"""StaffRecognition - 画面叠加：检测框 + 掩码 + 工装标签。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .identity import PersonObject

COLOR_UNIFORM = (0, 200, 0)      # BGR 绿色：穿工装
COLOR_NO_UNIFORM = (0, 0, 255)   # BGR 红色：未穿工装


_FONT_PATH = None
for _candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(_candidate).exists():
        _FONT_PATH = _candidate
        break


def _draw_cv_text(
    frame_bgr: cv2.Mat,
    text: str,
    org: tuple[int, int],
    font_size: int = 18,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """用 PIL 在 OpenCV 帧上绘制中文文本（cv2.putText 不支持中文）。

    只转换标签所在的局部区域，避免整帧 PIL 往返的开销。
    org 为文本左上角 (x, y)。
    """
    if not text or _FONT_PATH is None:
        return
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(_FONT_PATH, font_size)
    h, w = frame_bgr.shape[:2]
    x0, y0 = int(org[0]), int(org[1])
    x1, y1 = x0 + font_size * len(text) + 10, y0 + font_size + 6
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return

    patch = frame_bgr[y0:y1, x0:x1]
    img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    draw.text((2, 1), text, font=font, fill=bgr,
              stroke_width=1, stroke_fill=(0, 0, 0))
    frame_bgr[y0:y1, x0:x1] = cv2.cvtColor(
        np.asarray(img), cv2.COLOR_RGB2BGR
    )


def draw_tracked(frame_bgr, person: PersonObject) -> None:
    """在帧上就地绘制单个人员的框与标签（显示置信度，不显示 ID）。"""
    x1, y1, x2, y2 = [int(v) for v in person.xyxy]
    if person.uniform is None:
        color = (200, 200, 200)  # 灰色：待定
        label = f"{person.confidence:.2f} 待定"
    else:
        color = COLOR_UNIFORM if person.uniform else COLOR_NO_UNIFORM
        label = f"{person.confidence:.2f} {'工装' if person.uniform else '未工装'}"
    mask = getattr(person, "mask", None)
    if mask is not None:
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        overlay = frame_bgr.copy()
        colored = np.zeros_like(frame_bgr)
        colored[:, :, 1] = 120
        sel = mask_u8 > 0
        overlay[sel] = cv2.addWeighted(overlay[sel], 0.55, colored[sel], 0.45, 0)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        frame_bgr[:] = overlay
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    bar_h = 24
    bar_w = 18 * len(label) + 12
    top = y1 - bar_h - 4
    cv2.rectangle(frame_bgr, (x1, top), (x1 + bar_w, y1 - 2), color, -1)
    _draw_cv_text(frame_bgr, label, (x1 + 4, top + 1), font_size=16)


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
