"""视频输入：本地视频文件或 RTSP 流读取、降采样、丢帧。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass
class FrameInfo:
    """单帧处理结果。"""

    frame: cv2.Mat          # 降采样后的 BGR 帧
    frame_id: int           # 输入流中的原始帧号
    timestamp_ms: float     # 采集时间戳（毫秒）
    width: int              # 原始宽度
    height: int             # 原始高度
    fps: float              # 输入流帧率


class VideoSource:
    """封装 OpenCV VideoCapture：支持文件与 RTSP，统一输出降采样帧。"""

    def __init__(
        self,
        source: str,
        max_width: int = 1280,
        max_height: int = 720,
        rtsp_timeout_ms: int = 5000,
    ) -> None:
        self.source = source
        self.max_width = max_width
        self.max_height = max_height

        if source.startswith(("rtsp://", "http://", "https://")):
            # RTSP：优先使用 ffmpeg 后端，降低延迟
            backend = cv2.CAP_FFMPEG
            cv2.VideoCapture.setExceptionMode(True)
        else:
            backend = cv2.CAP_ANY

        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频源: {source}")

        self.raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._scale = self._compute_scale()

    def _compute_scale(self) -> float:
        """按 max_width/max_height 计算等比降采样系数。"""
        if self.raw_width <= 0 or self.raw_height <= 0:
            return 1.0
        scale = min(self.max_width / self.raw_width, self.max_height / self.raw_height, 1.0)
        return scale

    def frames(self, skip: int = 0) -> Iterator[FrameInfo]:
        """逐帧产出 FrameInfo；skip>0 时跳过中间帧（丢帧策略）。"""
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if frame_id % (skip + 1) != 0:
                frame_id += 1
                continue
            if self._scale < 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=self._scale,
                    fy=self._scale,
                    interpolation=cv2.INTER_AREA,
                )
            yield FrameInfo(
                frame=frame,
                frame_id=frame_id,
                timestamp_ms=frame_id * 1000.0 / self.fps if self.fps > 0 else 0.0,
                width=self.raw_width,
                height=self.raw_height,
                fps=self.fps,
            )
            frame_id += 1

    def release(self) -> None:
        """释放视频资源（技能约定：确保 release 被调用）。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
