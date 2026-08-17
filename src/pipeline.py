"""主处理管线：逐帧 检测→跟踪→工装识别→输出 的串联。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import load_config
from .detector import PersonDetector
from .output import JsonOutputWriter, build_person_record
from .tracker import ByteTrackTracker
from .uniform import UniformClassifier
from .video_input import VideoSource
from .visualizer import draw_stats, draw_tracked


@dataclass
class PipelineStats:
    """运行统计（用于控制台与日志）。"""

    frames_processed: int = 0
    persons_total: int = 0          # 累计出现的人次（含重复帧）
    uniform_count: int = 0          # 累计工装判定人次
    frames_fps_window: list[float] = field(default_factory=list)


class DetectionPipeline:
    """任务 A 主管线。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.max_frames: int = 0  # 0=不限；>0 表示最多处理 N 帧（调试/测试用）
        vcfg, mcfg, tcfg, ucfg, ocfg = (
            self.config["video"],
            self.config["model"],
            self.config["tracker"],
            self.config["uniform"],
            self.config["output"],
        )
        self.video_cfg = vcfg
        self.output_cfg = ocfg
        self.detector = PersonDetector(
            weights=mcfg["weights"],
            conf_thres=mcfg["conf_thres"],
            iou_thres=mcfg["iou_thres"],
            imgsz=mcfg["imgsz"],
            device=mcfg.get("device", ""),
        )
        self.tracker = ByteTrackTracker(
            lost_buffer=tcfg["lost_buffer"],
            track_high_thresh=tcfg["track_high_thresh"],
            track_low_thresh=tcfg["track_low_thresh"],
            new_track_thresh=tcfg["new_track_thresh"],
            detector=self.detector,   # 共享 YOLO 实例，避免重复推理
        )
        self.uniform = UniformClassifier(
            roi=(ucfg["roi"]["x_begin"], ucfg["roi"]["y_begin"], ucfg["roi"]["x_end"], ucfg["roi"]["y_end"]),
            hsv_ranges=((ucfg["hsv_red"]["h1_begin"], ucfg["hsv_red"]["h1_end"]),
                        (ucfg["hsv_red"]["h2_begin"], ucfg["hsv_red"]["h2_end"])),
            s_min=ucfg["hsv_red"]["s_min"],
            v_min=ucfg["hsv_red"]["v_min"],
            red_ratio_threshold=ucfg["red_ratio_threshold"],
            normalize=ucfg.get("normalize", True),
        )
        # 启动时加载模型，加载失败尽早暴露（而非逐帧崩溃）
        self.detector.load()

    def _process_frame(
        self,
        frame_bgr: cv2.Mat,
        frame_id: int,
        timestamp_ms: float,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """单帧处理：返回 (json_records, 人数, 工装人数)。"""
        # ByteTrack 的 track() 内部已包含检测，不重复调用 detect()
        tracked = self.tracker.update(frame_bgr, detections=[])
        records: list[dict[str, Any]] = []
        uniform_count = 0
        for obj in tracked:
            if not obj.is_confirmed:
                continue  # 丢弃未确认轨迹，避免冷启动闪烁
            uni = self.uniform.classify(frame_bgr, obj)
            if uni.is_uniform:
                uniform_count += 1
            if self.output_cfg["show"]:
                draw_tracked(frame_bgr, obj, uni)
            records.append(build_person_record(obj, uni))
        return records, len(records), uniform_count

    def run(
        self,
        source: str,
        json_path: str | Path | None = None,
        save_video: str | Path | None = None,
    ) -> PipelineStats:
        """运行主循环；Ctrl+C / 视频结束 / 窗口关闭时退出。"""
        stats = PipelineStats()
        json_writer = JsonOutputWriter(json_path) if json_path else None
        video_writer: cv2.VideoWriter | None = None

        with VideoSource(
            source,
            max_width=self.video_cfg["max_width"],
            max_height=self.video_cfg["max_height"],
            rtsp_timeout_ms=self.video_cfg.get("rtsp_timeout_ms", 5000),
        ) as vs:
            skip = int(self.video_cfg.get("skip_frames", 0))
            for info in vs.frames(skip=skip):
                if self.max_frames and stats.frames_processed >= self.max_frames:
                    break
                frame = info.frame
                if video_writer is None and save_video:
                    h, w = frame.shape[:2]
                    out_fps = self.output_cfg["video_fps"] or info.fps or 20.0
                    video_writer = cv2.VideoWriter(
                        str(save_video),
                        cv2.VideoWriter_fourcc(*self.output_cfg["video_codec"]),
                        out_fps,
                        (w, h),
                    )

                records, person_count, uniform_count = self._process_frame(
                    frame, info.frame_id, info.timestamp_ms
                )
                stats.frames_processed += 1
                stats.persons_total += person_count
                stats.uniform_count += uniform_count

                if json_writer and stats.frames_processed % self.output_cfg["json_every_n_frames"] == 0:
                    json_writer.write_frame(
                        frame_id=info.frame_id,
                        timestamp_ms=info.timestamp_ms,
                        records=records,
                    )

                if video_writer is not None:
                    video_writer.write(frame)

                if self.output_cfg["show"]:
                    draw_stats(frame, info.fps, person_count)
                    cv2.imshow("Task A - 员工检测", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

        if video_writer is not None:
            video_writer.release()
        if json_writer is not None:
            json_writer.close()
        cv2.destroyAllWindows()
        return stats
