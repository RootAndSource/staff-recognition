"""主处理管线：逐帧 检测→跟踪→工装识别→输出 的串联。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .config import load_config
from .detector import PersonDetector
from .identity import IdentityTracker
from .output import JsonOutputWriter, build_person_record
from .tracker import ByteTrackTracker
from .uniform import UniformClassifier, _RoiSpec
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
        self.min_box_size = int(mcfg.get("min_box_size", 0))
        self.tracker = ByteTrackTracker(
            lost_buffer=tcfg["lost_buffer"],
            track_high_thresh=tcfg["track_high_thresh"],
            track_low_thresh=tcfg["track_low_thresh"],
            new_track_thresh=tcfg["new_track_thresh"],
            detector=self.detector,   # 共享 YOLO 实例，避免重复推理
        )
        self.uniform = UniformClassifier(
            roi_main=_RoiSpec(
                ucfg["roi_main"]["x_begin"], ucfg["roi_main"]["y_begin"],
                ucfg["roi_main"]["x_end"], ucfg["roi_main"]["y_end"],
            ),
            roi_lower=_RoiSpec(
                ucfg["roi_lower"]["x_begin"], ucfg["roi_lower"]["y_begin"],
                ucfg["roi_lower"]["x_end"], ucfg["roi_lower"]["y_end"],
            ),
            hsv_ranges=((ucfg["hsv_red"]["h1_begin"], ucfg["hsv_red"]["h1_end"]),
                        (ucfg["hsv_red"]["h2_begin"], ucfg["hsv_red"]["h2_end"])),
            s_min=ucfg["hsv_red"]["s_min"],
            v_min=ucfg["hsv_red"]["v_min"],
            red_ratio_threshold=ucfg["red_ratio_threshold"],
            normalize=ucfg.get("normalize", True),
        )
        self.smooth_window = int(ucfg.get("smooth_window", 5))
        self._perf_times: dict[str, list[float]] = {
            "track": [],
            "uniform": [],
            "identity": [],
            "pipeline_total": [],
        }
        self._should_draw: bool = False  # 由 run() 按 save_video/show 决定
        icfg = self.config.get("identity", {})
        self.identity = IdentityTracker(
            uniform_window=int(icfg.get("uniform_window", 10)),
            uniform_lock_high=float(icfg.get("uniform_lock_high", 0.8)),
            uniform_lock_low=float(icfg.get("uniform_lock_low", 0.2)),
            track_ttl_frames=int(icfg.get("track_ttl_frames", 90)),
            match_distance_factor=float(icfg.get("match_distance_factor", 1.5)),
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
        t0 = time.perf_counter()
        tracked = self.tracker.update(frame_bgr, detections=[])
        t1 = time.perf_counter()
        uniform_results: dict[int, Any] = {}
        for obj in tracked:
            if not obj.is_confirmed:
                continue
            uniform_results[obj.track_id] = self.uniform.classify(frame_bgr, obj)
        t2 = time.perf_counter()

        persons = self.identity.update(frame_id, tracked, uniform_results)
        records: list[dict[str, Any]] = []
        uniform_count = 0
        for p in persons:
            if self._should_draw:
                draw_tracked(frame_bgr, p)
            records.append(build_person_record(p))
            if p.uniform:
                uniform_count += 1
        t3 = time.perf_counter()
        if self.output_cfg.get("profile", False):
            self._perf_times["track"].append((t1 - t0) * 1000)
            self._perf_times["uniform"].append((t2 - t1) * 1000)
            self._perf_times["identity"].append((t3 - t2) * 1000)
        return records, len(persons), uniform_count

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
        # 绘制叠加层：只要保存视频或显示窗口，就绘制检测框/标签/统计
        self._should_draw = bool(save_video) or bool(self.output_cfg.get("show", False))
        profile = bool(self.output_cfg.get("profile", False))
        profile_times: list[tuple[str, float]] = []

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

                t_frame = time.perf_counter()
                records, person_count, uniform_count = self._process_frame(
                    frame, info.frame_id, info.timestamp_ms
                )
                t_done = time.perf_counter()
                if profile:
                    profile_times.append(
                        ("pipeline_total", (t_done - t_frame) * 1000)
                    )
                    self._perf_times["pipeline_total"].append((t_done - t_frame) * 1000)
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
                    if self._should_draw:
                        draw_stats(frame, info.fps, person_count)
                    video_writer.write(frame)

                if self.output_cfg["show"]:
                    cv2.imshow("Task A - 员工检测", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

        if video_writer is not None:
            video_writer.release()
        if json_writer is not None:
            json_writer.close()
        cv2.destroyAllWindows()
        if profile and profile_times:
            import statistics
            for name, vals in self._perf_times.items():
                if not vals:
                    continue
                vals_sorted = sorted(vals)
                print(
                    f"[profile] {name:15s}: mean={statistics.mean(vals):6.1f}ms "
                    f"p50={vals_sorted[len(vals_sorted)//2]:6.1f}ms "
                    f"p99={vals_sorted[int(len(vals_sorted)*0.99)-1]:6.1f}ms "
                    f"max={max(vals):6.1f}ms n={len(vals)}"
                )
        return stats
