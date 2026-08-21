"""主处理管线：逐帧 检测→跟踪→工装识别→输出 的串联。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .config import load_config
from .detector import PersonDetector
from .fusion import DetectionFusion
from .segmentor import load_segmentation_engine
from .identity import IdentityTracker
from .instance_merge import InstanceMerger
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


def _scale_records(records: list[dict], scale: float, raw_w: int, raw_h: int) -> list[dict]:
    """把记录坐标还原到原图：bbox 乘 scale（等比缩放），mask_polygon 乘原图宽高（归一化）。"""
    out = []
    for rec in records:
        r = dict(rec)
        if "bbox" in r and r["bbox"]:
            r["bbox"] = [float(v) * scale for v in r["bbox"]]
        if "mask_polygon" in r and r["mask_polygon"]:
            r["mask_polygon"] = [
                [float(x) * raw_w, float(y) * raw_h]
                for x, y in r["mask_polygon"]
            ]
        if "parts" in r and r["parts"]:
            parts = []
            for part in r["parts"]:
                pp = dict(part)
                if part.get("bbox"):
                    pp["bbox"] = [float(v) * scale for v in part["bbox"]]
                if part.get("mask_polygon"):
                    pp["mask_polygon"] = [
                        [float(x) * raw_w, float(y) * raw_h]
                        for x, y in part["mask_polygon"]
                    ]
                parts.append(pp)
            r["parts"] = parts
        out.append(r)
    return out


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
        scfg = self.config.get("segmentation", {})
        self.seg_mode = bool(scfg.get("enabled", False))
        self.segmentor = None
        self.min_box_size = int(mcfg.get("min_box_size", 0))
        self.full_interval = int(mcfg.get("full_interval", 3))
        upper_w = mcfg.get("upper_weights") or mcfg["weights"]
        full_w = mcfg.get("full_weights") or mcfg["weights"]
        if self.seg_mode:
            seg_w = scfg.get("det_weights") or full_w
            self.seg_detector = PersonDetector(
                weights=seg_w,
                conf_thres=mcfg["conf_thres"],
                iou_thres=mcfg["iou_thres"],
                imgsz=mcfg["imgsz"],
                device=mcfg.get("device", ""),
            )
            self.upper_detector = self.seg_detector
            self.full_detector = self.seg_detector
        else:
            self.upper_detector = PersonDetector(
                weights=upper_w,
                conf_thres=mcfg["conf_thres"],
                iou_thres=mcfg["iou_thres"],
                imgsz=mcfg["imgsz"],
                device=mcfg.get("device", ""),
            )
            self.full_detector = PersonDetector(
                weights=full_w,
                conf_thres=mcfg["conf_thres"],
                iou_thres=mcfg["iou_thres"],
                imgsz=mcfg["imgsz"],
                device=mcfg.get("device", ""),
            )
        self.detector = self.upper_detector

        def _make_tracker(det):
            return ByteTrackTracker(
                track_high_thresh=tcfg["track_high_thresh"],
                track_low_thresh=tcfg["track_low_thresh"],
                new_track_thresh=tcfg["new_track_thresh"],
                detector=det,
                merge_overlap_iou=float(tcfg.get("merge_overlap_iou", 0.3)),
                tracker_yaml=str(
                    ByteTrackTracker.write_tracker_yaml(
                        Path("configs") / "bytetrack.yaml",
                        track_buffer=int(tcfg["track_buffer"]),
                        track_high_thresh=float(tcfg["track_high_thresh"]),
                        track_low_thresh=float(tcfg["track_low_thresh"]),
                        new_track_thresh=float(tcfg["new_track_thresh"]),
                        tracker_type=str(tcfg.get("name", "botsort")),
                        with_reid=bool(tcfg.get("with_reid", True)),
                        appearance_thresh=float(tcfg.get("appearance_thresh", 0.8)),
                        reid_model=str(tcfg.get("reid_model", "auto")),
                    )
                ),
            )

        self.upper_tracker = _make_tracker(self.upper_detector)
        self.full_tracker = _make_tracker(self.full_detector)
        self.tracker = self.upper_tracker  # ?????
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
            classifier_path=ucfg.get("classifier_path"),
            classifier_only=bool(ucfg.get("classifier_only", True)),
        )
        self._perf_times: dict[str, list[float]] = {
            "track": [],
            "uniform": [],
            "identity": [],
            "pipeline_total": [],
        }
        self._should_draw: bool = False  # 由 run() 按 save_video/show 决定
        icfg = self.config.get("identity", {})
        self.identity = IdentityTracker(
            uniform_pending_frames=int(icfg.get("uniform_pending_frames", 10)),
            uniform_window=int(icfg.get("uniform_window", 30)),
            uniform_confirm_frames=int(icfg.get("uniform_confirm_frames", 60)),
              uniform_confirm_high=float(icfg.get("uniform_confirm_high", 0.9)),
              reverse_streak_frames=int(icfg.get("reverse_streak_frames", 5)),
              track_ttl_frames=int(icfg.get("track_ttl_frames", 300)),
              boundary_ttl_frames=int(icfg.get("boundary_ttl_frames", 60)),
              match_distance_factor=float(icfg.get("match_distance_factor", 2.5)),
              appearance_corr_threshold=float(icfg.get("appearance_corr_threshold", 0.85)),
              person_confirm_frames=int(icfg.get("person_confirm_frames", 3)),
              leave_visible_ratio=float(icfg.get("leave_visible_ratio", 0.4)),
          )
        # 启动时加载模型，加载失败尽早暴露（而非逐帧崩溃）
        self.fusion = DetectionFusion()
        if self.seg_mode:
            self.segmentor = load_segmentation_engine(self.config, det_model=self.seg_detector)
            try:
                from ultralytics.trackers.utils.reid import build_encoder
                self.merger = InstanceMerger(
                    build_encoder(True, seg_w, None)
                )
            except Exception:
                self.merger = InstanceMerger(None)
        self.upper_detector.load()
        self.full_detector.load()

    def _segment_frame(self, frame_bgr: cv2.Mat) -> list[Any]:
        """分割模式：ByteTrack 跟踪派生框 + SAM2 掩码 -> FusedDetection 列表。"""
        tracked = self.upper_tracker.update(frame_bgr, detections=[])
        if not tracked:
            return []
        boxes = np.array([o.xyxy for o in tracked], dtype=float)
        confs = np.array([o.confidence for o in tracked], dtype=float)
        insts = self.segmentor.segment_with_boxes(frame_bgr, boxes, confs)
        from .fusion import FusedDetection
        fused = []
        for o, inst in zip(tracked, insts):
            if inst is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in inst.xyxy]
            roi = (x1, y1, x2, y1 + int(0.55 * (y2 - y1)))
            fused.append(
                FusedDetection(
                    xyxy=inst.xyxy,
                    confidence=inst.confidence,
                    box_type="mask",
                    upper_roi=roi,
                    bt_key=f"s{o.track_id}",
                    is_confirmed=o.is_confirmed,
                    mask=inst.mask,
                    polygon_norm=inst.polygon_norm,
                )
            )
        return fused

    def _process_frame(
        self,
        frame_bgr: cv2.Mat,
        frame_id: int,
        timestamp_ms: float,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """??????? (json_records, ??, ????)?"""
        t0 = time.perf_counter()
        if self.seg_mode:
            fused = self._segment_frame(frame_bgr)
            frame_h, frame_w = frame_bgr.shape[:2]
            t1 = t2 = t3 = time.perf_counter()
        else:
            upper_objs = self.upper_tracker.update(frame_bgr, detections=[])
            t1 = time.perf_counter()
            full_objs = []
            if self.full_interval and (frame_id % self.full_interval == 0 or not upper_objs):
                full_objs = self.full_tracker.update(frame_bgr, detections=[])
            t2 = time.perf_counter()
            frame_h, frame_w = frame_bgr.shape[:2]
            fused = self.fusion.fuse(
                [(o.track_id, o.xyxy, o.confidence) for o in upper_objs],
                [(o.track_id, o.xyxy, o.confidence) for o in full_objs],
                frame_w, frame_h,
            )
            t3 = time.perf_counter()
        uniform_results: dict[str, Any] = {}
        confirmed = [
            f for f in fused
            if f.is_confirmed
            and (not self.min_box_size or min(f.xyxy[2]-f.xyxy[0], f.xyxy[3]-f.xyxy[1]) >= self.min_box_size)
        ]
        if confirmed:
            batch = self.uniform.classify_batch(
                frame_bgr, confirmed, rois=[f.upper_roi for f in confirmed]
            )
            uniform_results = {f.bt_key: res for f, res in zip(confirmed, batch)}
        t4 = time.perf_counter()

        persons = self.identity.update(frame_id, frame_bgr, fused, uniform_results)
        if self.seg_mode and self.merger is not None:
            persons = self.merger.process(frame_bgr, persons, frame_id)
        records: list[dict[str, Any]] = []
        uniform_count = 0
        for p in persons:
            if self._should_draw:
                draw_tracked(frame_bgr, p)
            records.append(build_person_record(p))
            if p.uniform:
                uniform_count += 1
        t5 = time.perf_counter()
        if self.output_cfg.get("profile", False):
            self._perf_times["track"].append((t2 - t0) * 1000)
            self._perf_times["uniform"].append((t4 - t3) * 1000)
            self._perf_times["identity"].append((t5 - t4) * 1000)
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
            total = vs.frame_count if vs.frame_count and vs.frame_count > 0 else None
            desc = Path(source).name if not source.startswith(("rtsp://", "http://")) else "rtsp"
            frames_iter = vs.frames(skip=skip)
            with tqdm(total=total, desc=desc, unit="f", unit_scale=False, dynamic_ncols=True) as pbar:
                for info in frames_iter:
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
                        scale = 1.0
                        raw_w = info.width or frame.shape[1]
                        raw_h = info.height or frame.shape[0]
                        if frame.shape[1] and frame.shape[0]:
                            scale = max(raw_w / frame.shape[1], raw_h / frame.shape[0])
                        json_writer.write_frame(
                            frame_id=info.frame_id,
                            timestamp_ms=info.timestamp_ms,
                            records=_scale_records(records, scale, raw_w, raw_h),
                        )

                    if video_writer is not None:
                        if self._should_draw:
                            draw_stats(frame, info.fps, person_count)
                        video_writer.write(frame)

                    if self.output_cfg["show"]:
                        cv2.imshow("Task A - 员工检测", frame)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break

                    pbar.update(1)
                    pbar.set_postfix(people=person_count)

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