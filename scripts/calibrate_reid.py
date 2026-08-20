"""ReID 外观相似度阈值标定。

从视频跑 BoT-SORT 跟踪，收集每个 track 的检测框 crop 并提取 ReID embedding，
构造"同人（同一 track 不同帧）/ 异人（不同 track）"样本对，
输出余弦相似度分布与推荐关联/恢复阈值。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from src.detector import PersonDetector
from src.tracker import ByteTrackTracker
from src.video_input import VideoSource


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_calibrate(
    video: str,
    *,
    reid_weights: str = "models/yolo26n-cls.pt",
    det_weights: str = "models/yolo11x.pt",
    start_frame: int = 19000,
    frame_count: int = 600,
    sample_every: int = 10,
) -> None:
    det = PersonDetector(weights=det_weights, conf_thres=0.25, imgsz=960)
    yaml_path = ByteTrackTracker.write_tracker_yaml(
        "configs/botsort.yaml",
        track_buffer=150,
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        tracker_type="botsort",
        with_reid=True,
        reid_model="auto",
    )
    trk = ByteTrackTracker(detector=det, tracker_yaml=str(yaml_path))
    det.load()
    reid = YOLO(reid_weights)

    track_crops: dict[int, list[np.ndarray]] = defaultdict(list)
    seen = 0
    with VideoSource(video) as vs:
        for info in vs.frames():
            if info.frame_id < start_frame:
                continue
            if info.frame_id >= start_frame + frame_count:
                break
            if info.frame_id % sample_every != 0:
                continue
            objs = trk.update(info.frame, detections=[])
            for o in objs:
                x1, y1, x2, y2 = [int(v) for v in o.xyxy]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(info.frame.shape[1], x2), min(info.frame.shape[0], y2)
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                crop = info.frame[y1:y2, x1:x2]
                track_crops[o.track_id].append(crop)
            seen += 1

    # 提取 embedding
    emb_by_track: dict[int, list[np.ndarray]] = {}
    for tid, crops in track_crops.items():
        if len(crops) < 2:
            continue
        embs = []
        for c in crops:
            e = reid.embed([c])[0].detach().cpu().numpy()
            embs.append(e / (np.linalg.norm(e) + 1e-9))
        emb_by_track[tid] = embs

    same, diff = [], []
    rng = random.Random(42)
    for tid, embs in emb_by_track.items():
        for _ in range(min(30, len(embs))):
            a, b = rng.sample(embs, 2)
            same.append(cosine(a, b))
    ids = list(emb_by_track.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = rng.choice(emb_by_track[ids[i]])
            b = rng.choice(emb_by_track[ids[j]])
            diff.append(cosine(a, b))

    def report(name, vals):
        vals = np.array(vals)
        print(
            f"{name}: n={len(vals)} mean={vals.mean():.3f} "
            f"p5={np.percentile(vals, 5):.3f} p50={np.percentile(vals, 50):.3f} "
            f"p95={np.percentile(vals, 95):.3f}"
        )

    report("同人相似度", same)
    report("异人相似度", diff)
    if same and diff:
        low = np.percentile(same, 5)
        high = np.percentile(diff, 95)
        print(f"推荐关联阈值（同人p5 与 异人p95 中点）: {(low + high) / 2:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ReID 阈值标定")
    parser.add_argument("--video", default="data/preprocessed/005.mp4")
    parser.add_argument("--reid-weights", default="models/yolo26n-cls.pt")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()
    run_calibrate(
        args.video,
        reid_weights=args.reid_weights,
        start_frame=args.start_frame,
        frame_count=args.frames,
    )


if __name__ == "__main__":
    main()
