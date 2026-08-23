"""S1+S5 组合：YOLO26 全身检测 -> ByteTrack 跟踪，输出 track_id + JSONL + 叠加视频。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="S1+S5: YOLO26 detect + ByteTrack track")
    parser.add_argument("--source", default="outputs/tmp_005_part.mp4")
    parser.add_argument("--weights", default="models/yolo26x.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--track-buffer", type=int, default=150)
    parser.add_argument("--track-high", type=float, default=0.3)
    parser.add_argument("--track-low", type=float, default=0.1)
    parser.add_argument("--new-track", type=float, default=0.6)
    parser.add_argument("--merge-iou", type=float, default=0.3)
    parser.add_argument("--json", default="outputs/json/s15_005_part.jsonl")
    parser.add_argument("--video", default="outputs/mp4/s15_005_part.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0=full")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    model = YOLO(args.weights)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    tracker_cfg = Path("configs")
    tracker_cfg.mkdir(parents=True, exist_ok=True)
    tracker_yaml = tracker_cfg / "bytetrack_s15.yaml"
    tracker_yaml.write_text(
        "\n".join(
            [
                "tracker_type: bytetrack",
                f"track_high_thresh: {args.track_high}",
                f"track_low_thresh: {args.track_low}",
                f"new_track_thresh: {args.new_track}",
                f"track_buffer: {args.track_buffer}",
                "match_thresh: 0.8",
                "fuse_score: True",
            ]
        ),
        encoding="utf-8",
    )

    cap = cv2.VideoCapture(args.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    jf = json_path.open("w", encoding="utf-8")
    n_total = args.max_frames or total
    t0 = time.time()
    n = 0

    def merge_overlapping(tracked: list[tuple[int, np.ndarray, float]], iou_thres: float):
        """同一帧内高重叠框合并（防 YOLO 把同一人拆成两个框）。"""
        if len(tracked) < 2:
            return tracked
        ordered = sorted(tracked, key=lambda o: -o[2])
        kept: list[tuple[int, np.ndarray, float]] = []
        for tid, box, conf in ordered:
            overlap = False
            for _, kbox, _ in kept:
                a, b = box, kbox
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                x2, y2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
                area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
                iou = inter / max(area_a + area_b - inter, 1e-6)
                if iou >= iou_thres:
                    overlap = True
                    break
            if not overlap:
                kept.append((tid, box, conf))
        return kept

    with tqdm(total=n_total, desc="S1+S5", unit="frame") as pbar:
        while True:
            if args.max_frames and n >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.video, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
                )

            results = model.track(
                frame,
                persist=True,
                tracker=str(tracker_yaml),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=[0],
                device=args.device or None,
                verbose=False,
            )
            tracked = []
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for tid, box, cf in zip(ids, xyxy, confs):
                    tracked.append((int(tid), box.astype(np.float32), float(cf)))
            tracked = merge_overlapping(tracked, args.merge_iou)

            records = []
            for tid, box, cf in tracked:
                x1, y1, x2, y2 = [float(v) for v in box]
                records.append({"track_id": tid, "bbox": [x1, y1, x2, y2], "conf": round(cf, 4)})
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                cv2.putText(
                    frame, f"ID{tid} {cf:.2f}", (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
                )

            jf.write(json.dumps({"frame_id": n, "tracks": records}, ensure_ascii=False) + "\n")
            writer.write(frame)
            n += 1
            pbar.update(1)
            if n % 500 == 0:
                el = time.time() - t0
                pbar.set_postfix(fps=f"{n / max(el, 1e-6):.1f}", tracks=len(tracked))

    cap.release()
    if writer is not None:
        writer.release()
    jf.close()
    el = time.time() - t0
    print(f"done frames={n} elapsed={el:.0f}s avg_fps={n / max(el, 1e-6):.1f}")


if __name__ == "__main__":
    main()
