"""S1 渲染：从已有 JSONL 重绘叠加视频（调阈值用，无需重新推理）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="render S1 video from JSONL")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--json", default="outputs/json/s1_rfdetr_005.jsonl")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--out", default="outputs/mp4/s1_rfdetr_005_c04.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0=full")
    args = parser.parse_args()

    dets_by_frame = {}
    for line in Path(args.json).open(encoding="utf-8"):
        d = json.loads(line)
        dets_by_frame[d["frame_id"]] = d.get("dets") or d.get("tracks") or []

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    n = 0
    while True:
        if args.max_frames and n >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        for det in dets_by_frame.get(n, []):
            if det["conf"] < args.conf:
                continue
            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
            label = f"{det['conf']:.2f}"
            if det.get("track_id") is not None:
                label = f"ID{det['track_id']} {det['conf']:.2f}"
            cv2.putText(
                frame, label, (int(x1), max(18, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
            )
        writer.write(frame)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"done frames={n} -> {args.out}")


if __name__ == "__main__":
    main()
