"""S1 全身检测（仅检测）：更强预训练模型，逐帧检测并输出叠加视频 + JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 全身检测")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--weights", default="models/yolo26x.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--json", default="outputs/json/s1_005.jsonl")
    parser.add_argument("--video", default="outputs/mp4/s1_005.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0=全片")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    jf = json_path.open("w", encoding="utf-8")
    t0 = time.time()
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
                args.video, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
            )
        res = model.predict(
            frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            classes=[0], device=args.device or None, verbose=False,
        )[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, cf in zip(boxes, confs):
                x1, y1, x2, y2 = [int(v) for v in box]
                dets.append({"bbox": [float(v) for v in box], "conf": round(float(cf), 4)})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    frame, f"{cf:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
                )
        jf.write(json.dumps({"frame_id": n, "dets": dets}, ensure_ascii=False) + "\n")
        writer.write(frame)
        n += 1
        if n % 500 == 0:
            el = time.time() - t0
            print(f"progress {n}/{total or args.max_frames or '?'} elapsed={el:.0f}s "
                  f"fps={n / max(el, 1e-6):.1f} people={len(dets)}")
    cap.release()
    if writer:
        writer.release()
    jf.close()
    el = time.time() - t0
    print(f"done: {n} frames, {el:.0f}s ({n / max(el, 1e-6):.1f} fps) -> {json_path} / {args.video}")


if __name__ == "__main__":
    main()
