"""快速粗筛：YOLO26 检测 + ROI HSV 红色比例，定位"未穿工装"候选段。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def red_ratio(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 180))) & (s >= 70) & (v >= 70)
    return float(red.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--weights", default="models/yolo26x.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--start", type=int, default=3000)
    parser.add_argument("--end", type=int, default=28182)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=6)
    args = parser.parse_args()

    det = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    total = min(args.end, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    hits: list[tuple[int, list[float], float]] = []  # frame, box, min_red_ratio
    t0 = time.time()
    n = args.start
    with tqdm(total=(total - args.start) // args.step + 1, desc="scan", unit="f") as pbar:
        while n < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = cap.read()
            if not ok:
                break
            res = det.predict(
                frame, conf=args.conf, iou=0.5, imgsz=args.imgsz,
                classes=[0], device=None, verbose=False,
            )[0]
            if res.boxes is not None and len(res.boxes):
                for box in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = [int(v) for v in box]
                    bw, bh = x2 - x1, y2 - y1
                    r_main = red_ratio(frame, x1, y1 + int(0.15 * bh), x2, y1 + int(0.55 * bh))
                    r_lower = red_ratio(frame, x1, y1 + int(0.45 * bh), x2, y1 + int(0.85 * bh))
                    r = max(r_main, r_lower)
                    if r < 0.05:
                        hits.append((n, [float(v) for v in box], r))
            n += args.step
            pbar.update(1)
            if n % 5000 == 0:
                pbar.set_postfix(hits=len(hits))
    cap.release()
    print(f"scan done, {len(hits)} not-uniform candidate hits")
    if hits:
        segs = []
        cur = [hits[0]]
        for h in hits[1:]:
            if h[0] - cur[-1][0] <= args.step * 3:
                cur.append(h)
            else:
                segs.append(cur)
                cur = [h]
        segs.append(cur)
        for s in segs:
            if len(s) >= args.min_hits:
                print(f"  segment {s[0][0]}..{s[-1][0]} hits={len(s)} "
                      f"box={[round(v) for v in s[0][1]]}")


if __name__ == "__main__":
    main()
