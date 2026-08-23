"""S1+S2 组合：YOLO26 全身检测 -> SAM2.1 框提示分割，输出 JSONL + 叠加视频。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import SAM, YOLO


def mask_to_polygon(mask: np.ndarray, max_points: int = 40) -> np.ndarray:
    """Mask to simplified polygon (normalized coords)."""
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=float)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(cnt) <= max_points:
        return cnt
    closed = np.vstack([cnt, cnt[0]]).astype(np.float32)
    eps = 0.01 * cv2.arcLength(closed, True)
    while True:
        approx = cv2.approxPolyDP(closed, eps, True).reshape(-1, 2)
        if len(approx) <= max_points or eps <= 1e-6:
            break
        eps *= 0.6
    return approx


def main() -> None:
    parser = argparse.ArgumentParser(description="S1+S2: YOLO26 detect + SAM2 segment")
    parser.add_argument("--source", default="outputs/tmp_005_part.mp4")
    parser.add_argument("--det-weights", default="models/yolo26x.pt")
    parser.add_argument("--sam-weights", default="models/sam2.1_l.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--json", default="outputs/json/s12_005_part.jsonl")
    parser.add_argument("--video", default="outputs/mp4/s12_005_part.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0=full")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    det = YOLO(args.det_weights)
    sam = SAM(args.sam_weights)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    jf = json_path.open("w", encoding="utf-8")
    n_total = args.max_frames or total
    t0 = time.time()
    n = 0
    with tqdm(total=n_total, desc="S1+S2", unit="frame") as pbar:
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

            res = det.predict(
                frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                classes=[0], device=args.device or None, verbose=False,
            )[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None and len(res.boxes) else np.zeros((0, 4))
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None and len(res.boxes) else np.zeros((0,))

            dets = []
            if len(boxes):
                try:
                    sam_res = sam(frame, bboxes=boxes, verbose=False, device=args.device or None)[0]
                except Exception:
                    sam_res = None
                masks = sam_res.masks.data.cpu().numpy() if sam_res is not None and sam_res.masks is not None else None
                overlay = frame.copy()
                for idx, (box, cf) in enumerate(zip(boxes, confs)):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    poly_norm = []
                    if masks is not None and idx < len(masks):
                        mask_bool = masks[idx] > 0.5
                        poly = mask_to_polygon(mask_bool, 40)
                        if len(poly) >= 3:
                            hh, ww = frame.shape[:2]
                            poly_norm = (poly / np.array([ww, hh], dtype=float)).round(4).tolist()
                            poly_pts = poly.reshape(-1, 1, 2).astype(np.int32)
                            cv2.fillPoly(overlay, [poly_pts], (0, 220, 120))
                    dets.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": round(float(cf), 4),
                        "mask_polygon_norm": poly_norm,
                    })
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                for det_rec in dets:
                    x1, y1, x2, y2 = det_rec["bbox"]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                    cv2.putText(
                        frame, f"{det_rec['conf']:.2f}", (int(x1), max(18, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
                    )

            jf.write(json.dumps({"frame_id": n, "dets": dets}, ensure_ascii=False) + "\n")
            writer.write(frame)
            n += 1
            pbar.update(1)
            if n % 500 == 0:
                el = time.time() - t0
                pbar.set_postfix(fps=f"{n / max(el, 1e-6):.1f}", people=len(dets))

    cap.release()
    if writer is not None:
        writer.release()
    jf.close()
    el = time.time() - t0
    print(f"done frames={n} elapsed={el:.0f}s avg_fps={n / max(el, 1e-6):.1f}")


if __name__ == "__main__":
    main()
