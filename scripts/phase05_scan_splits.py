"""Phase 0.5 扫描：全片收集"遮挡分裂候选帧"。

用微调检测器（比 SAM2 快一个数量级）逐帧出框，筛选同帧内
垂直堆叠、x 中心对齐、且存在遮挡间隙的两框（疑似"上半身 + 脚"
被桌子挡成两段），输出候选列表与裁剪缩略图供人工目检。

用法：
  python scripts/phase05_scan_splits.py --source data/preprocessed/005.mp4 \
      --out-dir outputs/phase05
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.detector import PersonDetector


def _spatial_features(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_gap_ratio: float,
    max_gap_ratio: float,
    min_x_align: float,
) -> dict | None:
    """计算两派生框的空间特征；不满足"垂直堆叠 + 遮挡间隙"则返回 None。"""
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b
    wa, ha = x2a - x1a, y2a - y1a
    wb, hb = x2b - x1b, y2b - y1b
    if min(wa, wb, ha, hb) < 8:
        return None
    cxa = (x1a + x2a) / 2
    cxb = (x1b + x2b) / 2

    # 上下排列（允许轻微重叠），否则不是"两段"
    if y2a <= y1b:
        gap = y1b - y2a
    elif y2b <= y1a:
        gap = y1a - y2b
    else:
        return None
    if gap < 0:
        return None
    max_h = max(ha, hb)
    gap_ratio = gap / max_h
    if gap_ratio < min_gap_ratio or gap_ratio > max_gap_ratio:
        return None

    x_align = 1.0 - min(1.0, abs(cxa - cxb) / (0.5 * (wa + wb)))
    if x_align < min_x_align:
        return None

    height_ratio = min(ha, hb) / max_h
    area_ratio = (wa * ha) / max((wb * hb), 1.0)
    if height_ratio < 0.1 or height_ratio > 0.9:
        return None
    return {
        "x_align": round(float(x_align), 4),
        "gap": round(float(gap), 1),
        "gap_ratio": round(float(gap_ratio), 4),
        "height_ratio": round(float(height_ratio), 4),
        "area_ratio": round(float(area_ratio), 4),
        "top_center_x": round(float((x1a + x2a) / 2), 1),
        "top_center_y": round(float((y1a + y2a) / 2), 1),
    }


def _crop_thumb(
    frame: np.ndarray, boxes: list[np.ndarray], pad: int = 40, width: int = 520
) -> np.ndarray:
    """裁剪两框的合并区域、缩放并画框（坐标已换算到裁剪图），便于目检。"""
    xs0 = max(0, int(min(b[0] for b in boxes)) - pad)
    ys0 = max(0, int(min(b[1] for b in boxes)) - pad)
    xs1 = min(frame.shape[1], int(max(b[2] for b in boxes)) + pad)
    ys1 = min(frame.shape[0], int(max(b[3] for b in boxes)) + pad)
    crop = frame[ys0:ys1, xs0:xs1]
    if crop.size == 0:
        return frame
    scale = width / crop.shape[1]
    h = max(1, int(crop.shape[0] * scale))
    vis = cv2.resize(crop, (width, h), interpolation=cv2.INTER_AREA)
    for b in boxes:
        x1 = round((b[0] - xs0) * scale)
        y1 = round((b[1] - ys0) * scale)
        x2 = round((b[2] - xs0) * scale)
        y2 = round((b[3] - ys0) * scale)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0.5 遮挡分裂候选帧扫描")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0=全片")
    parser.add_argument("--out-dir", default="outputs/phase05")
    parser.add_argument("--min-gap-ratio", type=float, default=0.05)
    parser.add_argument("--max-gap-ratio", type=float, default=0.6)
    parser.add_argument("--min-x-align", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scfg = cfg.get("segmentation", {})
    det = PersonDetector(
        weights=scfg.get("det_weights", "models/yolo11x_finetuned.pt"),
        conf_thres=args.conf,
        iou_thres=0.5,
        imgsz=args.imgsz,
        device=scfg.get("device", ""),
    )
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    cand_f = (out_dir / "candidates.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    n_frames = 0
    n_cands = 0
    while True:
        if args.max_frames and n_frames >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_id = args.start_frame + n_frames
        dets = det.detect(frame)
        boxes = [d.xyxy for d in dets]

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                sp = _spatial_features(
                    boxes[i],
                    boxes[j],
                    min_gap_ratio=args.min_gap_ratio,
                    max_gap_ratio=args.max_gap_ratio,
                    min_x_align=args.min_x_align,
                )
                if sp is None:
                    continue
                cand = {
                    "frame_id": frame_id,
                    "box_a": [round(float(v), 1) for v in boxes[i]],
                    "box_b": [round(float(v), 1) for v in boxes[j]],
                    "spatial": sp,
                    "vis": f"f{frame_id}_i{i}_j{j}.png",
                }
                cand_f.write(json.dumps(cand, ensure_ascii=False) + "\n")
                n_cands += 1

                vis = _crop_thumb(frame, [boxes[i], boxes[j]])
                cv2.putText(
                    vis,
                    f"f{frame_id} ga={sp['gap_ratio']:.2f} xa={sp['x_align']:.2f}",
                    (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    vis,
                    f"f{frame_id} ga={sp['gap_ratio']:.2f} xa={sp['x_align']:.2f}",
                    (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.imwrite(str(vis_dir / cand["vis"]), vis)

        n_frames += 1
        if n_frames % 500 == 0:
            el = time.time() - t0
            fps = n_frames / max(el, 1e-6)
            print(
                f"progress {n_frames}/{total or args.max_frames or '?'} "
                f"elapsed={el:.0f}s fps={fps:.1f} cands={n_cands}"
            )

    cand_f.close()
    cap.release()
    el = time.time() - t0
    print(f"done: {n_frames} frames, {n_cands} candidates, {el:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
