"""M7 数据准备：收集"上半身掩码 crop"待标注集（工装/非工装分类）。

流程：检测器出框 → SAM2.1 全身掩码 → 上半身部位掩码（SAM2 上半身提示 ∩ 实例掩码）
→ 掩码外像素置灰的 crop。输出：
  - data/uniform_crops_mask/pending/    待人工归类的 crop（jpg）
  - data/uniform_crops_mask/meta.csv    弱标签（HSV 红色判定，仅供分层参考）

人工标注：把 pending/ 图片移动到 positive/（穿工装）或 negative/（未穿工装）。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.config import load_config
from src.detector import PersonDetector
from src.segmentor import load_segmentation_engine


def _crop_with_mask(frame: np.ndarray, mask: np.ndarray, box: np.ndarray, pad: int = 12) -> np.ndarray | None:
    """掩码外像素置灰的 crop；返回 BGR 图。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    m = mask[y1:y2, x1:x2]
    crop[~m] = 114
    return crop


def _red_ratio(bgr_crop: np.ndarray) -> float:
    """HSV 红色弱标签：红色像素占比（供分层参考，不作最终标签）。"""
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
    red = m1 | m2
    total = bgr_crop.shape[0] * bgr_crop.shape[1]
    return float(red.sum() / total) if total else 0.0


def run(
    source: str,
    out_dir: Path,
    *,
    sample_every: int = 30,
    max_samples: int = 2000,
    min_part_area: int = 1200,
    min_part_ratio: float = 0.15,
) -> None:
    cfg = load_config("config.yaml")
    det = PersonDetector(
        weights=cfg["segmentation"]["det_weights"],
        conf_thres=0.3,
        iou_thres=0.5,
        imgsz=960,
        device=cfg["segmentation"].get("device", ""),
    )
    seg = load_segmentation_engine(cfg)
    pending = out_dir / "pending"
    (out_dir / "positive").mkdir(parents=True, exist_ok=True)
    (out_dir / "negative").mkdir(parents=True, exist_ok=True)
    pending.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta: list[tuple[str, float, int]] = []
    saved = 0
    n_frames = 0
    t0 = time.time()
    while saved < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        fid = n_frames
        n_frames += 1
        if fid % sample_every != 0:
            continue
        dets = det.detect(frame)
        if not dets:
            continue
        boxes = np.array([d.xyxy for d in dets], dtype=float)
        confs = np.array([d.confidence for d in dets], dtype=float)
        insts = seg.segment_with_boxes(frame, boxes, confs)
        insts = [i for i in insts if i is not None]
        if not insts:
            continue
        parts = seg.segment_parts(frame, insts)
        for idx, (inst, part) in enumerate(zip(insts, parts)):
            if part is None or int(part.sum()) < min_part_area:
                continue
            full = int(inst.mask.sum())
            if full > 0 and int(part.sum()) / full < min_part_ratio:
                continue
            crop = _crop_with_mask(frame, part, inst.xyxy)
            if crop is None:
                continue
            name = f"005_{fid:05d}_{idx:02d}.jpg"
            cv2.imwrite(str(pending / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta.append((name, _red_ratio(crop), int(part.sum())))
            saved += 1
        if n_frames % 1000 == 0:
            el = time.time() - t0
            print(f"progress {n_frames}/{total} saved={saved} elapsed={el:.0f}s")
    cap.release()

    with (out_dir / "meta.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "hsv_weak_label", "part_area"])
        for name, red, area in meta:
            w.writerow([name, 1 if red > 0.12 else 0, area])
    print(f"done: saved={saved} (n_frames={n_frames}) -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 上半身掩码 crop 收集")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--out-dir", default="data/uniform_crops_mask")
    parser.add_argument("--sample-every", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--min-part-area", type=int, default=1200)
    args = parser.parse_args()
    run(
        args.source,
        Path(args.out_dir),
        sample_every=args.sample_every,
        max_samples=args.max_samples,
        min_part_area=args.min_part_area,
    )


if __name__ == "__main__":
    main()
