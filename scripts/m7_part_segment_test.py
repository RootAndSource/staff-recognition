"""M7 部位分割验证：对指定帧跑 全身掩码 + 上半身部位掩码，输出可视化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.segmentor import load_segmentation_engine


def _overlay(frame: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    vis = frame.copy()
    overlay = np.zeros_like(vis)
    overlay[mask] = color
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 部位分割验证")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--frames", default="1035,16700,16746,23952")
    parser.add_argument("--out-dir", default="outputs/m7/part_test")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    seg = load_segmentation_engine(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    for fid in [int(x) for x in args.frames.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            print(f"f{fid}: read fail")
            continue
        insts = seg.segment(frame)
        parts = seg.segment_parts(frame, insts)
        print(f"f{fid}: {len(insts)} instances")
        for i, (inst, part) in enumerate(zip(insts, parts)):
            full_area = int(inst.mask.sum())
            part_area = int(part.sum()) if part is not None else 0
            ratio = part_area / max(full_area, 1)
            print(
                f"  inst{i}: bbox={[round(float(v)) for v in inst.xyxy]} "
                f"full={full_area} part={part_area} part/full={ratio:.2f}"
            )
        # 可视化拼图：原图 / 全身掩码 / 上半身部位
        vis = frame.copy()
        vis = _overlay(vis, insts[0].mask, (0, 200, 255)) if insts else vis
        for i, (inst, part) in enumerate(zip(insts, parts)):
            if part is None:
                continue
            vis = _overlay(vis, part, (0, 255, 0))
            x1, y1, x2, y2 = [int(v) for v in inst.xyxy]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis, f"i{i} part/full={int(part.sum())/max(int(inst.mask.sum()),1):.2f}",
                (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        cv2.imwrite(str(out_dir / f"f{fid}.png"), vis)
    cap.release()
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
