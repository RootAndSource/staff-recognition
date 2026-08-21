"""Phase 0（v0.6 实例归并标定）：收集"垂直堆叠候选对"数据。

对视频片段逐帧跑实例分割（全身检测 + SAM2.1 large），筛选同帧内
可能属于同一人的"上下两段"候选对（空间先验），并导出：
  out_dir/instances.jsonl   每帧实例列表（bbox/center/面积/掩码）
  out_dir/candidates.jsonl  候选对 + 空间特征（x 对齐/gap/高度比）
  out_dir/vis/*.png         候选对可视化（画框 + 连线）供人工标定同人/异人

用法：
  python scripts/phase0_split_candidates.py --source data/preprocessed/005.mp4 \
      --start-frame 900 --max-frames 2100 --out-dir outputs/phase0/005
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.config import load_config
from src.segmentor import load_segmentation_engine


def _spatial_features(a: np.ndarray, b: np.ndarray) -> dict | None:
    """计算两实例派生框的空间特征；不满足垂直堆叠先验返回 None。"""
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b
    wa, ha = x2a - x1a, y2a - y1a
    wb, hb = x2b - x1b, y2b - y1b
    cxa = (x1a + x2a) / 2
    cxb = (x1b + x2b) / 2

    # 上下排列：一个在上一个在下（允许轻微重叠）
    if y2a <= y1b:      # a 在上
        top, bot = a, b
        gap = y1b - y2a
    elif y2b <= y1a:    # b 在上
        top, bot = b, a
        gap = y1a - y2b
    else:
        return None      # 垂直方向重叠过多，不是上下两段
    if gap < -0.2 * min(ha, hb):
        return None
    max_gap = 0.8 * max(ha, hb)
    if gap > max_gap:
        return None

    x_align = 1.0 - min(1.0, abs(cxa - cxb) / (0.5 * (wa + wb)))
    return {
        "x_align": round(float(x_align), 4),
        "gap": round(float(gap), 1),
        "gap_ratio": round(float(gap / max(max(ha, hb), 1.0)), 4),
        "height_ratio": round(float(min(ha, hb) / max(max(ha, hb), 1.0)), 4),
        "top_center_x": round(float((top[0] + top[2]) / 2), 1),
        "top_center_y": round(float((top[1] + top[3]) / 2), 1),
        "bot_center_x": round(float((bot[0] + bot[2]) / 2), 1),
        "bot_center_y": round(float((bot[1] + bot[3]) / 2), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 分裂候选对收集")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--start-frame", type=int, default=900)
    parser.add_argument("--max-frames", type=int, default=2100)
    parser.add_argument("--out-dir", default="outputs/phase0/005")
    parser.add_argument("--min-area-ratio", type=float, default=0.0003, help="实例最小面积占比")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seg = load_segmentation_engine(cfg)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{args.source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    inst_f = (out_dir / "instances.jsonl").open("w", encoding="utf-8")
    cand_f = (out_dir / "candidates.jsonl").open("w", encoding="utf-8")

    t0 = time.time()
    n_frames = 0
    n_cands = 0
    while n_frames < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        insts = seg.segment(frame)
        insts = [i for i in insts if i.mask.sum() / (h * w) >= args.min_area_ratio]

        frame_id = args.start_frame + n_frames
        inst_records = []
        for idx, inst in enumerate(insts):
            x1, y1, x2, y2 = [float(v) for v in inst.xyxy]
            inst_records.append(
                {
                    "id": idx,
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    "area_ratio": round(float(inst.mask.sum() / (h * w)), 5),
                    "confidence": round(float(inst.confidence), 4),
                }
            )
        inst_f.write(json.dumps({"frame_id": frame_id, "insts": inst_records}, ensure_ascii=False) + "\n")

        # 候选对：同帧内两实例，垂直堆叠 + x 中心对齐
        for i in range(len(insts)):
            for j in range(i + 1, len(insts)):
                sp = _spatial_features(insts[i].xyxy, insts[j].xyxy)
                if sp is None or sp["x_align"] < 0.5:
                    continue
                cand = {
                    "frame_id": frame_id,
                    "inst_a": i,
                    "inst_b": j,
                    "spatial": sp,
                }
                cand_f.write(json.dumps(cand, ensure_ascii=False) + "\n")
                n_cands += 1

                # 可视化：画掩码轮廓 + 框 + 连线
                vis = frame.copy()
                for inst in (insts[i], insts[j]):
                    mask_u8 = (inst.mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
                    x1, y1, x2, y2 = [int(v) for v in inst.xyxy]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 1)
                ca = insts[i].xyxy[:2] + insts[i].xyxy[2:] 
                cx_a = int((insts[i].xyxy[0] + insts[i].xyxy[2]) / 2)
                cy_a = int((insts[i].xyxy[1] + insts[i].xyxy[3]) / 2)
                cx_b = int((insts[j].xyxy[0] + insts[j].xyxy[2]) / 2)
                cy_b = int((insts[j].xyxy[1] + insts[j].xyxy[3]) / 2)
                cv2.line(vis, (cx_a, cy_a), (cx_b, cy_b), (0, 0, 255), 2)
                cv2.putText(
                    vis, f"f{frame_id} a{i}-b{j}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                cv2.imwrite(str(vis_dir / f"f{frame_id}_a{i}_b{j}.png"), vis)

        n_frames += 1
        if n_frames % 100 == 0:
            el = time.time() - t0
            print(f"  进度 {n_frames}/{args.max_frames}，耗时 {el:.0f}s，候选对 {n_cands}")

    inst_f.close()
    cand_f.close()
    cap.release()
    print(
        f"完成：{n_frames} 帧，候选对 {n_cands} 个；"
        f"实例 -> {out_dir}/instances.jsonl，候选 -> {out_dir}/candidates.jsonl，"
        f"可视化 -> {vis_dir}"
    )


if __name__ == "__main__":
    main()
