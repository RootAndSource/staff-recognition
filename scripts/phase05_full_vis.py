"""为候选帧生成全帧标注缩略图（保留上下文，便于目检同人/异人）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="生成候选帧全帧标注缩略图")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--candidates", default="outputs/phase05/candidates.jsonl")
    parser.add_argument("--frames", default="", help="逗号分隔 frame_id（留空=全部匹配过滤）")
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--max-gap", type=float, default=0.6)
    parser.add_argument("--min-xa", type=float, default=0.55)
    parser.add_argument("--min-hr", type=float, default=0.1)
    parser.add_argument("--max-hr", type=float, default=0.7)
    parser.add_argument("--out-dir", default="outputs/phase05/full_vis")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--limit", type=int, default=0, help="最多输出帧数（0=全部）")
    args = parser.parse_args()

    want = {int(x) for x in args.frames.split(",") if x.strip()}
    sel: dict[int, list[dict]] = {}
    for line in open(args.candidates, encoding="utf-8"):
        c = json.loads(line)
        sp = c["spatial"]
        if want and c["frame_id"] not in want:
            continue
        if not (
            args.min_gap <= sp["gap_ratio"] <= args.max_gap
            and sp["x_align"] >= args.min_xa
            and args.min_hr <= sp["height_ratio"] <= args.max_hr
        ):
            continue
        sel.setdefault(c["frame_id"], []).append(c)
    if not sel:
        raise SystemExit("no frames selected")
    if args.limit:
        fids = [
            fid
            for fid, _ in sorted(
                sel.items(),
                key=lambda kv: -max(c["spatial"]["gap_ratio"] for c in kv[1]),
            )
        ][: args.limit]
    else:
        fids = sorted(sel)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    for fid in fids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        for ci, c in enumerate(sel[fid]):
            color = (0, 0, 255) if ci == 0 else (0, 165, 255)
            for bb in (c["box_a"], c["box_b"]):
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            vis, f"f{fid} n={len(sel[fid])}",
            (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6,
        )
        cv2.putText(
            vis, f"f{fid} n={len(sel[fid])}",
            (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
        )
        scale = args.width / vis.shape[1]
        h = int(vis.shape[0] * scale)
        vis = cv2.resize(vis, (args.width, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"f{fid}.png"), vis)
    cap.release()
    print(f"written {len(fids)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
