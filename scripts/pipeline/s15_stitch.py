"""S5 轨迹缝合：把时间相邻、空间高度重叠但 ID 不同的轨迹段合并。

背景：ByteTrack 对低置信度/遮挡目标 ID 保持力弱，同一人会断成多段 ID。
缝合规则（离线后处理）：
  - 轨迹 A 在帧 t 结束，轨迹 B 在帧 t+k（k<=max_gap）开始；
  - A 末尾框与 B 起始框 IoU >= iou_thres；
  - 满足则把 B 的 track_id 统一改为 A（保留 A 的 ID）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / max(aa + bb - inter, 1e-6)


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def build_tracks(rows: list[dict]) -> dict[int, list[tuple[int, list[float], float]]]:
    tracks: dict[int, list[tuple[int, list[float], float]]] = {}
    for fid, r in enumerate(rows):
        for t in r.get("tracks", []):
            tracks.setdefault(int(t["track_id"]), []).append(
                (fid, t["bbox"], t["conf"])
            )
    for seq in tracks.values():
        seq.sort(key=lambda x: x[0])
    return tracks


def stitch(
    rows: list[dict],
    *,
    max_gap: int = 30,
    iou_thres: float = 0.5,
    min_span: int = 20,
) -> tuple[list[dict], int, int]:
    """返回 (新 rows, 合并次数, 剩余 ID 数)。"""
    tracks = build_tracks(rows)
    # 过滤过短轨迹（闪烁噪声），它们不作为"母轨迹"
    candidates = {tid: seq for tid, seq in tracks.items() if len(seq) >= min_span}
    merged: dict[int, int] = {}  # child_id -> parent_id

    # 按开始帧排序候选（母轨迹越早越优）
    order = sorted(candidates.items(), key=lambda kv: kv[1][0][0])
    parents: dict[int, list[tuple[int, list[float], float]]] = {}
    for tid, seq in order:
        # 已并入其它轨迹则跳过
        root = tid
        while root in merged:
            root = merged[root]
        if root != tid:
            continue
        matched = False
        for pid, pseq in parents.items():
            # 检查是否在 pseq 末尾附近出现且空间重叠
            p_end = pseq[-1][0]
            t_start = seq[0][0]
            gap = t_start - p_end - 1
            if gap < 0 or gap > max_gap:
                continue
            # 比较 pseq 末尾几帧与 seq 开头几帧（取最接近的一对）
            p_tail = [s for s in pseq if p_end - s[0] <= 5]
            t_head = [s for s in seq if s[0] - t_start <= 5]
            best = max(
                (iou(pa[1], tb[1]) for pa in p_tail for tb in t_head),
                default=0.0,
            )
            if best >= iou_thres:
                merged[tid] = pid
                parents[pid].extend(seq)
                parents[pid].sort(key=lambda x: x[0])
                matched = True
                break
        if not matched:
            parents[tid] = list(seq)

    # 应用合并：重写 track_id
    def resolve(tid: int) -> int:
        root = tid
        seen = set()
        while root in merged and root not in seen:
            seen.add(root)
            root = merged[root]
        return root

    n_merged = len(merged)
    for r in rows:
        for t in r.get("tracks", []):
            t["track_id"] = resolve(int(t["track_id"]))
    unique = len({t["track_id"] for r in rows for t in r.get("tracks", [])})
    return rows, n_merged, unique


def main() -> None:
    parser = argparse.ArgumentParser(description="S5 track stitching post-process")
    parser.add_argument("--json", default="outputs/json/s15_005_part.jsonl")
    parser.add_argument("--out", default="outputs/json/s15_005_part_stitched.jsonl")
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-span", type=int, default=20)
    args = parser.parse_args()

    rows = load(Path(args.json))
    new_rows, n_merged, unique = stitch(
        rows,
        max_gap=args.max_gap,
        iou_thres=args.iou,
        min_span=args.min_span,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged={n_merged} unique_ids={unique} -> {args.out}")


if __name__ == "__main__":
    main()
