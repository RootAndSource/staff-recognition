"""ID 稳定性评估：从 JSONL 统计短时遮挡恢复率（验收口径 ≥80%）。

遮挡事件定义：某 person_id 的记录出现断裂（gap ≤ 90 帧 = 3 秒），
断裂前后位置接近（中心距离 < 2.5 倍框宽），视为一次"短时遮挡事件"。
恢复成功 = 断裂后复用原 person_id；失败 = 变成新 person_id。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def evaluate_id_stability(
    jsonl_path: str | Path,
    *,
    max_gap_frames: int = 90,
    match_distance_factor: float = 2.5,
    segment_gap_frames: int = 15,
) -> dict:
    """统计遮挡事件与恢复率。"""
    rows = [
        json.loads(l.rstrip().rstrip(","))
        for l in open(jsonl_path, encoding="utf-8")
        if l.strip() not in ("[", "]")
    ]
    by_pid: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for r in rows:
        fid = r["frame_id"]
        for p in r["people"]:
            by_pid[p["person_id"]].append((fid, p["bbox"]))

    events: list[dict] = []
    for pid, recs in by_pid.items():
        recs.sort()
        # 切分连续片段（相邻记录 gap > segment_gap_frames 视为断裂）
        segments: list[list[tuple[int, list[float]]]] = []
        cur = [recs[0]]
        for prev, rec in zip(recs, recs[1:]):
            if rec[0] - prev[0] > segment_gap_frames:
                segments.append(cur)
                cur = [rec]
            else:
                cur.append(rec)
        segments.append(cur)
        if len(segments) < 2:
            continue
        # 相邻片段检查遮挡事件
        for a, b in zip(segments, segments[1:]):
            gap = b[0][0] - a[-1][0]
            if gap > max_gap_frames:
                continue
            box_a, box_b = a[-1][1], b[0][1]
            dist = float(np.linalg.norm(_center(box_a) - _center(box_b)))
            box_w = max(box_a[2] - box_a[0], 1.0)
            if dist > box_w * match_distance_factor:
                continue  # 位置变化大，不算遮挡事件
            events.append(
                {
                    "person_id": pid,
                    "gap_frames": gap,
                    "distance": round(dist, 1),
                    "box_w": round(box_w, 1),
                    "recovered": True,
                }
            )

    # 片段级接替事件：person A 的最后片段末尾 vs person B 的首片段开头
    # 时间 gap ≤ max_gap_frames 且位置接近 → A 消失后被新 ID B 接替（恢复失败）
    first_segments: dict[int, tuple[int, list[float]]] = {}
    last_segments: dict[int, tuple[int, list[float]]] = {}
    for pid, recs in by_pid.items():
        recs.sort()
        segments = []
        cur = [recs[0]]
        for prev, rec in zip(recs, recs[1:]):
            if rec[0] - prev[0] > segment_gap_frames:
                segments.append(cur)
                cur = [rec]
            else:
                cur.append(rec)
        segments.append(cur)
        first_segments[pid] = segments[0][0]
        last_segments[pid] = segments[-1][-1]

    replaced = 0
    for a in by_pid:
        f_end, box_end = last_segments[a]
        best: tuple[int, float] | None = None
        for b in by_pid:
            if b == a:
                continue
            f_start, box_start = first_segments[b]
            gap = f_start - f_end
            if 0 < gap <= max_gap_frames:
                dist = float(np.linalg.norm(_center(box_end) - _center(box_start)))
                box_w = max(box_end[2] - box_end[0], 1.0)
                if dist <= box_w * match_distance_factor:
                    if best is None or gap < best[0]:
                        best = (gap, dist)
        if best is not None:
            replaced += 1

    recovered = sum(1 for e in events if e["recovered"])
    total = recovered + replaced
    rate = recovered / total if total else 0.0
    return {
        "recovered_events": recovered,
        "replaced_events": replaced,
        "total_events": total,
        "recovery_rate": round(rate, 4),
        "threshold": 0.8,
        "passed": rate >= 0.8,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ID 稳定性评估（短时遮挡恢复率）")
    parser.add_argument("jsonl", help="管线输出的 JSONL 路径")
    parser.add_argument("--max-gap", type=int, default=90, help="遮挡事件最大帧间隔（默认 90=3 秒）")
    args = parser.parse_args()
    result = evaluate_id_stability(args.jsonl, max_gap_frames=args.max_gap)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
