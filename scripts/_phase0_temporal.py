"""Phase 0 时间特征：候选对跨帧跟踪，计算同现率与运动相关性。"""

import csv
import json
import statistics
from pathlib import Path

import numpy as np


def load_instances(path: Path) -> dict[int, list[dict]]:
    frames = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        frames[rec["frame_id"]] = rec["insts"]
    return frames


def track_instance(frames, frame_id: int, inst_id: int, window: int = 25):
    """从代表帧向两侧跟踪一个实例（按中心距离 + IoU 匹配），返回 {frame_id: (center, bbox)}。"""
    if frame_id not in frames or inst_id >= len(frames[frame_id]):
        return {}
    cur = frames[frame_id][inst_id]
    cur_bbox = np.array(cur["bbox"], dtype=float)
    track = {frame_id: (np.array(cur["center"], dtype=float), cur_bbox)}

    for direction in (-1, 1):
        prev_bbox = cur_bbox
        prev_center = np.array(cur["center"], dtype=float)
        for k in range(1, window + 1):
            fid = frame_id + direction * k
            if fid not in frames:
                break
            best = None
            best_score = 1e9
            for inst in frames[fid]:
                bb = np.array(inst["bbox"], dtype=float)
                ct = np.array(inst["center"], dtype=float)
                # 中心距离 + 尺寸差异
                d = float(np.linalg.norm(ct - prev_center))
                size_diff = abs(
                    (bb[2] - bb[0]) - (prev_bbox[2] - prev_bbox[0])
                ) + abs((bb[3] - bb[1]) - (prev_bbox[3] - prev_bbox[1]))
                score = d + 0.2 * size_diff
                if d < 1.2 * max((prev_bbox[2] - prev_bbox[0]), 20) and score < best_score:
                    best, best_score = inst, score
            if best is None:
                break
            track[fid] = (np.array(best["center"], dtype=float), np.array(best["bbox"], dtype=float))
            prev_center, prev_bbox = track[fid]
    return track


def temporal_features(track_a, track_b):
    """同现率、运动相关系数、位移幅度比。"""
    common = sorted(set(track_a) & set(track_b))
    all_frames = set(track_a) | set(track_b)
    co_occur = len(common) / max(len(all_frames), 1)
    if len(common) < 3:
        return {
            "co_occurrence": round(co_occur, 3),
            "motion_corr": None,
            "motion_ratio": None,
            "n_common": len(common),
        }
    da = []
    db = []
    for i in range(1, len(common)):
        f0, f1 = common[i - 1], common[i]
        da.append(track_a[f1][0] - track_a[f0][0])
        db.append(track_b[f1][0] - track_b[f0][0])
    da = np.asarray(da, dtype=float)
    db = np.asarray(db, dtype=float)
    corr = float(np.corrcoef(da, db)[0, 1]) if len(da) > 1 and np.std(da) > 1e-6 and np.std(db) > 1e-6 else None
    norm_a = float(np.linalg.norm(da.sum()))
    norm_b = float(np.linalg.norm(db.sum()))
    ratio = norm_a / norm_b if norm_b > 1e-6 else None
    return {
        "co_occurrence": round(co_occur, 3),
        "motion_corr": round(corr, 3) if corr is not None else None,
        "motion_ratio": round(ratio, 3) if ratio is not None else None,
        "n_common": len(common),
    }


def main() -> None:
    base = Path("outputs/phase0/005_full")
    frames = load_instances(base / "instances.jsonl")
    rows = list(csv.reader(open(base / "annotation_list.csv", encoding="gbk")))
    out_rows = []
    for r in rows[1:]:
        if len(r) < 7:
            continue
        cid, fid, nf, xa, gap, hr, label = (
            int(r[0]), int(r[1]), int(r[2]), float(r[3]), float(r[4]), float(r[5]), r[6].strip()
        )
        # 代表帧的 inst_a/inst_b 从 candidates.jsonl 里查
        cand = None
        for line in (base / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            if c["frame_id"] == fid and abs(c["spatial"]["x_align"] - xa) < 1e-3:
                cand = c
                break
        if cand is None:
            continue
        ta = track_instance(frames, fid, cand["inst_a"])
        tb = track_instance(frames, fid, cand["inst_b"])
        tf = temporal_features(ta, tb)
        out_rows.append((cid, fid, label, xa, gap, hr, tf))
        print(
            f"id{cid} f{fid} label={label} "
            f"co={tf['co_occurrence']} corr={tf['motion_corr']} ratio={tf['motion_ratio']} n={tf['n_common']}"
        )

    # 汇总同人 vs 异人
    for lab, name in (("1", "同人"), ("0", "异人")):
        rows_lab = [o for o in out_rows if o[2] == lab]
        cos = [o[6]["co_occurrence"] for o in rows_lab]
        corrs = [o[6]["motion_corr"] for o in rows_lab if o[6]["motion_corr"] is not None]
        print(f"\n{name} n={len(rows_lab)}")
        if cos:
            print(f"  co_occurrence: min={min(cos):.3f} max={max(cos):.3f} mean={statistics.mean(cos):.3f}")
        if corrs:
            print(f"  motion_corr:   min={min(corrs):.3f} max={max(corrs):.3f} mean={statistics.mean(corrs):.3f}")


if __name__ == "__main__":
    main()
