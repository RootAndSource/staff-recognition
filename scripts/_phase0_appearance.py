"""Phase 0 外观特征：候选对两段 crop 的 HSV 直方图相似度。"""

import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np


def hist_corr(frame, bbox1, bbox2):
    """两段 bbox crop 的 HSV 直方图相关系数。"""
    hists = []
    for bbox in (bbox1, bbox2):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            hists.append(None)
            continue
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist)
    if any(h is None for h in hists):
        return None
    return float(cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL))


def main() -> None:
    base = Path("outputs/phase0/005_full")
    rows = list(csv.reader(open(base / "annotation_list.csv", encoding="gbk")))
    cands = [
        json.loads(l)
        for l in (base / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    cap = cv2.VideoCapture("data/preprocessed/005.mp4")
    results = []
    for r in rows[1:]:
        if len(r) < 7:
            continue
        cid, fid, label = int(r[0]), int(r[1]), r[6].strip()
        xa = float(r[3])
        cand = next((c for c in cands if c["frame_id"] == fid and abs(c["spatial"]["x_align"] - xa) < 1e-3), None)
        if cand is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        # 从 instances.jsonl 取 bbox
        insts = None
        for line in (base / "instances.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["frame_id"] == fid:
                insts = rec["insts"]
                break
        if insts is None:
            continue
        bbox_a = insts[cand["inst_a"]]["bbox"]
        bbox_b = insts[cand["inst_b"]]["bbox"]
        corr = hist_corr(frame, bbox_a, bbox_b)
        results.append((cid, fid, label, corr))
        print(f"id{cid} f{fid} label={label} hsv_corr={corr}")
    cap.release()

    for lab, name in (("1", "同人"), ("0", "异人")):
        vals = [o[3] for o in results if o[2] == lab and o[3] is not None]
        if vals:
            print(
                f"{name} n={len(vals)} hsv_corr: min={min(vals):.3f} max={max(vals):.3f} "
                f"mean={statistics.mean(vals):.3f}"
            )


if __name__ == "__main__":
    main()
