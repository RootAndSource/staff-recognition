"""Phase 0 embedding 标定：同人两段 vs 异人两段的 ReID 相似度分布。"""

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from boxmot.reid import ReID

from src.config import load_config
from src.segmentor import load_segmentation_engine


def crop_with_mask(frame, bbox, mask):
    """掩码 crop：掩码外像素置灰（减少背景干扰）。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2].copy()
    if mask is not None:
        m = mask[y1:y2, x1:x2]
        crop[~m] = 114
    return crop


def main() -> None:
    base = Path("outputs/phase0/005_full")
    rows = list(csv.reader(open(base / "annotation_list.csv", encoding="gbk")))
    cands = [
        json.loads(l)
        for l in (base / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    insts_by_frame = {}
    for line in (base / "instances.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        insts_by_frame[rec["frame_id"]] = rec["insts"]

    cfg = load_config("config.yaml")
    seg = load_segmentation_engine(cfg)
    enc = ReID("models/osnet_x0_25_msmt17.pt", device="0", half=False)
    cap = cv2.VideoCapture("data/preprocessed/005.mp4")

    same, diff, same_m, diff_m = [], [], [], []
    for r in rows[1:]:
        if len(r) < 7:
            continue
        cid, fid, label = int(r[0]), int(r[1]), r[6].strip()
        xa = float(r[3])
        cand = next(
            (c for c in cands if c["frame_id"] == fid and abs(c["spatial"]["x_align"] - xa) < 1e-3),
            None,
        )
        if cand is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        insts = insts_by_frame.get(fid, [])
        if cand["inst_a"] >= len(insts) or cand["inst_b"] >= len(insts):
            continue
        # 重新跑分割拿掩码（instances.jsonl 未存掩码）
        seg_insts = seg.segment(frame)
        ia = insts[cand["inst_a"]]
        ib = insts[cand["inst_b"]]
        # 用 bbox 匹配分割实例取掩码
        def match_mask(bbox):
            best, best_iou = None, 0.0
            for s in seg_insts:
                x1, y1, x2, y2 = [float(v) for v in s.xyxy]
                bb = np.array([x1, y1, x2, y2])
                xa_, ya_ = max(bbox[0], bb[0]), max(bbox[1], bb[1])
                xb_, yb_ = min(bbox[2], bb[2]), min(bbox[3], bb[3])
                inter = max(0, xb_ - xa_) * max(0, yb_ - ya_)
                uni = (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) + (bb[2]-bb[0])*(bb[3]-bb[1]) - inter
                iou = inter / max(uni, 1e-6)
                if iou > best_iou:
                    best, best_iou = s, iou
            return best.mask if best is not None else None

        m_a = match_mask(ia["bbox"])
        m_b = match_mask(ib["bbox"])
        # bbox crop embedding
        crops_b = [crop_with_mask(frame, ia["bbox"], None), crop_with_mask(frame, ib["bbox"], None)]
        dets_xyxy = np.array(
            [[b[0], b[1], b[2], b[3]] for b in (ia["bbox"], ib["bbox"])], dtype=float
        )
        emb_b = enc(frame, boxes=dets_xyxy)
        # 掩码 crop embedding
        emb_m = None
        if m_a is not None and m_b is not None:
            crops_m = [crop_with_mask(frame, ia["bbox"], m_a), crop_with_mask(frame, ib["bbox"], m_b)]
            emb_m = enc(crops_m)

        def cos(x, y):
            x = np.asarray(x).reshape(-1).astype(float)
            y = np.asarray(y).reshape(-1).astype(float)
            return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

        cb = cos(emb_b[0], emb_b[1]) if emb_b is not None and len(emb_b) > 1 else None
        cm = cos(emb_m[0], emb_m[1]) if emb_m is not None and len(emb_m) > 1 else None
        print(
            f"id{cid} f{fid} label={label} bbox_crop_cos={cb:.3f} "
            f"mask_crop_cos={cm if cm is None else round(cm, 3)}"
        )
        (same if label == "1" else diff).append(cb)
        (same_m if label == "1" else diff_m).append(cm)

    cap.release()
    for name, vals in (
        ("same_bbox", same),
        ("diff_bbox", diff),
        ("same_mask", same_m),
        ("diff_mask", diff_m),
    ):
        vals = [v for v in vals if v is not None]
        if vals:
            print(
                f"{name} n={len(vals)} cos: min={min(vals):.3f} max={max(vals):.3f} "
                f"mean={statistics.mean(vals):.3f}"
            )


if __name__ == "__main__":
    main()
