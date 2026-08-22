"""Phase 0.5 提示方式对比实验（图像模式）。

对"上半身 + 脚"分裂帧，用 SAM 2.1 官方库对比四种提示方式，判断
能否输出"跨遮挡完整人体掩码"（一个掩码同时覆盖两段）：
  A  box-prompt 基线：两个框各出一个掩码（现状）
  B1 头顶单点：上半身框顶中心一点
  B2 两段各一点：上半身中心 + 脚中心同时 positive
  C  整帧自动分割：无提示全图 masks

覆盖判据：某掩码与两段 bbox 的交集像素同时超过各自面积的
--cover-ratio（默认 0.2）视为"覆盖两段"。

输入：--frames 为 "fid:ia,ib" 列表（从 phase0 instances.jsonl 取 bbox），
  或 --candidates 为 phase05 扫描产物（每行 {frame_id, box_a, box_b}）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _load_phase0_boxes() -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """从 Phase 0 产物加载标注为同人（label=1）的候选对 bbox。"""
    import csv

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

    out: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for r in rows[1:]:
        if len(r) < 7 or r[6].strip() != "1":
            continue
        cid, fid = int(r[0]), int(r[1])
        xa = float(r[3])
        cand = next(
            (c for c in cands if c["frame_id"] == fid and abs(c["spatial"]["x_align"] - xa) < 1e-3),
            None,
        )
        if cand is None:
            continue
        insts = insts_by_frame.get(fid, [])
        if cand["inst_a"] >= len(insts) or cand["inst_b"] >= len(insts):
            continue
        a = np.array(insts[cand["inst_a"]]["bbox"], dtype=float)
        b = np.array(insts[cand["inst_b"]]["bbox"], dtype=float)
        out.setdefault(fid, []).append((a, b))
    return out


def _box_pixels_in_mask(mask: np.ndarray, box: np.ndarray) -> float:
    """掩码与 bbox 区域的交集占 bbox 面积的比例。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    return float(mask[y1:y2, x1:x2].sum() / area)


def _masks_from_batch(mask_data) -> list[np.ndarray]:
    """从 SAM2 输出 tensor（B x H x W）转 bool 掩码列表。"""
    masks = mask_data.cpu().numpy() if isinstance(mask_data, torch.Tensor) else np.asarray(mask_data)
    if masks.ndim == 4:
        masks = masks[:, 0]
    return [(m > 0.0) for m in masks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0.5 提示方式对比")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--phase0-same", action="store_true", help="使用 Phase 0 同人对")
    parser.add_argument("--candidates", default=None, help="phase05 candidates.jsonl")
    parser.add_argument("--frames", default="", help="逗号分隔 frame_id 列表（配合 --candidates）")
    parser.add_argument("--out-dir", default="outputs/phase05/prompt_test")
    parser.add_argument("--cover-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.phase0_same:
        pairs_by_frame = _load_phase0_boxes()
    else:
        pairs_by_frame = {}
        if args.candidates:
            want = {int(x) for x in args.frames.split(",") if x.strip()}
            for line in open(args.candidates, encoding="utf-8"):
                c = json.loads(line)
                if want and c["frame_id"] not in want:
                    continue
                pairs_by_frame.setdefault(c["frame_id"], []).append(
                    (np.array(c["box_a"], dtype=float), np.array(c["box_b"], dtype=float))
                )
    if not pairs_by_frame:
        raise SystemExit("no test frames loaded")

    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "models/sam2.1_l.pt",
        device=args.device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(sam)
    auto_gen = SAM2AutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.6,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        max_detections_per_image=32,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    results = []

    for fid in sorted(pairs_by_frame):
        for (a, b) in pairs_by_frame[fid]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            predictor.set_image(frame)

            row: dict = {"frame_id": fid, "box_a": [float(v) for v in a], "box_b": [float(v) for v in b]}

            # A: box-prompt 基线（两个框分别出掩码）
            box_a = np.array([a[0], a[1], a[2], a[3]])
            box_b = np.array([b[0], b[1], b[2], b[3]])
            masks_a, _, _ = predictor.predict(
                box=box_a[None, :], multimask_output=False
            )
            masks_b, _, _ = predictor.predict(
                box=box_b[None, :], multimask_output=False
            )
            m_a = _masks_from_batch(masks_a)[0]
            m_b = _masks_from_batch(masks_b)[0]
            row["A_box"] = {
                "a_covers_a": round(_box_pixels_in_mask(m_a, a), 3),
                "a_covers_b": round(_box_pixels_in_mask(m_a, b), 3),
                "b_covers_a": round(_box_pixels_in_mask(m_b, a), 3),
                "b_covers_b": round(_box_pixels_in_mask(m_b, b), 3),
            }

            # B1: 头顶单点
            top = a if a[1] <= b[1] else b
            bot = b if top is a else a
            head_pt = np.array([[(top[0] + top[2]) / 2, top[1] + 2]], dtype=np.float32)
            masks_h, _, _ = predictor.predict(
                point_coords=head_pt, point_labels=np.array([1]), multimask_output=True
            )
            best_h = None
            for m in _masks_from_batch(masks_h):
                cov_top = _box_pixels_in_mask(m, top)
                cov_bot = _box_pixels_in_mask(m, bot)
                if best_h is None or (cov_top + cov_bot) > (best_h[1] + best_h[2]):
                    best_h = (m, cov_top, cov_bot)
            row["B1_head"] = {
                "top": [round(float(v), 1) for v in top],
                "bot": [round(float(v), 1) for v in bot],
                "covers_top": round(best_h[1], 3),
                "covers_bot": round(best_h[2], 3),
                "covers_both": int(best_h[1] >= args.cover_ratio and best_h[2] >= args.cover_ratio),
            }

            # B2: 两段各一点（同时 positive）
            two_pts = np.array(
                [[(top[0] + top[2]) / 2, (top[1] + top[3]) / 2], [(bot[0] + bot[2]) / 2, (bot[1] + bot[3]) / 2]],
                dtype=np.float32,
            )
            masks_2, _, _ = predictor.predict(
                point_coords=two_pts, point_labels=np.array([1, 1]), multimask_output=True
            )
            best_2 = None
            for m in _masks_from_batch(masks_2):
                cov_top = _box_pixels_in_mask(m, top)
                cov_bot = _box_pixels_in_mask(m, bot)
                if best_2 is None or (cov_top + cov_bot) > (best_2[1] + best_2[2]):
                    best_2 = (m, cov_top, cov_bot)
            row["B2_two"] = {
                "covers_top": round(best_2[1], 3),
                "covers_bot": round(best_2[2], 3),
                "covers_both": int(best_2[1] >= args.cover_ratio and best_2[2] >= args.cover_ratio),
            }

            # C: 整帧自动分割（segment everything）
            t_c = time.time()
            anns_auto = auto_gen.generate(frame)
            auto_masks = [a["segmentation"] for a in anns_auto]
            auto_both = None
            for m in auto_masks:
                cov_top = _box_pixels_in_mask(m, top)
                cov_bot = _box_pixels_in_mask(m, bot)
                if cov_top >= args.cover_ratio and cov_bot >= args.cover_ratio:
                    if auto_both is None or (cov_top + cov_bot) > (auto_both[1] + auto_both[2]):
                        auto_both = (m, cov_top, cov_bot)
            row["C_auto"] = {
                "found": int(auto_both is not None),
                "covers_top": round(auto_both[1], 3) if auto_both else None,
                "covers_bot": round(auto_both[2], 3) if auto_both else None,
                "total_masks": len(auto_masks),
                "seconds": round(time.time() - t_c, 1),
            }

            results.append(row)
            print(
                "f{}: B1={} B2={} C={} top={} bot={}".format(
                    fid,
                    row["B1_head"]["covers_both"],
                    row["B2_two"]["covers_both"],
                    row["C_auto"]["found"],
                    [round(v, 1) for v in top],
                    [round(v, 1) for v in bot],
                )
            )

            # 可视化拼图
            canvases = []
            for label, m, cov in (
                ("A_box_a", m_a, None),
                ("A_box_b", m_b, None),
                ("B1_head", best_h[0], best_h),
                ("B2_two", best_2[0], best_2),
                ("C_auto", auto_both[0] if auto_both else np.zeros((h, w), bool), auto_both),
            ):
                vis = frame.copy()
                overlay = np.zeros_like(vis)
                overlay[m] = (0, 200, 255)
                cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
                for bb, col in ((a, (0, 255, 0)), (b, (0, 255, 0))):
                    x1, y1, x2, y2 = [int(v) for v in bb]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                if cov:
                    cv2.putText(
                        vis, f"{label} a={cov[1]:.2f} b={cov[2]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    )
                else:
                    cv2.putText(
                        vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                canvases.append(vis)
            grid = np.vstack([np.hstack(canvases[0:2]), np.hstack(canvases[2:4]), np.hstack([canvases[4], canvases[4]])])
            grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2))
            cv2.imwrite(str(out_dir / f"f{fid}.png"), grid)

    cap.release()
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8",
    )
    n_b1 = sum(r["B1_head"]["covers_both"] for r in results)
    n_b2 = sum(r["B2_two"]["covers_both"] for r in results)
    n_c = sum(r["C_auto"]["found"] for r in results)
    print(f"summary: n={len(results)} B1_head_both={n_b1} B2_two_both={n_b2} C_auto_both={n_c}")


if __name__ == "__main__":
    main()
