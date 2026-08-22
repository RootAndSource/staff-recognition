"""Phase 0.5 视频模式实验（SAM 2.1 官方 video predictor）。

对目标分裂帧：截取 [fid-half, fid+half] 片段，首帧只给"头框"提示，
向后续帧传播掩码，检查在分裂帧处掩码是否覆盖"脚框"区域
（即 SAM 2 能否跨遮挡/跨帧输出完整人体掩码）。

用法：
  python scripts/phase05_video_test.py --fid 1035 \
      --box-head 1471,208,1578,387 --box-foot 214,946,529,1079 \
      --out-dir outputs/phase05/video_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam2.build_sam import build_sam2_video_predictor

from src.detector import PersonDetector


def _box_pixels_in_mask(mask: np.ndarray, box: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (x2 - x1) * (y2 - y1)
    return float(mask[y1:y2, x1:x2].sum() / area) if area > 0 else 0.0


def _write_frame_dir(source: str, fid: int, half: int, out_dir: Path) -> tuple[list[int], Path]:
    """从视频抽取 [fid-half, fid+half] 帧为 JPEG 目录，返回真实 frame_id 列表。"""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid - half))
    ids: list[int] = []
    idx = 0
    start = max(0, fid - half)
    while idx <= 2 * half:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ids.append(start + idx)
        idx += 1
    cap.release()
    return ids, frames_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0.5 视频模式实验")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--fid", type=int, required=True)
    parser.add_argument("--box-head", required=True, help="x1,y1,x2,y2")
    parser.add_argument("--box-foot", required=True, help="x1,y1,x2,y2")
    parser.add_argument("--half", type=int, default=30)
    parser.add_argument("--out-dir", default="outputs/phase05/video_test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    head = np.array([float(v) for v in args.box_head.split(",")])
    foot = np.array([float(v) for v in args.box_foot.split(",")])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, frames_dir = _write_frame_dir(args.source, args.fid, args.half, out_dir)
    if args.fid not in ids:
        raise SystemExit(f"fid {args.fid} not in extracted window {ids[0]}..{ids[-1]}")
    rel_fid = ids.index(args.fid)

    # 对片段首帧跑检测器，自动匹配目标头框（人可能已移动，不能用分裂帧坐标直接提示）
    det = PersonDetector(
        weights="models/yolo11x_finetuned.pt",
        conf_thres=0.3,
        iou_thres=0.5,
        imgsz=960,
        device=args.device,
    )
    first = cv2.imread(str(frames_dir / "000000.jpg"))
    dets = det.detect(first)
    if not dets:
        raise SystemExit("no detection on first frame")
    hc = np.array([(head[0] + head[2]) / 2, (head[1] + head[3]) / 2])
    best_d, best_d2 = None, float("inf")
    for d in dets:
        dc = np.array([(d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2])
        dist = float(np.hypot(*(dc - hc)))
        if dist < best_d2:
            best_d, best_d2 = d, dist
    head_first = best_d.xyxy
    print(f"first-frame head box matched: {[round(v, 1) for v in head_first]} "
          f"(dist={best_d2:.0f}px vs split-frame head)")

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "models/sam2.1_l.pt",
        device=args.device,
        apply_postprocessing=False,
        fill_hole_area=0,
    )

    t0 = time.time()
    state = predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=False)
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=0, box=head_first)
    print(f"init + prompt ok in {time.time() - t0:.1f}s, frames={len(ids)}")

    rows = []
    vis_rows = []
    for fidx, obj_ids, mask_tensor in predictor.propagate_in_video(state):
        mask = mask_tensor[0, 0].cpu().numpy() > 0.0
        cov_head = _box_pixels_in_mask(mask, head)
        cov_foot = _box_pixels_in_mask(mask, foot)
        rows.append(
            {
                "fid": ids[fidx],
                "rel": fidx,
                "covers_head": round(cov_head, 3),
                "covers_foot": round(cov_foot, 3),
            }
        )
        # 保存关键帧可视化（分裂帧 ±3 帧）
        if abs(fidx - rel_fid) <= 3:
            frame = cv2.imread(str(frames_dir / f"{fidx:06d}.jpg"))
            vis = frame.copy()
            overlay = np.zeros_like(vis)
            overlay[mask] = (0, 200, 255)
            cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
            for bb, name in ((head, "H"), (foot, "F")):
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis, f"{name} {_box_pixels_in_mask(mask, bb):.2f}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            cv2.putText(
                vis, f"fid={ids[fidx]} rel={fidx}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            vis_rows.append(vis)

    # 拼图
    if vis_rows:
        n = len(vis_rows)
        if n == 1:
            grid = vis_rows[0]
        elif n == 2:
            grid = np.hstack(vis_rows)
        else:
            half_n = (n + 1) // 2
            top = np.hstack(vis_rows[:half_n])
            bot = np.hstack(vis_rows[half_n:])
            if top.shape[1] != bot.shape[1]:
                scale = top.shape[1] / bot.shape[1]
                bot = cv2.resize(bot, (top.shape[1], int(bot.shape[0] * scale)))
            grid = np.vstack([top, bot])
        cv2.imwrite(str(out_dir / f"f{args.fid}.png"), grid)

    (out_dir / f"f{args.fid}_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    target = next((r for r in rows if r["fid"] == args.fid), None)
    print(
        f"done: target fid={args.fid} covers_foot={target['covers_foot'] if target else 'N/A'} "
        f"covers_head={target['covers_head'] if target else 'N/A'} "
        f"({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
