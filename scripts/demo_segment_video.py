"""M5 演示：检测器 + SAM2 在视频上输出人体实例掩码并渲染叠加视频。

用于在掩码标注完成前，先用最强的现成模型（SAM2.1 large）评估
斜俯视场景的实例分割效果（备选流程：检测器 + SAM2 box-prompt）。

用法示例：
  python scripts/demo_segment_video.py --source data/preprocessed/005.mp4 \
      --start-frame 900 --max-frames 300 \
      --out-video outputs/mp4/seg_demo_005.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO


def render_masks(
    frame: cv2.Mat,
    masks_data: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
) -> cv2.Mat:
    """把掩码叠加渲染到帧上（半透明填充 + 轮廓 + 检测框 + 置信度）。"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for i, mask in enumerate(masks_data):
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        # 半透明填充（绿色）
        colored = np.zeros_like(frame)
        colored[:, :, 1] = 120
        overlay[mask_u8 > 0] = cv2.addWeighted(
            overlay[mask_u8 > 0], 0.6, colored[mask_u8 > 0], 0.4, 0
        )
        # 轮廓
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        # 检测框 + 置信度
        if i < len(boxes):
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 1)
            label = f"{confs[i]:.2f}"
            cv2.putText(
                overlay, label, (x1, max(20, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="检测器 + SAM2 视频分割演示")
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--det-weights", default="models/yolo11x_finetuned.pt")
    parser.add_argument("--sam-weights", default="models/sam2.1_l.pt")
    parser.add_argument("--out-video", default="outputs/mp4/seg_demo.mp4")
    parser.add_argument("--start-frame", type=int, default=900)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-det", type=int, default=40)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    print(f"加载检测器 {args.det_weights} ...")
    det = YOLO(args.det_weights)
    print(f"加载 SAM {args.sam_weights} ...")
    sam = SAM(args.sam_weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    t0 = time.time()
    processed = 0
    while processed < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        det_res = det(frame, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, verbose=False)[0]
        if det_res.boxes is None or len(det_res.boxes) == 0:
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            processed += 1
            continue

        boxes = det_res.boxes.xyxy.cpu().numpy()
        confs = det_res.boxes.conf.cpu().numpy()
        sam_res = sam(frame, bboxes=boxes, verbose=False, device=args.device)[0]
        if sam_res.masks is None or len(sam_res.masks) == 0:
            vis = frame
        else:
            vis = render_masks(frame, sam_res.masks.data.cpu().numpy(), boxes, confs)

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(vis)
        processed += 1

        if processed % 30 == 0:
            el = time.time() - t0
            print(f"  进度 {processed}/{args.max_frames}，耗时 {el:.0f}s，"
                  f"平均 {el/processed:.2f}s/帧")

    cap.release()
    if writer is not None:
        writer.release()
    print(f"完成：处理 {processed} 帧，输出 {out_path}（总耗时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
