"""M5 掩码预标：检测框 + SAM box-prompt 生成人体实例掩码。

输出（out_dir 下）：
  labelme/  每帧一个 JSON（预填 person 多边形），供 labelme 人工修正
  yolo_seg/ YOLO-seg 格式 txt（归一化多边形，可直接用于训练/质检）

框来源（二选一）：
  --ann-dir  已有 YOLO 框标注目录（推荐，M2/M4 人工修正过的全身框，语义与掩码一致）
  --det-weights  检测器自动出框（默认全身检测器 yolo11x_finetuned.pt）

用法示例：
  python scripts/prepare_mask_annotations.py --frames-dir data/finetune/frames \
      --ann-dir data/finetune/annotations --sam-weights models/sam2.1_l.pt \
      --out-dir data/finetune/mask_annotations_saml
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO


def _to_polygons(masks_xyn: np.ndarray) -> list[np.ndarray]:
    """把 SAM 的归一化分段（Nx2 列表）转成绝对像素多边形。"""
    polys = []
    for seg in masks_xyn:
        if len(seg) < 3:
            continue
        polys.append(np.asarray(seg, dtype=float))
    return polys


def _simplify_polygon(poly: np.ndarray, img_w: int, img_h: int, max_points: int = 40) -> np.ndarray:
    """用 Douglas-Peucker 简化多边形，限制顶点数（方便 labelme 编辑与训练）。"""
    if len(poly) <= max_points:
        return poly
    px = poly * np.array([img_w, img_h], dtype=float)
    closed = np.vstack([px, px[0]]).astype(np.float32)
    eps = 0.01 * cv2.arcLength(closed, True)  # 按周长 1% 简化，逐步收紧直到达标
    while True:
        approx = cv2.approxPolyDP(closed, eps, True).reshape(-1, 2)
        if len(approx) <= max_points or eps <= 1e-6:
            break
        eps *= 0.6
    norm = approx / np.array([img_w, img_h], dtype=float)
    return norm


def _write_labelme_json(
    out_json: Path,
    img_path: Path,
    img_w: int,
    img_h: int,
    polys_px: list[list[list[float]]],
) -> None:
    """写 labelme 兼容 JSON（shapes 为 person 多边形，绝对像素坐标）。"""
    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [
            {
                "label": "person",
                "points": [[float(x), float(y)] for x, y in poly],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
            for poly in polys_px
        ],
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _write_yolo_seg_txt(out_txt: Path, img_w: int, img_h: int, polys_norm: list[np.ndarray]) -> None:
    """写 YOLO-seg 格式 txt：class + 归一化 xy 坐标（类 0=person）。"""
    lines = []
    for poly in polys_norm:
        pts = poly.reshape(-1)
        pts = [max(0.0, min(1.0, float(v))) for v in pts]
        lines.append("0 " + " ".join(f"{v:.6f}" for v in pts))
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 掩码预标（检测 + SAM）")
    parser.add_argument("--frames-dir", default="data/finetune/frames")
    parser.add_argument(
        "--ann-dir",
        default=None,
        help="已有 YOLO 框标注目录（推荐）；不提供则用检测器出框",
    )
    parser.add_argument("--det-weights", default="models/yolo11x_finetuned.pt")
    parser.add_argument("--sam-weights", default="models/sam2.1_l.pt")
    parser.add_argument("--out-dir", default="data/finetune/mask_annotations")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-det", type=int, default=40)
    parser.add_argument("--min-box-size", type=int, default=20, help="过滤过小检测框（像素短边）")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    labelme_dir = out_dir / "labelme"
    yolo_dir = out_dir / "yolo_seg"
    labelme_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    det = None
    ann_dir = Path(args.ann_dir) if args.ann_dir else None
    if ann_dir is not None:
        if not ann_dir.exists():
            raise FileNotFoundError(f"标注目录不存在：{ann_dir}")
        print(f"使用已有框标注：{ann_dir}（不加载检测器）")
    else:
        print(f"加载检测器 {args.det_weights} ...")
        det = YOLO(args.det_weights)
    print(f"加载 SAM {args.sam_weights} ...")
    sam = SAM(args.sam_weights)

    imgs = sorted(frames_dir.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"{frames_dir} 下没有图片")
    print(f"共 {len(imgs)} 帧，开始预标 ...")

    n_poly = 0
    for i, img_path in enumerate(imgs, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [跳过] 无法读取 {img_path.name}")
            continue
        h, w = frame.shape[:2]

        # 复制图片到 labelme 目录（labelme 只扫描同目录图片）
        img_dst = labelme_dir / img_path.name
        if not img_dst.exists():
            img_dst.write_bytes(img_path.read_bytes())

        if ann_dir is not None:
            ann_txt = ann_dir / f"{img_path.stem}.txt"
            if not ann_txt.exists():
                _write_labelme_json(labelme_dir / f"{img_path.stem}.json", img_path, w, h, [])
                _write_yolo_seg_txt(yolo_dir / f"{img_path.stem}.txt", w, h, [])
                continue
            xyxy = []
            for line in ann_txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(v) for v in parts[:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                xyxy.append([x1, y1, x2, y2])
            xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        else:
            det_res = det(frame, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, verbose=False)[0]
            boxes = det_res.boxes
            if boxes is None or len(boxes) == 0:
                _write_labelme_json(labelme_dir / f"{img_path.stem}.json", img_path, w, h, [])
                _write_yolo_seg_txt(yolo_dir / f"{img_path.stem}.txt", w, h, [])
                continue
            xyxy = boxes.xyxy.cpu().numpy()

        # 过滤过小框（斜俯视下小目标仍有意义，阈值取保守值）
        keep = (xyxy[:, 2] - xyxy[:, 0] >= args.min_box_size) & (
            xyxy[:, 3] - xyxy[:, 1] >= args.min_box_size
        )
        xyxy = xyxy[keep]
        if len(xyxy) == 0:
            _write_labelme_json(labelme_dir / f"{img_path.stem}.json", img_path, w, h, [])
            _write_yolo_seg_txt(yolo_dir / f"{img_path.stem}.txt", w, h, [])
            continue

        sam_res = sam(frame, bboxes=xyxy, verbose=False, device=args.device)[0]
        if sam_res.masks is None or len(sam_res.masks) == 0:
            _write_labelme_json(labelme_dir / f"{img_path.stem}.json", img_path, w, h, [])
            _write_yolo_seg_txt(yolo_dir / f"{img_path.stem}.txt", w, h, [])
            continue

        # 每个 box 一个掩码（SAM 按 box 顺序输出），简化多边形
        polys_norm = []
        for seg in sam_res.masks.xyn:
            if len(seg) >= 3:
                polys_norm.append(_simplify_polygon(seg, w, h))
        polys_px = [
            [[float(x) * w, float(y) * h] for x, y in seg]
            for seg in polys_norm
        ]
        _write_labelme_json(labelme_dir / f"{img_path.stem}.json", img_path, w, h, polys_px)
        _write_yolo_seg_txt(yolo_dir / f"{img_path.stem}.txt", w, h, polys_norm)
        n_poly += len(polys_norm)

        if i % 50 == 0 or i == len(imgs):
            print(f"  进度 {i}/{len(imgs)}，累计掩码 {n_poly}")

    print(f"完成：{len(imgs)} 帧，共 {n_poly} 个掩码")
    print(f"  labelme 预标 -> {labelme_dir}")
    print(f"  yolo-seg 预标 -> {yolo_dir}")


if __name__ == "__main__":
    main()
