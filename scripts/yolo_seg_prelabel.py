"""M5 掩码预标（YOLO11-seg 版）：用预训练 YOLO11-seg 直接输出人体实例掩码。

与 SAM 预标相比，YOLO11-seg 端到端出掩码（不依赖检测框提示），
在 person 类别上的掩码通常更贴合人体，人工修正量更小。

输出（out_dir 下）：
  labelme/  每帧一个 JSON（预填 person 多边形），供 labelme 人工修正
  yolo_seg/ YOLO-seg 格式 txt（归一化多边形，可直接用于训练/质检）

用法示例：
  python scripts/yolo_seg_prelabel.py --frames-dir data/finetune/frames \
      --weights models/yolo11x-seg.pt --out-dir data/finetune/mask_annotations_yolo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def _simplify_polygon(poly: np.ndarray, max_points: int = 40) -> np.ndarray:
    """简化多边形顶点数（Douglas-Peucker），poly 为原图像素坐标 Nx2。"""
    if len(poly) <= max_points:
        return poly
    import cv2

    closed = np.asarray(poly, dtype=np.float32)
    if not np.allclose(closed[0], closed[-1]):
        closed = np.vstack([closed, closed[0]])
    eps = 0.01 * cv2.arcLength(closed, True)
    while True:
        approx = cv2.approxPolyDP(closed, eps, True).reshape(-1, 2)
        if len(approx) <= max_points or eps <= 1e-6:
            break
        eps *= 0.6
    return approx


def _write_labelme_json(
    out_json: Path, img_path: Path, img_w: int, img_h: int, polys_px: list[np.ndarray]
) -> None:
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
    lines = []
    for poly in polys_norm:
        pts = poly.reshape(-1)
        pts = [max(0.0, min(1.0, float(v))) for v in pts]
        lines.append("0 " + " ".join(f"{v:.6f}" for v in pts))
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 掩码预标（YOLO11-seg）")
    parser.add_argument("--frames-dir", default="data/finetune/frames")
    parser.add_argument("--weights", default="models/yolo11x-seg.pt")
    parser.add_argument("--out-dir", default="data/finetune/mask_annotations_yolo")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-det", type=int, default=40)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001, help="过滤极小掩码（相对整图面积）")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    labelme_dir = out_dir / "labelme"
    yolo_dir = out_dir / "yolo_seg"
    labelme_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 YOLO11-seg {args.weights} ...")
    model = YOLO(args.weights)

    imgs = sorted(frames_dir.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"{frames_dir} 下没有图片")
    print(f"共 {len(imgs)} 帧，开始预标 ...")

    n_poly = 0
    for i, img_path in enumerate(imgs, 1):
        res = model.predict(
            str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        h, w = res.orig_shape
        polys_px: list[np.ndarray] = []
        polys_norm: list[np.ndarray] = []
        if res.masks is not None and res.boxes is not None:
            cls = res.boxes.cls.cpu().numpy()
            area_ratios = np.asarray(res.boxes.xywh.cpu().numpy())[:, 2] * np.asarray(
                res.boxes.xywh.cpu().numpy()
            )[:, 3] / (w * h)
            for c, poly_xy, area_ratio in zip(cls, res.masks.xy, area_ratios):
                if int(c) != 0:  # 只保留 person
                    continue
                if area_ratio < args.min_area_ratio:
                    continue
                poly = _simplify_polygon(np.asarray(poly_xy, dtype=float))
                if len(poly) >= 3:
                    polys_px.append(poly)
                    polys_norm.append(poly / np.array([w, h], dtype=float))

        # 图片复制到 labelme 目录（labelme 只扫描同目录图片）
        img_dst = labelme_dir / img_path.name
        if not img_dst.exists():
            img_dst.write_bytes(img_path.read_bytes())
        _write_labelme_json(labelme_dir / f"{img_path.stem}.json", img_path, w, h, polys_px)
        _write_yolo_seg_txt(yolo_dir / f"{img_path.stem}.txt", w, h, polys_norm)
        n_poly += len(polys_px)

        if i % 50 == 0 or i == len(imgs):
            print(f"  进度 {i}/{len(imgs)}，累计掩码 {n_poly}")

    print(f"完成：{len(imgs)} 帧，共 {n_poly} 个掩码")
    print(f"  labelme 预标 -> {labelme_dir}")
    print(f"  yolo-seg 预标 -> {yolo_dir}")


if __name__ == "__main__":
    main()
