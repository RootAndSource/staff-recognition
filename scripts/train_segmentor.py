"""M5 分割微调：YOLO11-seg 人体实例分割训练入口。

数据约定：
  data/finetune/frames/            帧图（jpg）
  data/finetune/mask_annotations/labelme/     labelme 多边形 JSON（人工修正后）
  data/finetune/mask_annotations/yolo_seg/     YOLO-seg 预标 txt（SAM 自动生成）

运行前先用 labelme 修正 SAM 预标（或直接使用 yolo_seg 预标做初版）。

用法示例：
  python scripts/train_segmentor.py --ann-dir data/finetune/mask_annotations/labelme \
      --weights models/yolo11x-seg.pt --epochs 50 --batch 4
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _to_yolo_seg(ann_dir: Path, out_dir: Path) -> None:
    """若标注目录是 labelme JSON，先转成 YOLO-seg txt；否则视为已有 txt 直接复制。"""
    from labelme2yolo import convert_one

    jsons = sorted(ann_dir.glob("*.json"))
    if jsons:
        print(f"检测到 labelme JSON {len(jsons)} 个，转换为 YOLO-seg ...")
        for j in jsons:
            convert_one(j, out_dir / f"{j.stem}.txt")
        return
    txts = sorted(ann_dir.glob("*.txt"))
    for t in txts:
        shutil.copy2(t, out_dir / t.name)


def prepare_mask_dataset(
    frames_dir: str | Path,
    ann_dir: str | Path,
    out_dir: str | Path,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """把 frames + 掩码标注划分为 train/val 并生成 data.yaml。"""
    frames_dir, ann_dir, out_dir = Path(frames_dir), Path(ann_dir), Path(out_dir)
    imgs = sorted(frames_dir.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"{frames_dir} 下没有图片")

    tmp_labels = out_dir / "_tmp_labels"
    tmp_labels.mkdir(parents=True, exist_ok=True)
    _to_yolo_seg(ann_dir, tmp_labels)

    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

    for split, subset in (("train", train_imgs), ("val", val_imgs)):
        img_out = out_dir / "images" / split
        lab_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy2(img, img_out / img.name)
            ann = tmp_labels / f"{img.stem}.txt"
            if ann.exists():
                shutil.copy2(ann, lab_out / ann.name)
            else:
                (lab_out / f"{img.stem}.txt").write_text("", encoding="utf-8")

    shutil.rmtree(tmp_labels, ignore_errors=True)
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )
    print(f"数据集划分完成：train={len(train_imgs)} val={len(val_imgs)} -> {out_dir}")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 YOLO11-seg 微调")
    parser.add_argument("--data", default="data/finetune")
    parser.add_argument(
        "--ann-dir",
        default="data/finetune/mask_annotations/labelme",
        help="掩码标注目录（labelme JSON 或 YOLO-seg txt）",
    )
    parser.add_argument("--weights", default="models/yolo11x-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="person-mask")
    parser.add_argument("--dry-run", action="store_true", help="只准备数据并打印命令，不训练")
    args = parser.parse_args()

    data_root = Path(args.data)
    ann_dir = Path(args.ann_dir)
    if not ann_dir.exists():
        # 回退到 SAM 预标目录
        fallback = data_root / "mask_annotations" / "yolo_seg"
        if fallback.exists():
            ann_dir = fallback
            print(f"指定的标注目录不存在，使用 SAM 预标：{ann_dir}")
        else:
            raise FileNotFoundError(f"标注目录不存在：{ann_dir}，请先运行 prepare_mask_annotations.py")

    out_dir = data_root / "dataset_seg"
    data_yaml = prepare_mask_dataset(
        data_root / "frames", ann_dir, out_dir, val_ratio=args.val_ratio
    )

    cmd = (
        f"python -m ultralytics yolo segment train model={args.weights} "
        f"data={data_yaml} epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
        f"device={args.device} patience={args.patience} project={args.project} name={args.name}"
    )
    print(f"训练命令：\n  {cmd}")
    if args.dry_run:
        return

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
