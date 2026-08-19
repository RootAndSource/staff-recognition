"""M2 检测微调：ultralytics YOLO 训练入口。

数据约定：data/finetune/ 下 frames/（jpg）与 annotations/（YOLO 格式 txt，类 0=person）。
运行前建议先用 CVAT 等工具人工修正伪标签（补坐姿/遮挡漏检、删雨伞等误检）。
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def prepare_dataset(
    frames_dir: str | Path,
    ann_dir: str | Path,
    out_dir: str | Path,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """把 frames/annotations 划分为 train/val 并生成 data.yaml。"""
    frames_dir, ann_dir, out_dir = Path(frames_dir), Path(ann_dir), Path(out_dir)
    imgs = sorted(frames_dir.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"{frames_dir} 下没有图片")
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
            ann = ann_dir / f"{img.stem}.txt"
            if ann.exists():
                shutil.copy2(ann, lab_out / ann.name)
            else:
                (lab_out / f"{img.stem}.txt").write_text("", encoding="utf-8")

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
    parser = argparse.ArgumentParser(description="M2 检测微调训练")
    parser.add_argument("--data", default="data/finetune")
    parser.add_argument("--weights", default="models/yolo11x.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15, help="验证集无提升时早停的 epoch 数")
    parser.add_argument("--dry-run", action="store_true", help="只准备数据并打印命令，不训练")
    args = parser.parse_args()

    data_root = Path(args.data)
    data_yaml = prepare_dataset(
        data_root / "frames",
        data_root / "annotations",
        data_root / "dataset",
        val_ratio=args.val_ratio,
    )

    cmd = (
        f"yolo detect train model={args.weights} data={data_yaml} "
        f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
        f"device={args.device} patience={args.patience} project=runs/finetune name=person "
        # Random Erasing（遮挡鲁棒）等增强由 ultralytics 默认开启的 augment 参数控制，
        # 也可追加 erasing=0.4 mosaic=1.0 copy_paste=0.3
    )
    print("训练命令：")
    print("  " + cmd)
    if not args.dry_run:
        from ultralytics import YOLO

        model = YOLO(args.weights)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
            project="runs/finetune",
            name="person",
            erasing=0.4,
            mosaic=1.0,
            copy_paste=0.3,
        )


if __name__ == "__main__":
    main()
