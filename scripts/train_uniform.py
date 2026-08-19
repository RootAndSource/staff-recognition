"""M3 工装分类器训练：MobileNetV3-Small 二分类（工装/非工装）。

数据约定：data/uniform_crops/sample/ 下 positive/（穿工装）、negative/（未穿工装）。
人工把 pending 或 sample 里的图片归类到这两个目录后运行。
输出：models/uniform_cls.pt（完整 checkpoint）与验证准确率。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class CropDataset(Dataset):
    def __init__(self, root: Path, transform) -> None:
        self.samples: list[tuple[Path, int]] = []
        for cls, label in (("positive", 1), ("negative", 0)):
            d = root / cls
            if d.exists():
                for p in sorted(d.glob("*.jpg")):
                    self.samples.append((p, label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 工装分类器训练")
    parser.add_argument("--data", default="data/uniform_crops/sample")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="models/uniform_cls.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.data)
    n_pos = len(list((root / "positive").glob("*.jpg"))) if (root / "positive").exists() else 0
    n_neg = len(list((root / "negative").glob("*.jpg"))) if (root / "negative").exists() else 0
    print(f"数据：positive={n_pos} negative={n_neg} 共 {n_pos + n_neg}")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("缺少类别数据，先完成标注归类")
    if args.dry_run:
        return

    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25)),  # 遮挡鲁棒
    ])
    dataset = CropDataset(root, transform)
    n_val = max(1, int(len(dataset) * 0.15))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    else:
        device = torch.device("cpu")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)
    # 类别权重（inverse frequency），缓解工装/非工装样本不平衡
    counts = torch.tensor([n_neg, n_pos], dtype=torch.float32)
    total = counts.sum()
    class_weight = total / (2.0 * counts.clamp(min=1))
    class_weight = class_weight.to(device)
    print(f"类别权重（negative/positive）: {class_weight.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"训练 {args.epochs} epochs（train={len(train_ds)} val={len(val_ds)}）...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        acc = correct / total

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += x.size(0)
        v_acc = v_correct / v_total
        print(f"epoch {epoch:3d}: train_acc={acc:.4f} val_acc={v_acc:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": ["negative", "positive"],
            "imgsz": args.imgsz,
        },
        args.out,
    )
    print(f"模型已保存 -> {args.out}（val_acc={v_acc:.4f}）")


if __name__ == "__main__":
    main()
