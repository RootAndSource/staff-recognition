"""M4 数据准备：全身/可见框标注 → 上半身框标注。

转换规则：取检测框上部 ratio（默认 0.55，头到腰）作为上半身框。
输出到 data/finetune/annotations_upper/，保留原标注不动。
转换后建议人工抽查修正（脚本会生成抽查清单）。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def convert_file(ann: Path, out_dir: Path, ratio: float) -> int:
    lines = [l for l in ann.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue  # 跳过格式异常行
        cls, cx, cy, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # 归一化坐标 → 像素（以标注自身的坐标系：w/h 是相对图片的，这里只做框内比例变换）
        # 上半身框：顶部不变，底部收至 top + ratio*h
        new_cy = cy - (h * (1 - ratio)) / 2
        new_h = h * ratio
        if new_h <= 0:
            continue
        out_lines.append(f"{cls} {cx:.6f} {new_cy:.6f} {w:.6f} {new_h:.6f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ann.name).write_text("\n".join(out_lines), encoding="utf-8")
    return len(out_lines)


def run(
    ann_dir: str | Path,
    out_dir: str | Path,
    *,
    ratio: float = 0.55,
    sample_count: int = 60,
    seed: int = 42,
) -> None:
    ann_dir, out_dir = Path(ann_dir), Path(out_dir)
    anns = sorted(ann_dir.glob("*.txt"))
    total_boxes = 0
    for ann in anns:
        total_boxes += convert_file(ann, out_dir, ratio)
    print(f"转换完成：{len(anns)} 个标注文件，{total_boxes} 个框 -> {out_dir}")

    # 抽查清单：随机抽 sample_count 个"有框"的标注
    non_empty = [a for a in anns if any(l.strip() for l in a.read_text(encoding="utf-8").splitlines())]
    rng = random.Random(seed)
    picked = sorted(rng.sample(non_empty, min(sample_count, len(non_empty))))
    list_path = out_dir / "review_list.txt"
    list_path.write_text("\n".join(str(p.stem) for p in picked), encoding="utf-8")
    print(f"抽查清单（{len(picked)} 帧）-> {list_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 全身标注 → 上半身标注")
    parser.add_argument("--ann-dir", default="data/finetune/annotations")
    parser.add_argument("--out-dir", default="data/finetune/annotations_upper")
    parser.add_argument("--ratio", type=float, default=0.55, help="上半身高度比例（头到腰）")
    parser.add_argument("--sample", type=int, default=60, help="抽查帧数")
    args = parser.parse_args()
    run(args.ann_dir, args.out_dir, ratio=args.ratio, sample_count=args.sample)


if __name__ == "__main__":
    main()
