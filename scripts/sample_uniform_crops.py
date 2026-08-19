"""M3 标注集采样：从 pending/ 按 HSV 弱标签分层抽取（默认正负各 250 张）。"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run(
    out_dir: str | Path,
    *,
    per_class: int = 250,
    seed: int = 42,
) -> None:
    out_dir = Path(out_dir)
    meta_path = out_dir / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} 不存在，先运行 collect_uniform_crops.py")

    by_label: dict[int, list[str]] = {0: [], 1: []}
    with meta_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_label[int(row["hsv_weak_label"])].append(row["filename"])

    rng = random.Random(seed)
    sample_dir = out_dir / "sample"
    for cls, sub in (("negative", 0), ("positive", 1)):
        (sample_dir / cls).mkdir(parents=True, exist_ok=True)
        files = by_label.get(sub, [])
        picked = rng.sample(files, min(per_class, len(files)))
        for name in picked:
            src = out_dir / "pending" / name
            if src.exists():
                shutil.copy2(src, sample_dir / cls / name)
        print(f"{cls}: 抽 {len(picked)} 张（可用 {len(files)}）")
    print(f"标注集 -> {sample_dir}（positive=穿工装，negative=未穿工装）")


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 标注集采样")
    parser.add_argument("--out-dir", default="data/uniform_crops")
    parser.add_argument("--per-class", type=int, default=250)
    args = parser.parse_args()
    run(args.out_dir, per_class=args.per_class)


if __name__ == "__main__":
    main()
