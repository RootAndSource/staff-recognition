"""M5 数据转换：labelme 多边形 JSON -> YOLO-seg 格式 txt。

用法示例：
  python scripts/labelme2yolo.py --json-dir data/finetune/mask_annotations/labelme \
      --out-dir data/finetune/mask_annotations/yolo_seg_fixed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_one(json_path: Path, out_txt: Path) -> int:
    """把单个 labelme JSON 转成 YOLO-seg txt，返回多边形数量。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w = int(data.get("imageWidth", 0))
    h = int(data.get("imageHeight", 0))
    if w <= 0 or h <= 0:
        # 兜底：从 imageData 无法取尺寸时跳过（应保证 JSON 含尺寸字段）
        return 0

    lines = []
    n = 0
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label != "person":
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        norm = []
        for x, y in pts:
            nx = max(0.0, min(1.0, float(x) / w))
            ny = max(0.0, min(1.0, float(y) / h))
            norm.extend((nx, ny))
        lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
        n += 1

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="labelme JSON -> YOLO-seg txt")
    parser.add_argument("--json-dir", default="data/finetune/mask_annotations/labelme")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir) if args.out_dir else json_dir.parent / "yolo_seg_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(json_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"{json_dir} 下没有 labelme JSON")

    total = 0
    for j in jsons:
        n = convert_one(j, out_dir / f"{j.stem}.txt")
        total += n
    print(f"完成：{len(jsons)} 个 JSON -> {out_dir}，共 {total} 个多边形")


if __name__ == "__main__":
    main()
