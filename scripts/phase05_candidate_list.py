"""把 Phase 0.5 扫描产物整理成目检清单（Markdown）。

输出：out-dir/候选帧清单.md，每行一帧（frame_id、两框、空间特征、
缩略图文件名），供人工翻图挑选"上半身 + 脚"同人分裂帧。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Phase 0.5 候选目检清单")
    parser.add_argument("--out-dir", default="outputs/phase05")
    parser.add_argument("--min-gap", type=float, default=0.15)
    parser.add_argument("--max-gap", type=float, default=0.45)
    parser.add_argument("--min-xa", type=float, default=0.7)
    parser.add_argument("--min-hr", type=float, default=0.1)
    parser.add_argument("--max-hr", type=float, default=0.5)
    parser.add_argument("--dedup-window", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cand_path = out_dir / "candidates.jsonl"
    vis_dir = out_dir / "vis"
    if not cand_path.exists():
        raise SystemExit(f"missing {cand_path}")

    vis_map: dict[int, list[str]] = {}
    if vis_dir.exists():
        for name in sorted(os.listdir(vis_dir)):
            if not name.endswith(".png") or not name.startswith("f"):
                continue
            try:
                fid = int(name.split("_")[0][1:])
            except ValueError:
                continue
            vis_map.setdefault(fid, []).append(name)

    rows = []
    for line in open(cand_path, encoding="utf-8"):
        c = json.loads(line)
        sp = c["spatial"]
        if not (
            args.min_gap <= sp["gap_ratio"] <= args.max_gap
            and sp["x_align"] >= args.min_xa
            and args.min_hr <= sp["height_ratio"] <= args.max_hr
        ):
            continue
        vis_names = c.get("vis") or vis_map.get(c["frame_id"], [])
        rows.append(
            {
                "frame_id": c["frame_id"],
                "box_a": c["box_a"],
                "box_b": c["box_b"],
                "gap_ratio": sp["gap_ratio"],
                "x_align": sp["x_align"],
                "height_ratio": sp["height_ratio"],
                "vis": ", ".join(f"`{v}`" for v in vis_names),
            }
        )
    rows.sort(key=lambda r: r["frame_id"])

    # 场景去重：同位置（两框中心 y 量化到 30px）且时间窗口内只留一帧
    deduped = []
    kept_buckets: set[tuple[int, int]] = set()
    for r in rows:
        cy_a = (r["box_a"][1] + r["box_a"][3]) / 2
        cy_b = (r["box_b"][1] + r["box_b"][3]) / 2
        key = (int(cy_a // 30), int(cy_b // 30))
        near = any(
            abs(r["frame_id"] - k[0]) <= args.dedup_window and k[1] == key
            for k in kept_buckets
        )
        if near:
            continue
        kept_buckets.add((r["frame_id"], key))
        deduped.append(r)
    rows = deduped

    md = ["# Phase 0.5 候选帧目检清单", ""]
    md.append(f"共 {len(rows)} 个候选对（垂直堆叠 + x 对齐 + 遮挡间隙）。")
    md.append("请翻图（`vis/` 目录，文件名 `f{frame_id}.png`），挑出**同一人被桌子")
    md.append("挡成'上半身 + 脚'两段**的帧，把 frame_id 告诉我。")
    md.append("")
    md.append("| frame_id | gap_ratio | x_align | 高比 | 缩略图 |")
    md.append("| --- | --- | --- | --- | --- |")
    for r in rows:
        md.append(
            f"| {r['frame_id']} | {r['gap_ratio']:.3f} | {r['x_align']:.3f} "
            f"| {r['height_ratio']:.2f} | {r['vis']} |"
        )
    md.append("")
    out_path = out_dir / "候选帧清单.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"written {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
