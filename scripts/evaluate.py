"""评估：抽帧 → 人工标注 CSV → 指标（工装准确率/精确率/召回率、ID 稳定性）。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from src.video_input import VideoSource


def sample_frames(video: str, sample_every: int, sample_count: int, out_dir: Path) -> list[Path]:
    """从视频中均匀抽帧存为 PNG，返回文件列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with VideoSource(video) as vs:
        for i, info in enumerate(vs.frames()):
            if i % sample_every != 0:
                continue
            if len(saved) >= sample_count:
                break
            p = out_dir / f"frame_{info.frame_id:06d}.png"
            cv2.imwrite(str(p), info.frame)
            saved.append(p)
    return saved


def write_annotation_template(csv_path: Path, frame_paths: list[Path]) -> None:
    """生成人工标注 CSV 模板：每行一帧，记录每人的工装标签。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_file", "person_index", "gt_uniform", "note"])
        for p in frame_paths:
            writer.writerow([p.name, 1, "", ""])


def run_evaluate(config: dict, video: str | None = None, sample_count: int = 100) -> None:
    """评估入口：当前先完成抽帧与标注模板生成，指标统计待标注回填后计算。"""
    ev = config["evaluate"]
    if video is None:
        candidates = sorted(Path("data").glob("*.mp4")) + sorted(
            Path("data").glob("*.avi")
        )
        if not candidates:
            raise FileNotFoundError("未指定 --video，且 data 下没有视频")
        video = str(candidates[0])

    frames_dir = Path("outputs/evaluate/eval_frames")
    paths = sample_frames(
        video,
        sample_every=ev["sample_every_n_frames"],
        sample_count=sample_count,
        out_dir=frames_dir,
    )
    write_annotation_template(Path(ev["annotation_csv"]), paths)
    print(f"抽帧完成：{len(paths)} 张 -> {frames_dir}")
    print(f"标注模板已生成 -> {ev['annotation_csv']}（人工填 gt_uniform 后重新运行以计算指标）")
