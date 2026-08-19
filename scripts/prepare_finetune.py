"""M2 数据准备：从现场视频均匀抽帧，生成检测微调/标注用数据集。

输出：
- data/finetune/frames/*.jpg      抽帧图片
- data/finetune/annotations/      伪标签（YOLO 格式 person 框，可选 --auto-label）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.detector import PersonDetector


def run(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    per_video: int = 25,
    auto_label: bool = False,
    det_weights: str = "models/yolo11x.pt",
) -> None:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    ann_dir = out_dir / "annotations"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if auto_label:
        ann_dir.mkdir(parents=True, exist_ok=True)
        det = PersonDetector(weights=det_weights, conf_thres=0.25, imgsz=960)
        det.load()

    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov")
    )
    print(f"找到 {len(videos)} 个视频，每个抽 {per_video} 帧...")
    total = 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"  无法打开 {v.name}，跳过")
            continue
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(count // per_video, 1)
        targets = list(range(0, count, step))[:per_video]
        for fid in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"{v.stem}_{fid:06d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame)
            if auto_label:
                dets = det.detect(frame)
                h, w = frame.shape[:2]
                lines = []
                for d in dets:
                    x1, y1, x2, y2 = d.xyxy
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (ann_dir / f"{Path(name).stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
            total += 1
        cap.release()
        print(f"  {v.name}: 抽 {len(targets)} 帧")
    print(f"完成：共 {total} 帧 -> {frames_dir}")
    if auto_label:
        print(f"伪标签（yolo11x person 框）-> {ann_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 抽帧与初始伪标签")
    parser.add_argument("--input-dir", default="data/preprocessed")
    parser.add_argument("--out-dir", default="data/finetune")
    parser.add_argument("--per-video", type=int, default=25)
    parser.add_argument("--auto-label", action="store_true",
                        help="用 yolo11x 自动生成 person 伪标签（YOLO 格式）")
    args = parser.parse_args()
    run(
        args.input_dir,
        args.out_dir,
        per_video=args.per_video,
        auto_label=args.auto_label,
    )


if __name__ == "__main__":
    main()
