"""M7 标注抽帧：从视频均匀抽帧，供人工标"人 + 工装"。

输出：data/finetune/frames_m7/{vid}_{frameid:06d}.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 标注抽帧")
    parser.add_argument("--video-dir", default="data/preprocessed")
    parser.add_argument("--videos", default="005,001,002,003,004", help="逗号分隔视频编号")
    parser.add_argument("--per-video", type=int, default=30, help="每视频抽帧数")
    parser.add_argument("--out-dir", default="data/finetune/frames_m7")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid in args.videos.split(","):
        vid = vid.strip()
        src = video_dir / f"{vid}.mp4"
        if not src.exists():
            print(f"skip {src} (missing)")
            continue
        cap = cv2.VideoCapture(str(src))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            print(f"skip {src} (no frames)")
            cap.release()
            continue
        n = min(args.per_video, total)
        step = total / n
        written = 0
        for i in range(n):
            fid = int(i * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"{vid}_{fid:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += 1
        cap.release()
        print(f"{vid}: {total} frames -> {written} samples")


if __name__ == "__main__":
    main()
