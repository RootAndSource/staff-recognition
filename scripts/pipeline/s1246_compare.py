"""S1246 对比视频：原图 / 无记忆(rule) / 记忆(mem) / 长记忆(memlong) 四格同屏。"""

from __future__ import annotations

import argparse
import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", default=[
        "outputs/mp4/s1246_005_6min_rule.mp4",
        "outputs/mp4/s1246_005_6min_mem.mp4",
        "outputs/mp4/s1246_005_6min_memlong.mp4",
    ])
    parser.add_argument("--labels", nargs="+", default=[
        "无记忆(rule)", "记忆(mem)", "长记忆(memlong)",
    ])
    parser.add_argument("--source", default="data/preprocessed/005.mp4")
    parser.add_argument("--start", type=int, default=10800)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--out", default="outputs/mp4/s1246_compare.mp4")
    args = parser.parse_args()

    caps = [cv2.VideoCapture(v) for v in args.videos]
    src = cv2.VideoCapture(args.source)
    fps = src.get(cv2.CAP_PROP_FPS) or 25.0
    src.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    w = int(src.get(3)); h = int(src.get(4))
    cell_w, cell_h = w // 2, h // 2

    writer = None
    n = 0
    while n < args.frames:
        ok_src, frame_src = src.read()
        if not ok_src:
            break
        ok_v = [cap.read() for cap in caps]
        if not all(o for o, _ in ok_v):
            break
        frames = [f for _, f in ok_v]
        cells = [cv2.resize(frame_src, (cell_w, cell_h))]
        for frame, label in zip(frames, args.labels):
            cell = cv2.resize(frame, (cell_w, cell_h))
            cv2.putText(cell, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(cell)
        canvas = np.zeros((2 * cell_h, 2 * cell_w, 3), np.uint8)
        canvas[:cell_h, :cell_w] = cells[0]
        canvas[:cell_h, cell_w:] = cells[1]
        canvas[cell_h:, :cell_w] = cells[2]
        canvas[cell_h:, cell_w:] = cells[3]
        if writer is None:
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * cell_w, 2 * cell_h)
            )
        writer.write(canvas)
        n += 1
        if n % 300 == 0:
            print(f"progress {n}/{args.frames}")
    for cap in caps:
        cap.release()
    src.release()
    if writer is not None:
        writer.release()
    print(f"done -> {args.out} frames={n}")


if __name__ == "__main__":
    main()
