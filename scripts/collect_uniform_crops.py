"""M3 数据准备：从现场视频收集上半身 crop（工装/非工装分类数据集）。

输出：
- data/uniform_crops/pending/   待人工归类的 crop（jpg，命名 video_frame_personid.jpg）
- data/uniform_crops/meta.csv   弱标签（HSV 判定，仅供参考，不作为最终标签）

人工标注：把 pending/ 下的图片移动到 positive/（穿工装）或 negative/（未穿工装）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.detector import PersonDetector
from src.tracker import ByteTrackTracker
from src.uniform import UniformClassifier, _RoiSpec
from src.video_input import VideoSource


def run(
    video_dir: str | Path,
    out_dir: str | Path,
    *,
    per_video_frames: int = 2000,
    sample_every: int = 30,
    min_box_area: int = 800,
    det_weights: str = "models/yolo11x.pt",
) -> None:
    video_dir, out_dir = Path(video_dir), Path(out_dir)
    pending = out_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (out_dir / "positive").mkdir(parents=True, exist_ok=True)
    (out_dir / "negative").mkdir(parents=True, exist_ok=True)

    det = PersonDetector(weights=det_weights, conf_thres=0.25, imgsz=960)
    yaml_path = ByteTrackTracker.write_tracker_yaml(
        "configs/botsort.yaml",
        track_buffer=150,
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        tracker_type="botsort",
        with_reid=True,
        reid_model="auto",
    )
    trk = ByteTrackTracker(detector=det, tracker_yaml=str(yaml_path))
    det.load()
    clf = UniformClassifier(
        roi_main=_RoiSpec(0.10, 0.15, 0.90, 0.55),
        roi_lower=_RoiSpec(0.10, 0.45, 0.90, 0.85),
        hsv_ranges=((0, 10), (170, 180)),
        s_min=70, v_min=70,
        red_ratio_threshold=0.08,
        normalize=True,
    )

    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov")
    )
    print(f"共 {len(videos)} 个视频，每个最多采 {per_video_frames // sample_every} 次...")
    meta_rows: list[tuple[str, int, int, int, float, float, int]] = []
    total = 0
    for v in videos:
        count_in_video = 0
        with VideoSource(str(v)) as vs:
            for info in vs.frames():
                if count_in_video >= per_video_frames:
                    break
                if info.frame_id % sample_every != 0:
                    continue
                objs = trk.update(info.frame, detections=[])
                for o in objs:
                    if not o.is_confirmed:
                        continue
                    x1, y1, x2, y2 = [int(vv) for vv in o.xyxy]
                    if (x2 - x1) * (y2 - y1) < min_box_area:
                        continue
                    # 上半身 crop：合并主/兜底 ROI（y 0.15-0.85）
                    res = clf.classify(info.frame, o)
                    rx1, ry1 = res.roi_main[0], res.roi_main[1]
                    rx2, ry2 = res.roi_lower[2], res.roi_lower[3]
                    if rx2 <= rx1 or ry2 <= ry1:
                        continue
                    crop = info.frame[ry1:ry2, rx1:rx2]
                    if crop.size == 0:
                        continue
                    name = f"{v.stem}_{info.frame_id:06d}_p{o.track_id}.jpg"
                    cv2.imwrite(str(pending / name), crop)
                    meta_rows.append(
                        (name, v.stem, info.frame_id, o.track_id,
                         res.red_ratio_connected, res.baseline_ratio,
                         1 if res.is_uniform else 0)
                    )
                    count_in_video += 1
                    total += 1
        print(f"  {v.name}: 采 {count_in_video} 张")

    with (out_dir / "meta.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "video", "frame", "person_id",
                    "red_connected", "baseline", "hsv_weak_label"])
        w.writerows(meta_rows)
    print(f"完成：共 {total} 张 crop -> {pending}")
    print(f"元数据（含 HSV 弱标签，仅参考）-> {out_dir / 'meta.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 工装 crop 收集")
    parser.add_argument("--video-dir", default="data/preprocessed")
    parser.add_argument("--out-dir", default="data/uniform_crops")
    parser.add_argument("--per-video-frames", type=int, default=2000)
    parser.add_argument("--sample-every", type=int, default=30)
    args = parser.parse_args()
    run(
        args.video_dir,
        args.out_dir,
        per_video_frames=args.per_video_frames,
        sample_every=args.sample_every,
    )


if __name__ == "__main__":
    main()
