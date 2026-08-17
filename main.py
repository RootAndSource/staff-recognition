"""任务 A 员工检测 - 命令行入口。

用法示例：
    python main.py run path/to/video.mp4 --config config.yaml --json outputs/run/out.jsonl
    python main.py run rtsp://user:pass@host:554/stream --show
    python main.py calibrate path/to/frame.jpg
    python main.py evaluate --frames 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.pipeline import DetectionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="任务 A 员工检测（人员检测 + 工装识别）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（默认 config.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="运行检测管线（视频文件或 RTSP）")
    run_p.add_argument("source", help="本地视频路径或 RTSP URL")
    run_p.add_argument("--json", help="JSONL 结构化输出路径（如 outputs/run/out.jsonl）")
    run_p.add_argument("--save-video", help="输出视频路径（如 outputs/run/out.mp4）")
    run_p.add_argument("--show", action="store_true", help="显示实时叠加画面")
    run_p.add_argument("--no-show", action="store_true", help="不显示画面（如无显示器环境）")
    run_p.add_argument("--max-frames", type=int, default=0, help="最多处理 N 帧后停止（0=不限制）")
    run_p.add_argument("--profile", action="store_true", help="打印各环节耗时（定位性能瓶颈）")

    batch_p = sub.add_parser("batch", help="批量处理目录下所有视频")
    batch_p.add_argument("--input-dir", default=None, help="输入视频目录（默认取 config 的 batch.input_dir）")
    batch_p.add_argument("--json-dir", default=None, help="JSON 输出目录（默认 outputs/json）")
    batch_p.add_argument("--mp4-dir", default=None, help="视频输出目录（默认 outputs/mp4）")
    batch_p.add_argument("--max-frames", type=int, default=0, help="每个视频最多处理 N 帧（0=不限制）")

    cal_p = sub.add_parser("calibrate", help="交互标定 HSV / ROI 参数")
    cal_p.add_argument("image", help="现场截图路径")

    ev_p = sub.add_parser("evaluate", help="抽帧 + 人工标注 + 指标计算")
    ev_p.add_argument("--video", default=None, help="评估用视频路径")
    ev_p.add_argument("--frames", type=int, default=100, help="抽帧数量（默认 100）")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "run":
        pipeline = DetectionPipeline(cfg)
        pipeline.max_frames = args.max_frames
        show = args.show or cfg["output"].get("show", True)
        if args.no_show:
            show = False
        cfg["output"]["show"] = show
        if args.profile:
            cfg["output"]["profile"] = True
        json_path = Path(args.json) if args.json else None
        video_path = Path(args.save_video) if args.save_video else None
        stats = pipeline.run(args.source, json_path=json_path, save_video=video_path)
        print(
            f"[完成] 处理 {stats.frames_processed} 帧；"
            f"累计人次 {stats.persons_total}；工装人次 {stats.uniform_count}"
        )
    elif args.command == "calibrate":
        from scripts.calibrate import run_calibrate

        run_calibrate(args.image, cfg)
    elif args.command == "batch":
        run_batch(cfg, args)
    elif args.command == "evaluate":
        from scripts.evaluate import run_evaluate

        run_evaluate(cfg, video=args.video, sample_count=args.frames)
    else:
        print(f"未知命令: {args.command}", file=sys.stderr)
        return 1
    return 0


def run_batch(cfg: dict, args: argparse.Namespace) -> None:
    """批量处理：输入目录下所有视频 → outputs/json、outputs/mp4 分类输出。"""
    import re

    bcfg = cfg.get("batch", {})
    input_dir = Path(args.input_dir or bcfg.get("input_dir", "data/preprocessed"))
    json_dir = Path(args.json_dir or bcfg.get("json_dir", "outputs/json"))
    mp4_dir = Path(args.mp4_dir or bcfg.get("mp4_dir", "outputs/mp4"))
    exts = bcfg.get("extensions", [".mp4", ".avi", ".mov"])

    if not input_dir.exists():
        print(f"[batch] 输入目录不存在: {input_dir}")
        return
    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not videos:
        print(f"[batch] 输入目录下没有视频: {input_dir}")
        return

    json_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch] 找到 {len(videos)} 个视频，开始处理...")

    for idx, video in enumerate(videos, start=1):
        stem = re.sub(r"[^\w\-]+", "_", video.stem)
        json_path = json_dir / f"{stem}.jsonl"
        mp4_path = mp4_dir / f"{stem}.mp4"
        print(f"[batch] ({idx}/{len(videos)}) {video.name} -> {mp4_path.name}")
        pipeline = DetectionPipeline(cfg)
        pipeline.max_frames = args.max_frames
        cfg["output"]["show"] = bool(cfg["output"].get("show", True))
        stats = pipeline.run(
            str(video),
            json_path=json_path,
            save_video=str(mp4_path),
        )
        print(
            f"[batch] 完成 {video.name}：{stats.frames_processed} 帧，"
            f"人次 {stats.persons_total}，工装人次 {stats.uniform_count}"
        )
    print(f"[batch] 全部完成，共 {len(videos)} 个视频。")


if __name__ == "__main__":
    raise SystemExit(main())
