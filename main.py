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
    elif args.command == "evaluate":
        from scripts.evaluate import run_evaluate

        run_evaluate(cfg, video=args.video, sample_count=args.frames)
    else:
        print(f"未知命令: {args.command}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
