"""StaffRecognition - 命令行入口。

用法示例：
    python main.py run path/to/video.mp4 --config config.yaml --json outputs/json/out.jsonl
    python main.py run rtsp://user:pass@host:554/stream --show
    python main.py evaluate --frames 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.pipeline import DetectionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StaffRecognition 员工检测与工装识别")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（默认 config.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="运行检测管线（视频文件或 RTSP）")
    run_p.add_argument("source", help="本地视频路径或 RTSP URL")
    run_p.add_argument("--json", help="JSONL 结构化输出路径（如 outputs/json/out.jsonl）")
    run_p.add_argument("--save-video", help="输出视频路径（如 outputs/mp4/out.mp4）")
    run_p.add_argument("--show", action="store_true", help="显示实时叠加画面")
    run_p.add_argument("--no-show", action="store_true", help="不显示画面（如无显示器环境）")
    run_p.add_argument("--max-frames", type=int, default=0, help="最多处理 N 帧后停止（0=不限制）")
    run_p.add_argument("--profile", action="store_true", help="打印各环节耗时（定位性能瓶颈）")

    batch_p = sub.add_parser("batch", help="批量处理目录下所有视频")
    batch_p.add_argument("--input-dir", default=None, help="输入视频目录（默认取 config 的 batch.input_dir）")
    batch_p.add_argument("--json-dir", default=None, help="JSON 输出目录（默认 outputs/json）")
    batch_p.add_argument("--mp4-dir", default=None, help="视频输出目录（默认 outputs/mp4）")
    batch_p.add_argument("--max-frames", type=int, default=0, help="每个视频最多处理 N 帧（0=不限制）")
    batch_p.add_argument("--workers", type=int, default=None, help="并行进程数（默认取 config 的 batch.workers）")

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
    elif args.command == "batch":
        run_batch(cfg, args)
    elif args.command == "evaluate":
        from scripts.eval.evaluate import run_evaluate

        run_evaluate(cfg, video=args.video, sample_count=args.frames)
    else:
        print(f"未知命令: {args.command}", file=sys.stderr)
        return 1
    return 0


def run_batch(cfg: dict, args: argparse.Namespace) -> None:
    """批量处理：输入目录下所有视频 → outputs/json、outputs/mp4 分类输出。"""
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
    workers = args.workers if args.workers is not None else int(bcfg.get("workers", 1))
    print(f"[batch] 找到 {len(videos)} 个视频（并行进程数 {workers}），开始处理...")

    if workers <= 1:
        for idx, video in enumerate(videos, start=1):
            _batch_process_one(cfg, video, json_dir, mp4_dir, args.max_frames, idx, len(videos))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_batch_process_one, cfg, video, json_dir, mp4_dir, args.max_frames, idx, len(videos))
                for idx, video in enumerate(videos, start=1)
            ]
            for f in futures:
                f.result()  # 异常在此抛出，保证结果按顺序汇总
    print(f"[batch] 全部完成，共 {len(videos)} 个视频。")


def _batch_process_one(
    cfg: dict,
    video: Path,
    json_dir: Path,
    mp4_dir: Path,
    max_frames: int,
    idx: int,
    total: int,
) -> None:
    """处理单个视频（供 batch 串行/并行调用）。"""
    import re

    stem = re.sub(r"[^\w\-]+", "_", video.stem)
    json_path = json_dir / f"{stem}.jsonl"
    mp4_path = mp4_dir / f"{stem}.mp4"
    print(f"[batch] ({idx}/{total}) {video.name} -> {mp4_path.name}")
    pipeline = DetectionPipeline(cfg)
    pipeline.max_frames = max_frames
    cfg["output"]["show"] = False  # 批处理不弹窗
    stats = pipeline.run(
        str(video),
        json_path=json_path,
        save_video=str(mp4_path),
    )
    print(
        f"[batch] 完成 {video.name}：{stats.frames_processed} 帧，"
        f"人次 {stats.persons_total}，工装人次 {stats.uniform_count}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
