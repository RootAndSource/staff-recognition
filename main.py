"""StaffRecognition - 命令行入口。

用法示例：
    python main.py run path/to/video.mp4 --config config.yaml --json outputs/json/out.jsonl
    python main.py run rtsp://user:pass@host:554/stream --show
    python main.py evaluate --frames 100
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import threading
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.pipeline import DetectionPipeline

_WORKER_POSITION = 0  # 每个 worker 进程的进度条行号（批量多进程时分配）


def _init_worker(counter: "mp.Value", lock: "mp.Lock", write_lock: "mp.RLock") -> None:
    """每个 worker 进程启动时调用：分配唯一进度条行号，并共享终端写锁。"""
    global _WORKER_POSITION
    with lock:
        _WORKER_POSITION = counter.value
        counter.value += 1
    tqdm.set_lock(write_lock)


def _log_printer(
    log_queue: "mp.Queue",
    stop_event: "mp.Event",
    write_lock: "mp.RLock",
) -> None:
    """主进程渲染线程：从队列取日志/进度消息，统一打印日志并更新各 worker 进度条。

    所有 tqdm 实例都创建在主进程，position=worker_id，跨进程不再有行号竞争。
    """
    tqdm.set_lock(write_lock)
    pbars: dict[int, tqdm] = {}
    while True:
        try:
            msg = log_queue.get(timeout=0.2)
        except queue.Empty:
            if stop_event.is_set() and log_queue.empty():
                break
            continue
        kind = msg[0]
        if kind == "log":
            tqdm.write(msg[1])
        elif kind == "progress":
            _, pos, name, current, total = msg
            pbar = pbars.get(pos)
            if pbar is None or pbar.desc != name:
                if pbar is not None:
                    pbar.close()
                pbar = tqdm(
                    total=total,
                    desc=name,
                    unit="f",
                    unit_scale=False,
                    dynamic_ncols=True,
                    position=pos,
                    leave=False,
                )
                pbars[pos] = pbar
            pbar.n = current
            pbar.refresh()
        elif kind == "finish":
            pbar = pbars.pop(msg[1], None)
            if pbar is not None:
                pbar.close()
    for pbar in pbars.values():
        pbar.close()


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
    tqdm.write(f"[batch] 找到 {len(videos)} 个视频（并行进程数 {workers}），开始处理...")

    if workers <= 1:
        for idx, video in enumerate(videos, start=1):
            _batch_process_one(cfg, video, json_dir, mp4_dir, args.max_frames, idx, len(videos))
    else:
        from concurrent.futures import ProcessPoolExecutor

        counter = mp.Value("i", 0)
        lock = mp.Lock()
        write_lock = mp.RLock()
        manager = mp.Manager()
        log_queue = manager.Queue()
        stop_event = threading.Event()
        tqdm.set_lock(write_lock)
        printer = threading.Thread(target=_log_printer, args=(log_queue, stop_event, write_lock), daemon=True)
        printer.start()
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(counter, lock, write_lock),
        ) as ex:
            futures = [
                ex.submit(
                    _batch_process_one,
                    cfg,
                    video,
                    json_dir,
                    mp4_dir,
                    args.max_frames,
                    idx,
                    len(videos),
                    log_queue,
                )
                for idx, video in enumerate(videos, start=1)
            ]
            for f in futures:
                f.result()  # 异常在此抛出，保证结果按顺序汇总
        stop_event.set()
        printer.join(timeout=5)
        manager.shutdown()
    tqdm.write(f"[batch] 全部完成，共 {len(videos)} 个视频。")


def _batch_process_one(
    cfg: dict,
    video: Path,
    json_dir: Path,
    mp4_dir: Path,
    max_frames: int,
    idx: int,
    total: int,
    log_queue: "mp.Queue | None" = None,
) -> None:
    """处理单个视频（供 batch 串行/并行调用）。"""
    import re

    stem = re.sub(r"[^\w\-]+", "_", video.stem)
    json_path = json_dir / f"{stem}.jsonl"
    mp4_path = mp4_dir / f"{stem}.mp4"
    _emit_log(log_queue, f"[batch] ({idx}/{total}) {video.name} -> {mp4_path.name}")
    pipeline = DetectionPipeline(cfg)
    pipeline.max_frames = max_frames
    cfg["output"]["show"] = False  # 批处理不弹窗

    run_kwargs = {}
    if log_queue is not None:
        # 并行：进度经队列回主进程统一渲染（避免跨进程 tqdm 行竞争）
        def _progress(current: int, total_frames: int | None) -> None:
            log_queue.put(("progress", _WORKER_POSITION, video.name, current, total_frames))

        run_kwargs["progress_cb"] = _progress
    else:
        # 串行：直接用本进程 tqdm
        run_kwargs["pbar_position"] = _WORKER_POSITION
        run_kwargs["pbar_leave"] = False

    stats = pipeline.run(
        str(video),
        json_path=json_path,
        save_video=str(mp4_path),
        **run_kwargs,
    )
    if log_queue is not None:
        log_queue.put(("finish", _WORKER_POSITION))
    _emit_log(
        log_queue,
        f"[batch] 完成 {video.name}：{stats.frames_processed} 帧，"
        f"人次 {stats.persons_total}，工装人次 {stats.uniform_count}",
    )


def _emit_log(log_queue: "mp.Queue | None", msg: str) -> None:
    """批量日志：并行时经队列回主进程打印，串行时直接输出。"""
    if log_queue is not None:
        log_queue.put(("log", msg))
    else:
        tqdm.write(msg)


if __name__ == "__main__":
    raise SystemExit(main())
