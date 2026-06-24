"""
批量提取视频帧脚本（多进程加速版）

输出目录结构：
  output/
  ├── by_video/          # 同一视频的帧放在同一子文件夹（按帧序号 1,2,3... 命名）
  │   ├── video_name/
  │   │   ├── 001.png
  │   │   └── ...
  │   └── ...
  └── by_frame/          # 同一帧序号的图片放在同一子文件夹（按相机/视频序号命名）
      ├── 1/
      │   ├── 001.png
      │   └── ...
      └── ...

超参数：
  --data_dir    : 视频所在目录（或单个视频文件），默认 video
  --output_dir  : 输出目录，默认 output
  --num_frames  : 每个视频提取多少帧，默认 0（0 表示全部）
  --target_fps  : 目标帧率，如视频 120fps 设为 30 则每秒取 30 帧（默认 0 = 不限）
  --sample_mode : 采样方式：uniform / head（默认 uniform）
  --ext         : 视频文件扩展名过滤，默认 .mp4
  --output_mode : 输出组织方式：by_video / by_frame / both（默认 by_frame）
  --workers     : 并行进程数，默认 CPU 核数 / 2
  --use_seek    : 启用 seek 快速跳帧（稀疏采样时极大提速，默认关闭）
  --start_idx   : 相机/视频编号的起始值（默认 1，用于多批次续接编号）
  --img_format  : 输出图片格式：png / jpg（默认 png；jpg 写盘更快、体积更小）
  --jpg_quality : jpg 质量 1-100（默认 95，仅 --img_format jpg 生效）
  --png_compression : png 压缩级别 0-9（默认 1，越小越快、体积越大）

优先级：先按 target_fps 降采样，再按 num_frames 限制总帧数。
速度建议：稀疏采样用 --use_seek；追求最快写盘用 --img_format jpg。

示例：
    python preprocess/batch_cut_video.py --data_dir ./videos --output_dir ./frames --num_frames 100 --target_fps 30 --output_mode both --workers 4 --use_seek --img_format jpg
"""

import os
import argparse
import multiprocessing
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Iterator, Tuple

import cv2
import numpy as np

# 抑制 GStreamer / OpenCV 无关警告
os.environ["GST_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


# ─────────────────────────── 配置与任务数据结构 ───────────────────────────

@dataclass(frozen=True)
class Config:
    """所有视频共享的处理配置（一次构造，传给每个 worker）。"""
    by_video_dir: str
    by_frame_dir: str
    num_frames: int
    target_fps: float
    sample_mode: str
    output_mode: str
    use_seek: bool
    total_videos: int
    img_ext: str                       # ".png" / ".jpg"
    imwrite_params: List[int] = field(default_factory=list)

    @property
    def need_by_video(self) -> bool:
        return self.output_mode in ("by_video", "both")

    @property
    def need_by_frame(self) -> bool:
        return self.output_mode in ("by_frame", "both")


@dataclass(frozen=True)
class VideoTask:
    """单个视频的处理任务。"""
    video_path: str
    cam_idx: int        # 相机/视频编号（已含 start_idx 偏移），用于 by_frame 文件名
    bv_subdir: str      # by_video 子目录


# ─────────────────────────── 工具函数 ───────────────────────────

def get_sample_indices(total_frames: int, num_frames: int, sample_mode: str,
                       original_fps: float = 0, target_fps: float = 0) -> Optional[List[int]]:
    """
    计算要提取的帧索引（0-based 有序列表），或 None 表示全部帧。
    处理顺序：先按 target_fps 做时间降采样，再按 num_frames 限制总数。
    """
    # ── Step 1：按 target_fps 降采样 ──────────────────────────────────
    if target_fps > 0 and original_fps > 0 and target_fps < original_fps:
        step = original_fps / target_fps
        # 用整数算术生成索引并去重，避免浮点累加误差
        count = int(total_frames / step) + 1
        indices = sorted({int(round(i * step)) for i in range(count)
                          if int(round(i * step)) < total_frames})
    else:
        indices = None  # None = 全部帧

    # ── Step 2：按 num_frames 截取 ────────────────────────────────────
    if num_frames > 0:
        pool = indices if indices is not None else list(range(total_frames))
        if num_frames >= len(pool):
            return indices  # 已经足够少
        if sample_mode == "head":
            return pool[:num_frames]
        # uniform
        sel = np.linspace(0, len(pool) - 1, num_frames, dtype=int)
        return [pool[i] for i in sel]

    return indices


def open_video(video_path: str) -> cv2.VideoCapture:
    """尝试用 FFMPEG 后端打开，失败则用默认后端。"""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
        return cap
    cap.release()
    return cv2.VideoCapture(video_path)


def _probe_video(video_path: str) -> Tuple[int, float]:
    """快速探测视频的帧数和帧率（不解码）。"""
    cap = open_video(video_path)
    if not cap.isOpened():
        return 0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total, fps


def _write_safe(path: str, frame, params: List[int]) -> None:
    """写盘辅助：异常时打印警告而非崩溃整个进程。"""
    try:
        cv2.imwrite(path, frame, params)
    except Exception as exc:
        print(f"\n  [警告] 写盘失败 {path}: {exc}", flush=True)


def _decode_frames(cap: cv2.VideoCapture, indices: Optional[List[int]],
                   total: int, use_seek: bool) -> Iterator[Tuple[int, "np.ndarray"]]:
    """
    按需产出 (frame_seq, frame)，frame_seq 为所选序列中的 1-based 序号。
    - seek 模式：直接定位每个目标帧（稀疏采样最快）。
    - 顺序模式：grab() 仅 demux 跳过不需要的帧，retrieve() 只解码需要的帧。
    """
    # ── Seek 模式 ───────────────────────────────────────────────────
    if use_seek and indices is not None:
        for frame_seq, raw_idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(raw_idx))
            ret, frame = cap.read()
            if ret:
                yield frame_seq, frame
        return

    # ── 顺序模式 ───────────────────────────────────────────────────
    sample_set = set(indices) if indices is not None else None
    target_count = len(sample_set) if sample_set is not None else total
    max_raw = max(indices) if indices else (total - 1)

    raw_idx = 0
    seq = 0
    while seq < target_count:
        if not cap.grab():            # grab 仅 demux，比 read 快
            break
        if sample_set is None or raw_idx in sample_set:
            ret, frame = cap.retrieve()  # 仅对需要的帧做完整解码
            if not ret:
                break
            seq += 1
            yield seq, frame
        if raw_idx >= max_raw:        # 已越过最后一个目标帧，提前退出
            break
        raw_idx += 1


# ─────────────────────────── 单视频处理（多进程 worker） ───────────────────────────

def process_video(task: VideoTask, cfg: Config) -> int:
    """处理一个视频：解码 + 写盘。返回写出的帧数。"""
    cv2.setNumThreads(1)  # 多进程下避免 OpenCV 线程争用导致的过度调度

    try:
        cap = open_video(task.video_path)
        if not cap.isOpened():
            print(f"  [跳过] 无法打开：{os.path.basename(task.video_path)}", flush=True)
            return 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = get_sample_indices(total, cfg.num_frames, cfg.sample_mode, fps, cfg.target_fps)
        actual = total if indices is None else len(indices)

        extra = f"  原始fps={fps:.1f}  目标fps={cfg.target_fps}" if cfg.target_fps > 0 and fps > 0 else ""
        print(f"[{task.cam_idx:03d}/{cfg.total_videos}] {os.path.basename(task.video_path)}"
              f"  总帧数={total}  提取={actual}{extra}", flush=True)

        report_every = max(1, actual // 20)
        frame_count = 0

        for frame_seq, frame in _decode_frames(cap, indices, total, cfg.use_seek):
            if cfg.need_by_video:
                _write_safe(os.path.join(task.bv_subdir, f"{frame_seq:03d}{cfg.img_ext}"),
                            frame, cfg.imwrite_params)
            if cfg.need_by_frame:
                _write_safe(os.path.join(cfg.by_frame_dir, str(frame_seq),
                                         f"{task.cam_idx:03d}{cfg.img_ext}"),
                            frame, cfg.imwrite_params)
            frame_count = frame_seq
            if frame_seq % report_every == 0 or frame_seq == actual:
                print(f"  [{task.cam_idx:03d}] 进度：{frame_seq}/{actual} 帧", end="\r", flush=True)

        cap.release()
        print(f"  [{task.cam_idx:03d}] 完成：共写出 {frame_count} 帧{' ' * 20}", flush=True)
        return frame_count

    except Exception as exc:
        print(f"\n  [{task.cam_idx:03d}] 错误：{exc}", flush=True)
        import traceback
        traceback.print_exc()
        return 0


# ─────────────────────────── 主入口 ───────────────────────────

def _collect_video_files(data_dir: str, ext: str) -> Tuple[str, List[str]]:
    """返回 (实际目录, 排序后的视频文件名列表)；支持传入单个视频文件。"""
    if os.path.isfile(data_dir):
        return os.path.dirname(data_dir), [os.path.basename(data_dir)]
    files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() == ext.lower()
    )
    return data_dir, files


def main():
    cpu_count = multiprocessing.cpu_count()
    default_workers = max(1, cpu_count // 2)

    parser = argparse.ArgumentParser(description="批量提取视频帧（多进程加速版）")
    parser.add_argument("--data_dir", type=str, default="video", help="视频所在目录或单个视频文件")
    parser.add_argument("--start_idx", type=int, default=1,
                        help="相机/视频编号的起始值（默认 1，可续接之前的批次）")
    parser.add_argument("--output_dir", type=str, default="output", help="输出根目录")
    parser.add_argument("--num_frames", type=int, default=0, help="每视频提取帧数（0=全部）")
    parser.add_argument("--target_fps", type=float, default=0,
                        help="目标帧率（如原始120fps设30则每秒取30帧，0=不限）")
    parser.add_argument("--sample_mode", type=str, default="uniform", choices=["uniform", "head"],
                        help="采样方式：uniform=均匀采样，head=严格前N帧")
    parser.add_argument("--ext", type=str, default=".mp4", help="视频文件扩展名")
    parser.add_argument("--output_mode", type=str, default="by_frame",
                        choices=["by_video", "by_frame", "both"],
                        help="输出组织方式：by_video / by_frame / both（默认 by_frame）")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"并行处理视频的进程数（默认 {default_workers}）")
    parser.add_argument("--use_seek", action="store_true",
                        help="启用 seek 快速跳帧（稀疏采样时显著提速，默认关闭）")
    parser.add_argument("--img_format", type=str, default="png", choices=["png", "jpg"],
                        help="输出图片格式（jpg 写盘更快、体积更小，默认 png）")
    parser.add_argument("--jpg_quality", type=int, default=95, help="jpg 质量 1-100（默认 95）")
    parser.add_argument("--png_compression", type=int, default=1,
                        help="png 压缩级别 0-9（默认 1，越小越快、体积越大）")
    args = parser.parse_args()

    data_dir, video_files = _collect_video_files(os.path.abspath(args.data_dir), args.ext)
    if not video_files:
        print(f"在 {data_dir} 中未找到 {args.ext} 视频文件")
        return

    output_dir = os.path.abspath(args.output_dir)
    by_video_dir = os.path.join(output_dir, "by_video")
    by_frame_dir = os.path.join(output_dir, "by_frame")

    # 编码参数
    if args.img_format == "jpg":
        img_ext = ".jpg"
        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)]
    else:
        img_ext = ".png"
        imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_compression)]

    cfg = Config(
        by_video_dir=by_video_dir,
        by_frame_dir=by_frame_dir,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        sample_mode=args.sample_mode,
        output_mode=args.output_mode,
        use_seek=args.use_seek,
        total_videos=len(video_files),
        img_ext=img_ext,
        imwrite_params=imwrite_params,
    )

    # ── 预建所有子目录（主进程统一创建，worker 内不调用 makedirs，避免竞争）──────
    if cfg.need_by_video:
        for filename in video_files:
            os.makedirs(os.path.join(by_video_dir, os.path.splitext(filename)[0]), exist_ok=True)

    if cfg.need_by_frame:
        print("预扫描视频帧数以提前建立目录...")
        max_frames = 0
        for filename in video_files:
            total, fps = _probe_video(os.path.join(data_dir, filename))
            if total > 0:
                indices = get_sample_indices(total, args.num_frames, args.sample_mode,
                                             fps, args.target_fps)
                needed = total if indices is None else len(indices)
                max_frames = max(max_frames, needed)
        print(f"预建 by_frame 子目录（共 {max_frames} 个）...")
        for i in range(1, max_frames + 1):
            os.makedirs(os.path.join(by_frame_dir, str(i)), exist_ok=True)

    # ── 构造任务列表（cam_idx 含 start_idx 偏移）─────────────────────────────
    tasks = [
        VideoTask(
            video_path=os.path.join(data_dir, filename),
            cam_idx=idx + args.start_idx - 1,
            bv_subdir=os.path.join(by_video_dir, os.path.splitext(filename)[0]),
        )
        for idx, filename in enumerate(video_files, start=1)
    ]

    workers = min(args.workers, len(video_files))
    print(f"共找到 {len(video_files)} 个视频，启用 {workers} 进程并行处理...\n")

    worker = partial(process_video, cfg=cfg)
    total_written = 0
    if workers <= 1:
        for task in tasks:
            total_written += worker(task)
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            for n in pool.imap_unordered(worker, tasks):
                total_written += n

    print(f"\n全部完成！共写出 {total_written} 帧。")
    if cfg.need_by_video:
        print(f"  按视频组织：{by_video_dir}")
    if cfg.need_by_frame:
        print(f"  按帧序号组织：{by_frame_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 打包为 exe 时需要
    main()
