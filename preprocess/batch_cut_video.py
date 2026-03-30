"""
批量提取视频帧脚本

输出目录结构：
  output/
  ├── by_video/          # 同一视频的帧放在同一子文件夹（按帧序号 1,2,3... 命名）
  │   ├── 001/
  │   │   ├── 1.jpg
  │   │   ├── 2.jpg
  │   │   └── ...
  │   └── ...
  └── by_frame/          # 同一帧序号的图片放在同一子文件夹（按视频序号 1,2,3... 命名）
      ├── 1/
      │   ├── 1.jpg
      │   ├── 2.jpg
      │   └── ...
      └── ...

超参数：
  --data_dir    : 视频所在目录，默认 ../data
  --output_dir  : 输出目录，默认 ../output
  --num_frames  : 每个视频提取多少帧，默认 0（0 表示提取全部帧）
  --ext         : 视频文件扩展名过滤，默认 .mp4
"""

import os
import argparse
import cv2
import numpy as np

# 抑制 GStreamer / OpenCV 无关警告
os.environ["GST_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


def get_sample_indices(total_frames: int, num_frames: int, sample_mode: str):
    """
    返回要提取的帧索引集合（0-based）。
    num_frames == 0 时返回 None（表示提取全部，无需查表）；
    sample_mode=uniform 时均匀采样 num_frames 帧；
    sample_mode=head 时严格提取前 num_frames 帧。
    """
    if num_frames == 0 or num_frames >= total_frames:
        return None  # None 表示全部
    if sample_mode == "head":
        return set(range(num_frames))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return set(indices.tolist())


def open_video(video_path: str):
    """尝试用 FFMPEG 后端打开，失败则用默认后端。"""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
        return cap
    cap.release()
    return cv2.VideoCapture(video_path)


def process_video(video_path: str, video_idx: int, video_name: str,
                  bv_subdir: str, by_frame_dir: str,
                  num_frames: int, sample_mode: str, total_videos: int):
    """
    顺序读取视频，遇到需要的帧立即写盘（不缓存到内存）。
    sample_set 在拿到真实 total 后再计算，避免 probe 失败导致误判。
    """
    cap = open_video(video_path)
    if not cap.isOpened():
        print(f"  [跳过] 无法打开：{os.path.basename(video_path)}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 用真实 total 计算采样集合
    sample_set = get_sample_indices(total, num_frames, sample_mode)
    actual = total if sample_set is None else len(sample_set)
    print(f"[{video_idx:03d}/{total_videos}] {os.path.basename(video_path)}"
          f"  总帧数={total}  提取={actual}")

    frame_seq = 0   # 本视频已写出的帧计数（1-based 序号）
    raw_idx = 0     # 当前读到第几帧（0-based）
    report_every = max(1, actual // 20)  # 每完成 5% 打印一次

    while True:
        # head 模式：已写够直接停止，无需读完整个视频
        if sample_set is not None and frame_seq >= len(sample_set):
            break

        ret, frame = cap.read()
        if not ret:
            break

        if sample_set is None or raw_idx in sample_set:
            frame_seq += 1

            # by_video：视频名子目录，帧序号命名
            cv2.imwrite(os.path.join(bv_subdir, f"{frame_seq}.jpg"), frame)

            # by_frame：帧序号子目录，视频序号命名
            bf_subdir = os.path.join(by_frame_dir, str(frame_seq))
            os.makedirs(bf_subdir, exist_ok=True)
            cv2.imwrite(os.path.join(bf_subdir, f"{video_idx}.jpg"), frame)

            if frame_seq % report_every == 0 or frame_seq == actual:
                print(f"  进度：{frame_seq}/{actual} 帧", end="\r", flush=True)

        raw_idx += 1

    cap.release()
    print(f"  完成：共写出 {frame_seq} 帧{' ' * 20}")
    return frame_seq


def main():
    parser = argparse.ArgumentParser(description="批量提取视频帧")
    parser.add_argument("--data_dir",   type=str, default="video",   help="视频所在目录")
    parser.add_argument("--output_dir", type=str, default="output", help="输出根目录")
    parser.add_argument("--num_frames", type=int, default=0,        help="每视频提取帧数（0=全部）")
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "head"],
        help="采样方式：uniform=均匀采样，head=严格前N帧",
    )
    parser.add_argument("--ext",        type=str, default=".mp4",   help="视频文件扩展名")
    args = parser.parse_args()

    data_dir   = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    video_files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() == args.ext.lower()
    )
    if not video_files:
        print(f"在 {data_dir} 中未找到 {args.ext} 视频文件")
        return

    by_video_dir = os.path.join(output_dir, "by_video")
    by_frame_dir = os.path.join(output_dir, "by_frame")
    os.makedirs(by_video_dir, exist_ok=True)
    os.makedirs(by_frame_dir, exist_ok=True)

    print(f"共找到 {len(video_files)} 个视频，开始提取帧...\n")

    for video_idx, filename in enumerate(video_files, start=1):
        video_path = os.path.join(data_dir, filename)
        video_name = os.path.splitext(filename)[0]

        bv_subdir = os.path.join(by_video_dir, video_name)
        os.makedirs(bv_subdir, exist_ok=True)

        process_video(video_path, video_idx, video_name,
                      bv_subdir, by_frame_dir,
                      args.num_frames, args.sample_mode, len(video_files))

    print("\n全部完成！")
    print(f"  按视频组织：{by_video_dir}")
    print(f"  按帧序号组织：{by_frame_dir}")


if __name__ == "__main__":
    main()
