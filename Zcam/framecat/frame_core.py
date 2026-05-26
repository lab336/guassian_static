import os
import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    return cap


def get_target_frame(cap, frame_index, allow_missing):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total:
        if allow_missing and total > 0:
            frame_index = total - 1
        else:
            raise IndexError(f"请求帧 {frame_index} 超出范围 (总帧数 {total})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"读取帧失败: {frame_index}")
    return frame


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def horizontal_stack(frames, gap=8, bg_color=(0, 0, 0)):
    heights = [f.shape[0] for f in frames]
    h = max(heights)
    widths = [f.shape[1] for f in frames]
    total_w = sum(widths) + gap * (len(frames) - 1)
    canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
    canvas[:] = bg_color
    x = 0
    for f in frames:
        canvas[0:f.shape[0], x:x + f.shape[1]] = f
        x += f.shape[1] + gap
    return canvas


def grid_stack(frames, cols, gap=8, bg_color=(0, 0, 0)):
    if cols <= 0:
        return horizontal_stack(frames, gap, bg_color)
    rows = math.ceil(len(frames) / cols)
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    cell_w, cell_h = w, h
    grid_w = cols * cell_w + (cols - 1) * gap
    grid_h = rows * cell_h + (rows - 1) * gap
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    canvas[:] = bg_color
    for idx, f in enumerate(frames):
        r = idx // cols
        c = idx % cols
        y = r * (cell_h + gap)
        x = c * (cell_w + gap)
        canvas[y:y + f.shape[0], x:x + f.shape[1]] = f
    return canvas


def _find_chinese_font():
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/SIMLI.TTF",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def put_label(img, filename, frame_index, total_frames):
    # 将 BGR 转 Pillow RGB
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = _find_chinese_font()
    try:
        font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    margin = 6
    alpha_bg = 140  # 半透明背景

    # 文本内容
    left_text = filename
    right_text = f"第{frame_index}帧/共{total_frames}帧"

    # 计算文本尺寸
    def text_box(t):
        bbox = draw.textbbox((0, 0), t, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    lt_w, lt_h = text_box(left_text)
    rt_w, rt_h = text_box(right_text)
    W, H = pil_img.size

    # 绘制左上背景
    left_bg = (0, 0, 0, alpha_bg)
    right_bg = (0, 0, 0, alpha_bg)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([0, 0, lt_w + margin * 2, lt_h + margin * 2], fill=left_bg)
    # 右上背景区域
    rx1 = W - (rt_w + margin * 2)
    ov_draw.rectangle([rx1, 0, rx1 + rt_w + margin * 2, rt_h + margin * 2], fill=right_bg)
    # 合成半透明背景
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
    draw2 = ImageDraw.Draw(pil_img)
    draw2.text((margin, margin), left_text, font=font, fill=(255, 255, 255, 255))
    draw2.text((rx1 + margin, margin), right_text, font=font, fill=(255, 255, 255, 255))

    # 返回 BGR
    new_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img[:, :] = new_img


def extract_and_composite(videos, frame=None, time=None, height=360, cols=0, allow_missing=False):
    if frame is None and time is None:
        raise ValueError("必须提供 frame 或 time")
    frames = []
    for path in videos:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        cap = open_video(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if time is not None:
            if fps <= 0:
                cap.release()
                raise RuntimeError(f"无法获取 FPS: {path}")
            frame_index = int(round(time * fps))
        else:
            frame_index = frame
        f = get_target_frame(cap, frame_index, allow_missing)
        cap.release()
        resized = resize_to_height(f, height)
        put_label(resized, os.path.basename(path), frame_index, total_frames)
        frames.append(resized)
    composite = grid_stack(frames, cols if cols > 0 else 0)
    return composite
