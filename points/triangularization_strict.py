r"""
多相机三维点云重建 Pipeline
从多相机去畸变图像 + 已知相机内外参，使用 RoMa v2 生成三维点云。

用法:
    python main.py
    python main.py --frames 1,2,3 --roma_setting fast
    python main.py --num_samples 8000 --neighbor_range 5
可单帧也可多帧：
python .\points\triangularization_strict.py --num_samples 8000 --neighbor_range 5 --data_dir .\data2\dense\ --output_dir .\data2\dense\points

"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from romav2 import RoMaV2


# ======================== 配置 ========================

@dataclass
class Config:
    """Pipeline 可调参数。"""
    data_dir: str = "images_processed"       # 数据根目录
    output_dir: str = "output"               # 输出目录
    num_samples: int = 5000                  # 每对图像采样匹配数
    target_points: int = 50000               # 最终目标点数
    min_confidence: float = 0.1              # RoMa overlap 置信度过滤阈值
    ransac_threshold: float = 1.0            # RANSAC 阈值 (像素，严格提高可靠性)
    reproj_threshold: float = 2.0            # 逐对重投影误差阈值 (像素，严格)
    max_depth: float = 100.0                 # 三角化最大深度
    min_depth: float = 0.05                  # 三角化最小深度
    min_angle: float = 1.5                   # 最小三角化角度 (度)，过滤退化三角化
    min_observations: int = 2                # 点至少被多少个视图对独立三角化
    min_visible_views: int = 3               # 多视角可见性：至少被 N 个相机看到
    neighbor_range: int = 3                  # 每个相机与前后 N 个邻居配对
    sor_k: int = 10                          # 统计离群值去除 - 近邻数
    sor_std: float = 3.0                     # 统计离群值去除 - 标准差倍数
    roma_setting: str = "fast"               # RoMa v2 模式: turbo/fast/base/precise
    compile: bool = False                    # torch.compile (Windows 须关闭)


# ======================== COLMAP 文件解析 ========================

def parse_cameras(cameras_path: Path) -> dict:
    """解析 COLMAP cameras.txt，返回 {camera_id: {K, width, height}}。"""
    cameras = {}
    with open(cameras_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            if model == "PINHOLE":
                fx, fy, cx, cy = map(float, parts[4:8])
            else:
                raise ValueError(f"不支持的相机模型: {model}")
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            cameras[cam_id] = {"K": K, "width": width, "height": height}
    return cameras


def parse_images(images_path: Path) -> dict:
    """解析 COLMAP images.txt，返回 {image_name: {R, t, camera_id}}。

    R, t 满足: x_cam = R @ x_world + t (world-to-camera)。
    """
    images = {}
    with open(images_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        # COLMAP 四元数 -> 旋转矩阵 (scipy 用 [x,y,z,w] 格式)
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz], dtype=np.float64)
        images[name] = {"R": R, "t": t, "camera_id": camera_id}
        i += 2  # 跳过 POINTS2D 行
    return images


def scale_intrinsics(K: np.ndarray, colmap_w: int, colmap_h: int,
                     actual_w: int, actual_h: int) -> np.ndarray:
    """若实际图像分辨率与 COLMAP 不同，缩放内参矩阵。"""
    if colmap_w == actual_w and colmap_h == actual_h:
        return K
    sx = actual_w / colmap_w
    sy = actual_h / colmap_h
    K_scaled = K.copy()
    K_scaled[0, :] *= sx
    K_scaled[1, :] *= sy
    return K_scaled


# ======================== 三角化与过滤 ========================

def triangulate_points(K1, R1, t1, K2, R2, t2, pts1, pts2):
    """从两个视角三角化三维点。"""
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64),
                                  pts2.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3:]).T
    return pts3d


def compute_triangulation_angles(R1, t1, R2, t2, pts3d):
    """计算每个点的三角化角度（两相机射线夹角，度）。角度越大精度越高。"""
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    ray1 = pts3d - C1.reshape(1, 3)
    ray2 = pts3d - C2.reshape(1, 3)
    n1 = np.linalg.norm(ray1, axis=1)
    n2 = np.linalg.norm(ray2, axis=1)
    cos_a = np.sum(ray1 * ray2, axis=1) / (n1 * n2 + 1e-12)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def filter_by_depth(R1, t1, R2, t2, pts3d, min_d, max_d):
    """过滤深度不合理的点。"""
    d1 = (R1 @ pts3d.T + t1.reshape(3, 1))[2]
    d2 = (R2 @ pts3d.T + t2.reshape(3, 1))[2]
    return (d1 > min_d) & (d1 < max_d) & (d2 > min_d) & (d2 < max_d)


def filter_by_reprojection(K1, R1, t1, K2, R2, t2, pts1, pts2, pts3d, thr):
    """通过重投影误差过滤不可靠的三维点。"""
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    h = np.hstack([pts3d, np.ones((len(pts3d), 1))]).T
    proj1 = P1 @ h; proj1 = (proj1[:2] / proj1[2:]).T
    proj2 = P2 @ h; proj2 = (proj2[:2] / proj2[2:]).T
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)
    return (err1 < thr) & (err2 < thr)


# ======================== 体素合并与多视角验证 ========================

def merge_and_count_observations(points, colors):
    """体素网格合并空间相近点。

    自动根据场景包围盒计算体素大小。同一体素内的点取均值位置、
    均值颜色，并统计观测次数（被多少次独立三角化命中）。
    观测次数高的点意味着被多个相机对一致地重建出来，可靠性高。
    """
    if len(points) == 0:
        return points, colors, np.array([], dtype=int)
    bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    voxel_size = max(bbox_diag * 0.003, 1e-6)
    origin = points.min(axis=0)
    vox_idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    vox_idx -= vox_idx.min(axis=0)
    dims = vox_idx.max(axis=0) + 1
    keys = (vox_idx[:, 0] * (dims[1] * dims[2])
            + vox_idx[:, 1] * dims[2]
            + vox_idx[:, 2])
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    n_vox = len(unique_keys)
    # 向量化聚合 (np.bincount 比循环快几十倍)
    mx = np.bincount(inverse, weights=points[:, 0], minlength=n_vox) / counts
    my = np.bincount(inverse, weights=points[:, 1], minlength=n_vox) / counts
    mz = np.bincount(inverse, weights=points[:, 2], minlength=n_vox) / counts
    merged_pts = np.column_stack([mx, my, mz])
    mr = np.bincount(inverse, weights=colors[:, 0].astype(float), minlength=n_vox) / counts
    mg = np.bincount(inverse, weights=colors[:, 1].astype(float), minlength=n_vox) / counts
    mb = np.bincount(inverse, weights=colors[:, 2].astype(float), minlength=n_vox) / counts
    merged_clr = np.column_stack([mr, mg, mb]).astype(np.uint8)
    return merged_pts, merged_clr, counts


def multiview_visibility_filter(pts3d, valid_images, K_cache, images_info,
                                actual_w, actual_h, min_views):
    """保留能被至少 min_views 个相机看到（正深度 + 画面内）的点。"""
    n = len(pts3d)
    if n == 0:
        return np.ones(0, dtype=bool)
    visible = np.zeros(n, dtype=int)
    h = np.hstack([pts3d, np.ones((n, 1))]).T  # 4xN
    for name in valid_images:
        info = images_info[name]
        K = K_cache[name]
        R, t = info["R"], info["t"]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        proj = P @ h  # 3xN
        depth = proj[2]
        safe_d = np.where(depth > 0, depth, 1.0)
        u, v = proj[0] / safe_d, proj[1] / safe_d
        in_view = ((depth > 0) & (u >= 0) & (u < actual_w)
                   & (v >= 0) & (v < actual_h))
        visible += in_view.astype(int)
    return visible >= min_views


def multiview_color_blend(pts3d, valid_images, K_cache, images_info,
                          img_dir, actual_w, actual_h, img_cache=None):
    """从所有可见视角混合提取颜色（比单视角更稳健）。"""
    n = len(pts3d)
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    color_sum = np.zeros((n, 3), dtype=np.float64)
    color_cnt = np.zeros(n, dtype=int)
    h = np.hstack([pts3d, np.ones((n, 1))]).T
    for name in valid_images:
        info = images_info[name]
        K = K_cache[name]
        R, t = info["R"], info["t"]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        proj = P @ h
        depth = proj[2]
        safe_d = np.where(depth > 0, depth, 1.0)
        u = np.round(proj[0] / safe_d).astype(int)
        v = np.round(proj[1] / safe_d).astype(int)
        if img_cache is not None and name in img_cache:
            img_rgb = img_cache[name]
            h_img, w_img = img_rgb.shape[:2]
        else:
            img = cv2.imread(str(img_dir / name))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 按当前图像真实尺寸判断可见性，避免不同分辨率图像导致越界索引。
        valid = ((depth > 0) & (u >= 0) & (u < w_img)
                 & (v >= 0) & (v < h_img))
        if not np.any(valid):
            continue
        idx = np.where(valid)[0]
        color_sum[idx] += img_rgb[v[idx], u[idx]].astype(np.float64)
        color_cnt[idx] += 1
    color_cnt = np.maximum(color_cnt, 1)
    return (color_sum / color_cnt[:, None]).astype(np.uint8)


# ======================== 点云去噪 ========================

def statistical_outlier_removal(points, colors, normals=None, k=20, std_mul=2.0):
    """统计离群值去除：剔除到 k 近邻平均距离异常大的点。"""
    if len(points) < k + 1:
        return (points, colors, normals) if normals is not None else (points, colors)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_mul * mean_d.std()
    mask = mean_d < thr
    if normals is not None:
        return points[mask], colors[mask], normals[mask]
    return points[mask], colors[mask]


# ======================== 密度感知降采样 ========================

def density_aware_downsample(points, colors, target_n, normals=None, density_k=16):
    """密度感知降采样：密集区域多删点，稀疏区域（如花瓣）几乎不删。

    策略：
    1. 用 KDTree 计算每个点的局部密度（k 近邻平均距离的倒数）
    2. 采样概率 ∝ 1/density（稀疏点概率高，密集点概率低）
    3. 按概率采样 target_n 个点
    """
    if len(points) <= target_n:
        return (points, colors, normals) if normals is not None else (points, colors)

    n = len(points)
    # 计算局部密度：k 近邻平均距离越小 → 密度越高
    tree = cKDTree(points)
    k = min(density_k, n - 1)
    dists, _ = tree.query(points, k=k + 1)  # 第 0 列是自身(距离=0)
    mean_dist = dists[:, 1:].mean(axis=1)    # 每个点到 k 近邻的平均距离

    # 避免除零
    mean_dist = np.maximum(mean_dist, 1e-10)

    # 采样权重 = 平均距离（稀疏的点距离大 → 权重大 → 更容易被保留）
    weights = mean_dist.copy()

    # 对最稀疏的 10% 点额外加权，确保花瓣等漂浮物被保留
    sparse_threshold = np.percentile(weights, 90)
    weights[weights >= sparse_threshold] *= 3.0

    # 归一化为概率
    prob = weights / weights.sum()

    rng = np.random.default_rng(42)
    chosen = rng.choice(n, size=target_n, replace=False, p=prob)
    if normals is not None:
        return points[chosen], colors[chosen], normals[chosen]
    return points[chosen], colors[chosen]


# ======================== 法向量估计 ========================

def estimate_normals(points, k=20):
    """PCA 局部平面拟合估计法向量。

    对每个点取其 k 近邻，用协方差矩阵的最小特征值对应的特征向量
    作为法向量（即局部平面法线）。

    Args:
        points: (N, 3) 点云坐标
        k: 近邻数，默认 20

    Returns:
        normals: (N, 3) 单位法向量（方向未统一）
    """
    n = len(points)
    if n < k + 1:
        k = max(n - 1, 3)
    if n < 3:
        return np.ones_like(points)

    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)  # 第 0 列是自身
    neighbors = points[idx[:, 1:]]        # (N, k, 3)，去掉自身

    # 每个点对其近邻做 PCA：协方差矩阵最小特征值 → 法向量
    centers = neighbors.mean(axis=1)      # (N, 3)
    centered = neighbors - centers[:, None, :]  # (N, k, 3)

    # 向量化协方差计算：对每个点求 3x3 协方差
    # cov[i] = centered[i].T @ centered[i] / k
    cov = np.einsum("nki,nkj->nij", centered, centered) / (k - 1)  # (N, 3, 3)

    # 批量特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # (N, 3), (N, 3, 3)

    # 最小特征值对应的特征向量（第一列，eigh 返回升序）
    normals = eigenvectors[:, :, 0]  # (N, 3)

    # 归一化（数值安全）
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)

    return normals


def orient_normals_towards_cameras(normals, points, valid_images,
                                    images_info, K_cache=None):
    """将法向量统一朝向相机方向。

    对每个点，计算其到所有相机位置的视线方向，取平均视线方向。
    若法向量与平均视线方向反向（点积 < 0），则翻转法向量。
    这样所有法向量都指向"外侧"（面向相机）。

    Args:
        normals: (N, 3) 未统一朝向的单位法向量
        points: (N, 3) 点云坐标
        valid_images: 有效的图像名列表
        images_info: {name: {R, t, camera_id}} 相机位姿

    Returns:
        normals: (N, 3) 统一朝向后（面向相机）的单位法向量
    """
    n = len(points)
    if n == 0:
        return normals

    # 计算所有相机中心
    cameras_C = []
    for name in valid_images:
        info = images_info[name]
        R, t = info["R"], info["t"]
        C = -R.T @ t  # 相机中心在世界坐标系
        cameras_C.append(C)
    cameras_C = np.array(cameras_C)  # (M, 3)

    # 对每个点，计算到所有相机的平均视线方向
    # 用 KDTree 找最近相机（近似），足够判断朝向
    cam_tree = cKDTree(cameras_C)
    _, nearest_cam_idx = cam_tree.query(points, k=1)
    view_dirs = cameras_C[nearest_cam_idx] - points  # (N, 3)
    view_dirs = view_dirs / np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-12)

    # 若法向量与视线方向反向（点积 < 0），翻转
    dot = np.sum(normals * view_dirs, axis=1)
    flip = dot < 0
    normals[flip] = -normals[flip]

    return normals


# ======================== 颜色提取（单视角，用于中间阶段） ========================

def extract_colors(img_path, K, R, t, pts3d, img_cache=None):
    """将三维点投影到图像上获取 RGB 颜色。"""
    name = Path(img_path).name
    if img_cache is not None and name in img_cache:
        img_rgb = img_cache[name]
        H, W = img_rgb.shape[:2]
    else:
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((len(pts3d), 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
    P = K @ np.hstack([R, t.reshape(3, 1)])
    h = np.hstack([pts3d, np.ones((len(pts3d), 1))]).T
    proj = P @ h
    uv = (proj[:2] / proj[2:]).T
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    colors = np.zeros((len(pts3d), 3), dtype=np.uint8)
    colors[valid] = img_rgb[v[valid], u[valid]]
    return colors


# ======================== PLY 保存 ========================

def save_ply(filepath, points, colors=None, normals=None):
    """保存点云为 .ply 文件（带 RGB 颜色和法向量）。"""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    has_normals = normals is not None and len(normals) == n
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
    )
    if has_normals:
        header += "property float nx\nproperty float ny\nproperty float nz\n"
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(np.array(points[i], dtype=np.float32).tobytes())
            if has_normals:
                f.write(np.array(normals[i], dtype=np.float32).tobytes())
            if has_color:
                f.write(np.array(colors[i], dtype=np.uint8).tobytes())


# ======================== 配对策略 ========================

def build_pairs(n_images, neighbor_range):
    """生成相邻相机配对列表（不循环）。"""
    pairs = []
    for i in range(n_images):
        for j in range(i + 1, min(i + 1 + neighbor_range, n_images)):
            pairs.append((i, j))
    return pairs


# ======================== 并行工具 ========================

def get_n_workers() -> int:
    """返回 80% 的 CPU 核数（至少 1）。"""
    return max(1, int(os.cpu_count() * 0.8))


def preload_images(img_dir: Path, image_names: list, n_workers: int) -> dict:
    """并行将图像预加载为 RGB numpy 数组，返回 {name: img_rgb}。"""
    def _load(name):
        img = cv2.imread(str(img_dir / name))
        if img is None:
            return name, None
        return name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_load, image_names))
    return {name: img for name, img in results if img is not None}


# ======================== 单帧处理（拆分为 GPU 阶段 + CPU 阶段，供流水线使用） ========================

def _gpu_match(frame_id, frame_dir, cameras, images_info, model, cfg):
    """Phase 1-GPU：图像预加载 + RoMa 特征匹配。
    返回供 CPU 阶段使用的中间状态字典，失败时返回 None。
    """
    img_dir = frame_dir / "images"
    if not img_dir.exists():
        img_dir = frame_dir

    available = sorted(
        [f.name for f in img_dir.iterdir()
         if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
        key=lambda x: (0, int(Path(x).stem)) if Path(x).stem.isdigit() else (1, Path(x).stem),
    )
    # 若实际文件名与 COLMAP images_info 键不匹配（如 0.png vs cam003frame001.png），
    # 按相机 ID 排序后做位置映射
    if available and not any(n in images_info for n in available):
        sorted_db = sorted(images_info.keys(),
                           key=lambda k: images_info[k]["camera_id"])
        if len(available) == len(sorted_db):
            images_info = {a: images_info[d] for a, d in zip(available, sorted_db)}
            tqdm.write(f"  [帧 {frame_id}] 文件名不匹配，按相机ID位置映射 {len(available)} 张")
        else:
            tqdm.write(f"  [帧 {frame_id}] 图像数({len(available)})与"
                       f"COLMAP记录({len(sorted_db)})不匹配，跳过")
            return None
    valid_images = [n for n in available if n in images_info]
    if len(valid_images) < 2:
        tqdm.write(f"  [帧 {frame_id}] 有效图像不足 (< 2)，跳过")
        return None

    n_workers = get_n_workers()
    tqdm.write(f"  [帧 {frame_id}] 预加载 {len(valid_images)} 张图像 (workers={n_workers})...")
    img_cache = preload_images(img_dir, valid_images, n_workers)
    if not img_cache:
        tqdm.write(f"  [帧 {frame_id}] 图像加载失败，跳过")
        return None

    first_img = img_cache[valid_images[0]]
    actual_h, actual_w = first_img.shape[:2]

    K_cache = {}
    for name in valid_images:
        info = images_info[name]
        cam = cameras[info["camera_id"]]
        K_cache[name] = scale_intrinsics(cam["K"], cam["width"], cam["height"],
                                         actual_w, actual_h)

    pairs = build_pairs(len(valid_images), cfg.neighbor_range)

    match_data = []
    gpu_bar = tqdm(pairs, desc=f"  帧 {frame_id} Phase1-GPU", leave=False)
    for idx_i, idx_j in gpu_bar:
        ni, nj = valid_images[idx_i], valid_images[idx_j]
        path_i, path_j = str(img_dir / ni), str(img_dir / nj)
        try:
            preds = model.match(path_i, path_j)
        except Exception as e:
            tqdm.write(f"    匹配失败 ({ni}<->{nj}): {e}")
            continue
        matches, overlaps, _, _ = model.sample(preds, cfg.num_samples)
        kpA, kpB = model.to_pixel_coordinates(matches, actual_h, actual_w,
                                               actual_h, actual_w)
        pts_i = kpA.cpu().numpy()
        pts_j = kpB.cpu().numpy()
        conf = overlaps.cpu().numpy()
        cmask = conf > cfg.min_confidence
        pts_i, pts_j = pts_i[cmask], pts_j[cmask]
        if len(pts_i) >= 8:
            match_data.append((ni, nj, pts_i, pts_j))

    return dict(
        frame_id=frame_id,
        img_dir=img_dir,
        valid_images=valid_images,
        K_cache=K_cache,
        match_data=match_data,
        img_cache=img_cache,
        actual_w=actual_w,
        actual_h=actual_h,
        images_info=images_info,
    )


def _cpu_phases(state, cfg):
    """Phase 1-CPU ~ Phase 5：三角化、合并、可见性、颜色混合、法向量估计。
    接收 _gpu_match() 返回的状态字典，返回 (points, colors, normals)。
    """
    if state is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3))

    frame_id     = state["frame_id"]
    img_dir      = state["img_dir"]
    valid_images = state["valid_images"]
    K_cache      = state["K_cache"]
    match_data   = state["match_data"]
    img_cache    = state["img_cache"]
    actual_w     = state["actual_w"]
    actual_h     = state["actual_h"]
    images_info  = state["images_info"]

    if not match_data:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3))

    # ==================== Phase 1-CPU: 并行三角化与过滤 ====================
    def _triangulate_pair(args):
        ni, nj, pts_i, pts_j = args
        Ki, Kj = K_cache[ni], K_cache[nj]
        Ri, ti = images_info[ni]["R"], images_info[ni]["t"]
        Rj, tj = images_info[nj]["R"], images_info[nj]["t"]
        n_match = len(pts_i)

        E, emask = cv2.findEssentialMat(
            pts_i, pts_j, Ki,
            method=cv2.USAC_MAGSAC, prob=0.999999,
            threshold=cfg.ransac_threshold, maxIters=10000,
        )
        if emask is None:
            return None
        emask = emask.ravel().astype(bool)
        n_inlier = int(emask.sum())
        if n_inlier < 10:
            return None
        in_i, in_j = pts_i[emask], pts_j[emask]

        pts3d = triangulate_points(Ki, Ri, ti, Kj, Rj, tj, in_i, in_j)

        angles = compute_triangulation_angles(Ri, ti, Rj, tj, pts3d)
        amask = angles >= cfg.min_angle
        n_angle = int(amask.sum())
        pts3d, in_i, in_j = pts3d[amask], in_i[amask], in_j[amask]

        dmask = filter_by_depth(Ri, ti, Rj, tj, pts3d, cfg.min_depth, cfg.max_depth)
        pts3d, in_i, in_j = pts3d[dmask], in_i[dmask], in_j[dmask]
        if len(pts3d) == 0:
            return None

        rmask = filter_by_reprojection(Ki, Ri, ti, Kj, Rj, tj,
                                       in_i, in_j, pts3d, cfg.reproj_threshold)
        pts3d, in_i = pts3d[rmask], in_i[rmask]
        if len(pts3d) == 0:
            return None

        clr = extract_colors(str(img_dir / ni), Ki, Ri, ti, pts3d, img_cache)
        return pts3d, clr, n_match, n_inlier, n_angle

    n_workers = get_n_workers()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pair_results = list(tqdm(
            pool.map(_triangulate_pair, match_data),
            total=len(match_data),
            desc=f"  帧 {frame_id} Phase1-CPU",
            leave=False,
        ))

    raw_pts, raw_clr = [], []
    total_raw = 0
    for (ni, nj, *_), result in zip(match_data, pair_results):
        if result is None:
            continue
        pts3d, clr, n_match, n_inlier, n_angle = result
        raw_pts.append(pts3d)
        raw_clr.append(clr)
        total_raw += len(pts3d)
        tqdm.write(f"    [{ni}<->{nj}] 匹配:{n_match} 内点:{n_inlier} "
                   f"角度ok:{n_angle} 三角化:{len(pts3d)}")

    if not raw_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3))

    all_pts = np.vstack(raw_pts)
    all_clr = np.vstack(raw_clr)
    tqdm.write(f"  [帧 {frame_id}] Phase1: {total_raw} 个原始点")

    # ==================== Phase 2 ====================
    merged_pts, merged_clr, obs = merge_and_count_observations(all_pts, all_clr)
    obs_mask = obs >= cfg.min_observations
    merged_pts, merged_clr = merged_pts[obs_mask], merged_clr[obs_mask]
    tqdm.write(f"  [帧 {frame_id}] Phase2: 合并 {total_raw} -> {len(merged_pts)} "
               f"(obs >= {cfg.min_observations})")

    if len(merged_pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3))

    # ==================== Phase 3 ====================
    vis_mask = multiview_visibility_filter(
        merged_pts, valid_images, K_cache, images_info,
        actual_w, actual_h, cfg.min_visible_views)
    merged_pts, merged_clr = merged_pts[vis_mask], merged_clr[vis_mask]
    tqdm.write(f"  [帧 {frame_id}] Phase3: 可见性 -> {len(merged_pts)} "
               f"(>= {cfg.min_visible_views} views)")

    if len(merged_pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3))

    # ==================== Phase 4: 多视角颜色混合 ====================
    tqdm.write(f"  [帧 {frame_id}] Phase4: 多视角颜色混合...")
    colors = multiview_color_blend(
        merged_pts, valid_images, K_cache, images_info,
        img_dir, actual_w, actual_h, img_cache)

    # ==================== Phase 5: 法向量估计 ====================
    tqdm.write(f"  [帧 {frame_id}] Phase5: 估计法向量 (k=20)...")
    normals = estimate_normals(merged_pts, k=20)
    normals = orient_normals_towards_cameras(
        normals, merged_pts, valid_images, images_info)

    return merged_pts, colors, normals


def process_frame(frame_id, frame_dir, cameras, images_info, model, cfg):
    """顺序执行版（向后兼容）。流水线模式请直接调用 _gpu_match + _cpu_phases。"""
    state = _gpu_match(frame_id, frame_dir, cameras, images_info, model, cfg)
    return _cpu_phases(state, cfg)  # (points, colors, normals)


# ======================== 主函数 ========================

def main():
    parser = argparse.ArgumentParser(
        description="多相机三维点云重建 (RoMa v2)")
    parser.add_argument("--data_dir", default="F:/by_frame",
                        help="数据根目录")
    parser.add_argument("--output_dir", default="output",
                        help="输出目录")
    parser.add_argument("--frames", default=None,
                        help="逗号分隔的帧号，如 1,2,3 (默认全部)")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="每对图像采样匹配数")
    parser.add_argument("--target_points", type=int, default=50000,
                        help="每帧最终目标点数")
    parser.add_argument("--neighbor_range", type=int, default=3,
                        help="相邻相机配对范围")
    parser.add_argument("--roma_setting", default="turbo",
                        choices=["turbo", "fast", "base", "precise"],
                        help="RoMa v2 速度/精度模式")
    parser.add_argument("--min_confidence", type=float, default=0.1,
                        help="最小 overlap 置信度")
    parser.add_argument("--ransac_threshold", type=float, default=1.0,
                        help="RANSAC 阈值 (像素)")
    parser.add_argument("--reproj_threshold", type=float, default=2.0,
                        help="逐对重投影误差阈值 (像素)")
    parser.add_argument("--max_depth", type=float, default=100.0,
                        help="最大深度")
    parser.add_argument("--min_angle", type=float, default=1.5,
                        help="最小三角化角度 (度)")
    parser.add_argument("--min_observations", type=int, default=2,
                        help="点至少被多少对相机独立三角化")
    parser.add_argument("--min_visible_views", type=int, default=3,
                        help="点至少被多少个相机看到")
    parser.add_argument("--sor_k", type=int, default=10,
                        help="SOR 近邻数")
    parser.add_argument("--sor_std", type=float, default=3.0,
                        help="SOR 标准差倍数")
    parser.add_argument("--sparse_dir", default=None,
                        help="COLMAP sparse/0 目录 (默认: data_dir/sparse/0)")
    parser.add_argument("--compile", action="store_true",
                        help="启用 torch.compile (需要 Triton)")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_points=args.target_points,
        min_confidence=args.min_confidence,
        ransac_threshold=args.ransac_threshold,
        reproj_threshold=args.reproj_threshold,
        max_depth=args.max_depth,
        min_angle=args.min_angle,
        min_observations=args.min_observations,
        min_visible_views=args.min_visible_views,
        neighbor_range=args.neighbor_range,
        sor_k=args.sor_k,
        sor_std=args.sor_std,
        roma_setting=args.roma_setting,
        compile=args.compile,
    )

    data_path = Path(cfg.data_dir)
    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. 解析 COLMAP 相机参数 ----
    if args.sparse_dir:
        sparse_dir = Path(args.sparse_dir)
    else:
        # 优先 sparse/（undistort_from_calib 输出），再尝试 sparse/0/（COLMAP 标准）
        for _cand in [data_path / "sparse", data_path / "sparse" / "0"]:
            if (_cand / "cameras.txt").exists():
                sparse_dir = _cand
                break
        else:
            sparse_dir = data_path / "sparse" / "0"  # fallback
    print(f"[1/4] 解析相机参数: {sparse_dir}")
    cameras = parse_cameras(sparse_dir / "cameras.txt")
    images_info = parse_images(sparse_dir / "images.txt")
    print(f"      相机数: {len(cameras)}, COLMAP 图像数: {len(images_info)}")

    # ---- 2. 加载 RoMa v2 ----
    print(f"[2/4] 加载 RoMa v2  (setting={cfg.roma_setting}, compile={cfg.compile})")
    model = RoMaV2(RoMaV2.Cfg(compile=cfg.compile))
    model.apply_setting(cfg.roma_setting)
    dev = next(model.parameters()).device
    print(f"      模型就绪，设备: {dev}")

    # ---- 3. 确定帧列表 ----
    # 支持三种布局：
    # A) data_dir/images/{frame_id}/  (undistort_from_calib 批量输出)
    # B) data_dir/{frame_id}/images/  (旧版多帧)
    # C) data_dir/images/             (单帧)
    images_subdir = data_path / "images"
    if images_subdir.exists():
        _sub_dirs = [d for d in images_subdir.iterdir() if d.is_dir()]
        if _sub_dirs:
            # Layout A：images/ 下有子目录 → 多帧
            frame_base = images_subdir
            all_frame_ids = sorted(
                [d.name for d in _sub_dirs],
                key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
            )
        else:
            # Layout C：images/ 下直接是图像 → 单帧
            frame_base = data_path
            all_frame_ids = ["."]
    else:
        # Layout B：data_dir/ 下有编号子目录
        frame_base = data_path
        all_frame_ids = sorted(
            [d.name for d in data_path.iterdir()
             if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x),
        )

    if args.frames:
        frame_ids = [s.strip() for s in args.frames.split(",")]
    else:
        frame_ids = all_frame_ids
    print(f"[3/4] 待处理帧: {len(frame_ids)} 帧  (图像根目录: {frame_base})")
    print(f"      参数: samples={cfg.num_samples}, target={cfg.target_points}, "
          f"neighbors={cfg.neighbor_range}, "
          f"ransac_thr={cfg.ransac_threshold}, reproj_thr={cfg.reproj_threshold}")

    # ---- 4. 流水线处理：GPU 匹配与 CPU 后处理并行 ----
    # 主线程：逐帧做 GPU 匹配 (_gpu_match)
    # 后台线程：消费匹配结果，执行三角化/过滤/合并/颜色 + 去噪 + 保存
    # CUDA 操作会释放 GIL，使后台线程真正与 GPU 并行运行。
    import queue as _queue
    import threading as _threading

    _work_q: _queue.Queue = _queue.Queue(maxsize=2)   # 背压：防止 GPU 超前太多帧

    def _cpu_worker():
        while True:
            item = _work_q.get()
            if item is None:          # sentinel
                break
            out_idx, frame_tag, state = item
            try:
                points, colors, normals = _cpu_phases(state, cfg)
                if len(points) == 0:
                    tqdm.write(f"  [帧 {frame_tag}] 无有效三维点")
                    continue
                before = len(points)
                points, colors, normals = statistical_outlier_removal(
                    points, colors, normals, k=cfg.sor_k, std_mul=cfg.sor_std)
                after_sor = len(points)
                points, colors, normals = density_aware_downsample(
                    points, colors, cfg.target_points, normals)
                ply_path = out_path / f"{out_idx}.ply"
                save_ply(ply_path, points, colors, normals)
                tqdm.write(f"  [帧 {frame_tag}] 点数: {before} -> {after_sor}(去噪) "
                           f"-> {len(points)}(降采样)  => {ply_path}")
            except Exception as exc:
                tqdm.write(f"  [帧 {frame_tag}] CPU 处理异常: {exc}")

    _cpu_thread = _threading.Thread(target=_cpu_worker, daemon=True)
    _cpu_thread.start()

    print("[4/4] 开始重建 (GPU/CPU 流水线) ...")
    for out_idx, fid in enumerate(tqdm(frame_ids, desc="总进度"), start=1):
        fdir = frame_base if fid == "." else (frame_base / fid)
        frame_tag = "root" if fid == "." else fid
        has_images_subdir = (fdir / "images").exists()
        has_direct_images = fdir.exists() and (
            any(fdir.glob("*.jpg")) or any(fdir.glob("*.png")) or any(fdir.glob("*.jpeg"))
        )
        if not (has_images_subdir or has_direct_images):
            tqdm.write(f"  [帧 {frame_tag}] 目录不存在或无图像，跳过")
            continue

        # images.txt 中的键可能带帧前缀（如 "1/cam1.png"），转为局部文件名
        prefix = fid + "/"
        if fid != "." and any(k.startswith(prefix) for k in images_info):
            frame_images_info = {k[len(prefix):]: v
                                 for k, v in images_info.items()
                                 if k.startswith(prefix)}
        else:
            frame_images_info = images_info

        # GPU 阶段在主线程执行（CUDA 操作期间 GIL 释放，CPU 线程可并行运行）
        state = _gpu_match(frame_tag, fdir, cameras, frame_images_info, model, cfg)
        # 阻塞直到队列有空位（即 CPU 线程已开始处理前一帧，自然形成背压）
        _work_q.put((out_idx, frame_tag, state))

    _work_q.put(None)       # sentinel：通知 CPU 线程退出
    _cpu_thread.join()      # 等待最后一帧完成

    print("\n全部完成！输出目录:", out_path.resolve())


if __name__ == "__main__":
    main()
