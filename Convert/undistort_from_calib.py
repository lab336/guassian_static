#!/usr/bin/env python3
"""
使用pycolmap对libCalib格式的calib.json进行图像去畸变

特点:
- 使用FULL_OPENCV模型包含k3参数，避免鱼眼效果
- 自动裁剪使主点居中（3DGS训练要求）
- 统一所有图像尺寸
- 输出COLMAP格式（PINHOLE模型，无畸变）

使用方法:
python Convert/undistort_from_calib.py --calib calib.json --images_dir images --output_dir undistorted --workers 20

python Convert/undistort_from_calib.py  --calib data/li_video/calib.json  --images_dir data/li_video/images/by_frame   --output_dir data/li_video/undistorted
"""

import argparse
import json
import os
from pathlib import Path
import glob
from typing import Dict, Any, Tuple, List, Optional
import shutil
import numpy as np
import cv2
import concurrent.futures

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False
    print("警告: pycolmap未安装，请运行 pip install pycolmap")

import re

def _natural_sort_key(path_str: str):
    """自然排序key，使 2.jpg 排在 10.jpg 前面"""
    base = os.path.basename(path_str)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', base)]


# =============================================================================
# Calib.json 解析
# =============================================================================

def extract_intrinsics(cam_entry: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], Dict[str, float]]:
    """从相机条目提取内参 (fx, fy), (cx, cy), (width, height), dist_coeffs"""
    model = cam_entry.get('model', {})
    data = model.get('ptr_wrapper', {}).get('data', {})
    params = data.get('parameters', {})
    crt = data.get('CameraModelCRT', {})
    base = crt.get('CameraModelBase', {})
    img_size = base.get('imageSize', {})
    width = int(img_size.get('width', 0))
    height = int(img_size.get('height', 0))

    f = params.get('f', {}).get('val')
    ar = params.get('ar', {}).get('val', 1.0)
    fx = float(f) if f is not None else float(params.get('fx', {}).get('val', 0.0))
    fy = float(fx * ar) if f is not None else float(params.get('fy', {}).get('val', 0.0))
    cx = float(params.get('cx', {}).get('val', 0.0))
    cy = float(params.get('cy', {}).get('val', 0.0))

    # 畸变系数
    dist = {
        'k1': float(params.get('k1', {}).get('val', 0.0)),
        'k2': float(params.get('k2', {}).get('val', 0.0)),
        'k3': float(params.get('k3', {}).get('val', 0.0)),
        'k4': float(params.get('k4', {}).get('val', 0.0)),
        'p1': float(params.get('p1', {}).get('val', 0.0)),
        'p2': float(params.get('p2', {}).get('val', 0.0)),
    }
    
    return (fx, fy), (cx, cy), (width, height), dist


def extract_extrinsics(cam_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """从相机条目提取外参（4x4变换矩阵，world-to-camera）"""
    # 优先从 cam_entry['transform'] 读取（libCalib 格式）
    transform = cam_entry.get('transform', {})
    if transform:
        rotation = transform.get('rotation', {})
        translation = transform.get('translation', {})
    else:
        # 回退到 model.ptr_wrapper.data.pose
        model = cam_entry.get('model', {})
        data = model.get('ptr_wrapper', {}).get('data', {})
        pose = data.get('pose', {})
        rotation = pose.get('rotation', {})
        translation = pose.get('translation', {})
    
    if not rotation or not translation:
        return None
    
    # 旋转格式判断
    if 'rx' in rotation:
        # Rodrigues 向量格式 (rx, ry, rz)
        rvec = np.array([rotation['rx'], rotation['ry'], rotation['rz']], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
    elif 'w' in rotation:
        # 四元数格式 (w, x, y, z)
        qw = rotation.get('w', 1.0)
        qx = rotation.get('x', 0.0)
        qy = rotation.get('y', 0.0)
        qz = rotation.get('z', 0.0)
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    elif 'data' in rotation:
        # 旋转矩阵格式
        R = np.array(rotation['data']).reshape(3, 3)
    else:
        return None
    
    # 平移
    if 'data' in translation:
        t = np.array(translation['data']).reshape(3)
    elif 'x' in translation:
        t = np.array([translation['x'], translation['y'], translation['z']])
    else:
        return None
    
    # 构建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """四元数转旋转矩阵"""
    # 归一化
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-10:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """旋转矩阵转四元数"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return qw, qx, qy, qz


# =============================================================================
# COLMAP文件操作
# =============================================================================

def create_colmap_sparse(cameras_data: List[Dict], output_dir: str) -> str:
    """创建COLMAP稀疏重建文件（文本格式，带畸变参数）"""
    sparse_dir = os.path.join(output_dir, 'sparse', '0')
    os.makedirs(sparse_dir, exist_ok=True)
    
    # 写入cameras.txt（FULL_OPENCV模型，包含k3）
    cameras_txt_path = os.path.join(sparse_dir, 'cameras.txt')
    with open(cameras_txt_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_data)}\n")
        
        for cam in cameras_data:
            cam_id = cam['id']
            width = cam['width']
            height = cam['height']
            fx, fy = cam['fx'], cam['fy']
            cx, cy = cam['cx'], cam['cy']
            dist = cam['dist']
            
            k1 = dist['k1']
            k2 = dist['k2']
            k3 = dist['k3']
            k4 = dist.get('k4', 0.0)
            p1 = dist['p1']
            p2 = dist['p2']
            k5, k6 = 0.0, 0.0
            
            # FULL_OPENCV模型: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            f.write(f"{cam_id} FULL_OPENCV {width} {height} "
                    f"{fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f} "
                    f"{k1:.10f} {k2:.10f} {p1:.10f} {p2:.10f} "
                    f"{k3:.10f} {k4:.10f} {k5:.10f} {k6:.10f}\n")
    
    # 写入images.txt
    images_txt_path = os.path.join(sparse_dir, 'images.txt')
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of registered images: {len(cameras_data)}\n")
        
        for cam in cameras_data:
            cam_id = cam['id']
            qw, qx, qy, qz = cam['quaternion']
            tx, ty, tz = cam['translation']
            img_name = cam['image_name']
            
            f.write(f"{cam_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} {img_name}\n")
            f.write("\n")
    
    # 写入空的points3D.txt
    points3d_txt_path = os.path.join(sparse_dir, 'points3D.txt')
    with open(points3d_txt_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    return sparse_dir


def parse_cameras_txt(cameras_path: str) -> dict:
    """解析cameras.txt文件"""
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def parse_images_txt(images_path: str) -> dict:
    """解析images.txt文件"""
    images = {}
    with open(images_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]
            
            images[img_id] = {
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': cam_id,
                'name': name
            }
            i += 2
        else:
            i += 1
    
    return images


# =============================================================================
# 主点居中处理
# =============================================================================

def compute_unified_crop(cameras: dict) -> Tuple[int, int, dict]:
    """计算统一的裁剪参数，确保所有相机的主点都在中心"""
    crop_info = {}
    
    for cam_id, cam in cameras.items():
        width = cam['width']
        height = cam['height']
        params = cam['params']
        
        if cam['model'] == 'PINHOLE' and len(params) >= 4:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            print(f"警告: Camera {cam_id} 模型 {cam['model']} 不支持")
            continue
        
        left = cx
        right = width - cx
        top = cy
        bottom = height - cy
        
        half_w = min(left, right)
        half_h = min(top, bottom)
        
        crop_info[cam_id] = {
            'new_width': int(2 * half_w),
            'new_height': int(2 * half_h),
            'fx': fx,
            'fy': fy,
            'old_cx': cx,
            'old_cy': cy
        }
    
    if not crop_info:
        raise RuntimeError("没有有效的相机数据")
    
    min_width = min(c['new_width'] for c in crop_info.values())
    min_height = min(c['new_height'] for c in crop_info.values())
    
    unified_width = min_width - (min_width % 2)
    unified_height = min_height - (min_height % 2)
    
    for cam_id in crop_info:
        cam = cameras[cam_id]
        cx, cy = cam['params'][2], cam['params'][3]
        
        half_w = unified_width / 2
        half_h = unified_height / 2
        
        crop_info[cam_id]['unified_width'] = unified_width
        crop_info[cam_id]['unified_height'] = unified_height
        crop_info[cam_id]['crop_x'] = int(cx - half_w)
        crop_info[cam_id]['crop_y'] = int(cy - half_h)
    
    return unified_width, unified_height, crop_info


def center_principal_point(undistorted_dir: str, output_dir: str) -> Tuple[int, int]:
    """裁剪去畸变后的图像使主点居中"""
    input_sparse = os.path.join(undistorted_dir, 'sparse')
    input_images = os.path.join(undistorted_dir, 'images')
    
    # 如果是二进制格式，先转换
    if os.path.exists(os.path.join(input_sparse, 'cameras.bin')):
        print("  转换二进制格式到文本...")
        rec = pycolmap.Reconstruction(input_sparse)
        sparse_txt = os.path.join(undistorted_dir, 'sparse_txt')
        os.makedirs(sparse_txt, exist_ok=True)
        rec.write_text(sparse_txt)
        input_sparse = sparse_txt
    
    cameras = parse_cameras_txt(os.path.join(input_sparse, 'cameras.txt'))
    images = parse_images_txt(os.path.join(input_sparse, 'images.txt'))
    
    unified_width, unified_height, crop_info = compute_unified_crop(cameras)
    print(f"  统一输出尺寸: {unified_width} x {unified_height}")
    print(f"  主点位置: ({unified_width/2}, {unified_height/2}) [完全居中]")
    
    output_images = os.path.join(output_dir, 'images')
    output_sparse = os.path.join(output_dir, 'sparse', '0')
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_sparse, exist_ok=True)
    
    print("  裁剪图像...")
    for img_id, img_data in images.items():
        cam_id = img_data['camera_id']
        img_name = img_data['name']
        
        if cam_id not in crop_info:
            continue
        
        crop = crop_info[cam_id]
        img_path = os.path.join(input_images, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        crop_x = max(0, min(crop['crop_x'], w - crop['unified_width']))
        crop_y = max(0, min(crop['crop_y'], h - crop['unified_height']))
        crop_w = crop['unified_width']
        crop_h = crop['unified_height']
        
        cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        cv2.imwrite(os.path.join(output_images, img_name), cropped)
    
    # 写入cameras.txt（主点在中心）
    with open(os.path.join(output_sparse, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        
        for cam_id, cam in cameras.items():
            if cam_id not in crop_info:
                continue
            fx = crop_info[cam_id]['fx']
            fy = crop_info[cam_id]['fy']
            cx = unified_width / 2.0
            cy = unified_height / 2.0
            f.write(f"{cam_id} PINHOLE {unified_width} {unified_height} "
                    f"{fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")
    
    # 复制images.txt
    shutil.copy(os.path.join(input_sparse, 'images.txt'),
                os.path.join(output_sparse, 'images.txt'))
    
    # 创建空的points3D.txt
    with open(os.path.join(output_sparse, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    return unified_width, unified_height


# =============================================================================
# 批量处理（by_frame 等子目录结构）
# =============================================================================

def build_camera_processors(cams: List[Dict], reference_dir: str,
                             pattern: str) -> Tuple[List[Dict], int, int]:
    """
    预计算每个相机的去畸变参数（仅计算一次，供批量处理使用）。
    用参考帧目录中的样本图像确定实际图像尺寸。

    Returns: (processors列表, 统一输出宽度, 统一输出高度)
    """
    # 从参考目录找样本图像
    sample_images = []
    for pat in [pattern, '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        sample_images = sorted(glob.glob(os.path.join(reference_dir, pat)),
                               key=_natural_sort_key)
        if sample_images:
            break

    processors = []
    for i, cam_entry in enumerate(cams):
        (fx, fy), (cx, cy), (w0, h0), dist = extract_intrinsics(cam_entry)

        # 从样本图像获取实际尺寸，并按比例缩放内参
        actual_w, actual_h = w0, h0
        if i < len(sample_images):
            sample = cv2.imread(sample_images[i])
            if sample is not None:
                actual_h, actual_w = sample.shape[:2]
                if w0 > 0 and h0 > 0 and (actual_w != w0 or actual_h != h0):
                    sx, sy = actual_w / w0, actual_h / h0
                    fx, fy = fx * sx, fy * sy
                    cx, cy = cx * sx, cy * sy
        elif w0 <= 0 or h0 <= 0:
            print(f"  警告: 相机 {i} 尺寸未知，跳过")
            continue

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([dist['k1'], dist['k2'], dist['p1'],
                      dist['p2'], dist['k3']], dtype=np.float64)

        # 计算最优新相机矩阵（alpha=1 保留所有源像素）
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, D, (actual_w, actual_h), 1, (actual_w, actual_h))

        # 预计算去畸变映射（只算一次，后续所有帧复用）
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (actual_w, actual_h), cv2.CV_32FC1)

        # 提取外参
        T_w2c = extract_extrinsics(cam_entry)
        if T_w2c is not None:
            R_ext = T_w2c[:3, :3]
            t_ext = T_w2c[:3, 3]
            qw_, qx_, qy_, qz_ = rotation_matrix_to_quaternion(R_ext)
        else:
            qw_, qx_, qy_, qz_ = 1.0, 0.0, 0.0, 0.0
            t_ext = np.zeros(3)

        processors.append({
            'idx':       i,
            'cam_id':    i + 1,
            'map1':      map1,
            'map2':      map2,
            'src_width':  actual_w,
            'src_height': actual_h,
            'cx_new':    float(new_K[0, 2]),
            'cy_new':    float(new_K[1, 2]),
            'fx_new':    float(new_K[0, 0]),
            'fy_new':    float(new_K[1, 1]),
            'qw': qw_, 'qx': qx_, 'qy': qy_, 'qz': qz_,
            'tx': float(t_ext[0]), 'ty': float(t_ext[1]), 'tz': float(t_ext[2]),
        })

    if not processors:
        raise RuntimeError("未能构建任何相机处理器")

    # 计算统一裁剪尺寸（使所有相机主点居中）
    min_half_w = min(min(p['cx_new'], p['src_width']  - p['cx_new']) for p in processors)
    min_half_h = min(min(p['cy_new'], p['src_height'] - p['cy_new']) for p in processors)
    unified_w = int(2 * min_half_w)
    unified_h = int(2 * min_half_h)
    unified_w -= unified_w % 2
    unified_h -= unified_h % 2

    for p in processors:
        p['unified_w'] = unified_w
        p['unified_h'] = unified_h
        p['crop_x']    = int(p['cx_new'] - unified_w / 2)
        p['crop_y']    = int(p['cy_new'] - unified_h / 2)
        p['final_fx']  = p['fx_new']
        p['final_fy']  = p['fy_new']
        p['final_cx']  = unified_w / 2.0
        p['final_cy']  = unified_h / 2.0

    return processors, unified_w, unified_h


def apply_processor(img: np.ndarray, proc: Dict) -> np.ndarray:
    """对单张图像应用预计算的去畸变映射，并裁剪使主点居中。"""
    h, w = img.shape[:2]
    if w != proc['src_width'] or h != proc['src_height']:
        img = cv2.resize(img, (proc['src_width'], proc['src_height']))

    undist = cv2.remap(img, proc['map1'], proc['map2'], cv2.INTER_LINEAR)

    cw = proc['unified_w']
    ch = proc['unified_h']
    uh, uw = undist.shape[:2]
    cx = max(0, min(proc['crop_x'], uw - cw))
    cy = max(0, min(proc['crop_y'], uh - ch))
    return undist[cy:cy + ch, cx:cx + cw]


def process_batch_by_frame(images_dir: str, output_dir: str,
                           processors: List[Dict],
                           unified_w: int, unified_h: int,
                           pattern: str,
                           num_workers: int = 0) -> None:
    """
    批量处理 by_frame 结构的图像目录，保持原有子目录结构。
    使用多线程并行处理各帧，tqdm 实时显示进度。

    输入:  images_dir/{frame_id}/{images}
    输出:  output_dir/images/{frame_id}/{images}
           output_dir/sparse/cameras.txt  (PINHOLE，主点居中)
           output_dir/sparse/images.txt
           output_dir/sparse/points3D.txt
    """
    in_root    = Path(images_dir)
    out_images = Path(output_dir) / 'images'
    out_sparse = Path(output_dir) / 'sparse' / '0'
    out_images.mkdir(parents=True, exist_ok=True)
    out_sparse.mkdir(parents=True, exist_ok=True)

    # 帧目录排序：数字名称按数值，其余按字典序
    def _sort_key(d: Path):
        return (0, int(d.name)) if d.name.isdigit() else (1, d.name)

    frame_dirs = sorted([d for d in in_root.iterdir() if d.is_dir()], key=_sort_key)
    total = len(frame_dirs)

    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 4, 8)
    print(f"  找到 {total} 个帧目录，{len(processors)} 个相机，{num_workers} 个线程")

    # ------------------------------------------------------------------
    # 单帧处理函数（在线程池中并行调用）
    # ------------------------------------------------------------------
    def _process_frame(frame_dir: Path) -> Tuple[Path, List[Dict]]:
        records: List[Dict] = []
        frame_imgs: List[str] = []
        for pat in [pattern, '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            frame_imgs = sorted(glob.glob(str(frame_dir / pat)),
                                key=_natural_sort_key)
            if frame_imgs:
                break
        if not frame_imgs:
            return frame_dir, records

        out_frame = out_images / frame_dir.name
        out_frame.mkdir(parents=True, exist_ok=True)

        for j, img_path in enumerate(frame_imgs):
            if j >= len(processors):
                break
            img = cv2.imread(img_path)
            if img is None:
                continue
            result   = apply_processor(img, processors[j])
            out_name = f"{processors[j]['cam_id']:03d}.png"
            cv2.imwrite(str(out_frame / out_name), result)
            records.append({
                'cam_id': processors[j]['cam_id'],
                'name':   out_name,
            })
        return frame_dir, records

    # ------------------------------------------------------------------
    # 并行提交所有帧，tqdm 实时更新进度
    # ------------------------------------------------------------------
    frame_results: Dict[Path, List[Dict]] = {}
    pbar = tqdm(total=total, desc="  去畸变", unit="帧", ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") \
           if HAS_TQDM else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_fd = {executor.submit(_process_frame, fd): fd for fd in frame_dirs}
        for future in concurrent.futures.as_completed(future_to_fd):
            fd, records = future.result()
            frame_results[fd] = records
            if pbar is not None:
                pbar.update(1)
            elif len(frame_results) % 100 == 0 or len(frame_results) == total:
                print(f"  进度: {len(frame_results)}/{total} 帧")

    if pbar is not None:
        pbar.close()

    # 统计总输出数量
    total_images = sum(len(recs) for recs in frame_results.values())
    print(f"  共输出 {total_images} 张图像")

    # 写 cameras.txt
    with open(str(out_sparse / 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(processors)}\n")
        for p in processors:
            f.write(f"{p['cam_id']} PINHOLE {unified_w} {unified_h} "
                    f"{p['final_fx']:.10f} {p['final_fy']:.10f} "
                    f"{p['final_cx']:.10f} {p['final_cy']:.10f}\n")

    # 写 images.txt（只写第一帧，使用 calib.json 中的实际外参）
    first_frame_recs = frame_results.get(frame_dirs[0], [])
    proc_by_cam = {p['cam_id']: p for p in processors}
    with open(str(out_sparse / 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(first_frame_recs)}, mean observations per image: 0\n")
        for img_id, rec in enumerate(first_frame_recs, start=1):
            p = proc_by_cam[rec['cam_id']]
            f.write(f"{img_id} {p['qw']:.16f} {p['qx']:.16f} {p['qy']:.16f} {p['qz']:.16f} "
                    f"{p['tx']:.16f} {p['ty']:.16f} {p['tz']:.16f} {rec['cam_id']} {rec['name']}\n")
            f.write("\n")

    # 写 points3D.txt（空）
    with open(str(out_sparse / 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='使用 OpenCV/pycolmap 对 calib.json 进行图像去畸变 - 用于3DGS训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单目录模式（需要 pycolmap）：
  python undistort_from_calib.py --calib calib.json --images_dir images --output_dir output

  # 批量模式（images_dir 含子目录，如 by_frame，无需 pycolmap）：
  python undistort_from_calib.py --calib calib.json --images_dir images/by_frame --output_dir output
        """)
    parser.add_argument('--calib', type=str, required=True,
                        help='calib.json文件路径')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='原始图像目录（若包含子目录则自动启用批量模式）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='图像文件匹配模式 (默认: *.png)')
    parser.add_argument('--workers', type=int, default=0,
                        help='批量模式线程数，0=自动（默认: 0）')

    args = parser.parse_args()

    print("="*60)
    print("Calib.json图像去畸变工具")
    print("="*60)

    # 步骤1: 解析calib.json（两种模式共用）
    print(f"\n[1/?] 解析calib.json: {args.calib}")
    with open(args.calib, 'r') as f:
        calib = json.load(f)

    cams = calib.get('Calibration', {}).get('cameras', [])
    if not cams:
        raise RuntimeError('calib.json中未找到相机')

    print(f"  找到 {len(cams)} 个相机")

    # 检测模式：images_dir 含子目录 → 批量模式，否则 → 单目录模式
    images_dir_path = Path(args.images_dir)
    subdirs = sorted(
        [d for d in images_dir_path.iterdir() if d.is_dir()],
        key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name)
    )
    batch_mode = len(subdirs) > 0

    if batch_mode:
        # ------------------------------------------------------------------
        # 批量模式：预计算去畸变映射，逐帧处理，保持子目录结构输出
        # ------------------------------------------------------------------
        print(f"\n检测到批量模式（{len(subdirs)} 个子目录）")

        ref_dir = subdirs[0]
        print(f"\n[2/3] 预计算相机去畸变参数（参考帧: {ref_dir.name}）...")
        processors, unified_w, unified_h = build_camera_processors(
            cams, str(ref_dir), args.pattern)
        print(f"  {len(processors)} 个相机，统一输出尺寸: {unified_w} x {unified_h}")
        print(f"  主点位置: ({unified_w/2}, {unified_h/2}) [完全居中]")

        print(f"\n[3/3] 批量处理帧目录...")
        os.makedirs(args.output_dir, exist_ok=True)
        process_batch_by_frame(
            args.images_dir, args.output_dir,
            processors, unified_w, unified_h, args.pattern,
            num_workers=args.workers)

        print("\n" + "="*60)
        print("处理完成！")
        print("="*60)
        print(f"输出目录: {args.output_dir}")
        print(f"  - 图像: {os.path.join(args.output_dir, 'images')}/")
        print(f"    （子目录结构同 {images_dir_path.name}/）")
        print(f"  - cameras.txt: {os.path.join(args.output_dir, 'sparse', '0', 'cameras.txt')}")
        print(f"  - images.txt:  {os.path.join(args.output_dir, 'sparse', '0', 'images.txt')}")
        print(f"  - points3D.txt:{os.path.join(args.output_dir, 'sparse', '0', 'points3D.txt')}")
        print(f"\n图像尺寸: {unified_w} x {unified_h}")
        print(f"主点位置: ({unified_w/2}, {unified_h/2}) [完全居中]")
        print(f"相机模型: PINHOLE (无畸变)")
        print("\n✓ 可直接用于3DGS训练")

    else:
        # ------------------------------------------------------------------
        # 单目录模式：原有 pycolmap 流程（保持不变）
        # ------------------------------------------------------------------
        if not HAS_PYCOLMAP:
            print("错误: pycolmap未安装，请运行 pip install pycolmap")
            return 1

        # 创建临时目录
        temp_dir = os.path.join(args.output_dir, '_temp')
        os.makedirs(temp_dir, exist_ok=True)

        # 步骤2: 查找图像并匹配相机
        print(f"\n[2/4] 准备图像...")
        image_paths = sorted(glob.glob(os.path.join(args.images_dir, args.pattern)))
        if not image_paths:
            for ext in ['*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_paths = sorted(glob.glob(os.path.join(args.images_dir, ext)))
                if image_paths:
                    break

        if not image_paths:
            raise RuntimeError(f"在 {args.images_dir} 中未找到图像")

        print(f"  找到 {len(image_paths)} 张图像")

        # 准备图像目录
        images_input_dir = os.path.join(temp_dir, 'images_input')
        os.makedirs(images_input_dir, exist_ok=True)

        # 图像和相机一一对应
        per_cam = len(image_paths) == len(cams)

        cameras_data = []
        for i, img_path in enumerate(image_paths):
            cam_idx = i if per_cam else 0
            cam_entry = cams[cam_idx]

            # 提取内参
            (fx, fy), (cx, cy), (w0, h0), dist = extract_intrinsics(cam_entry)

            # 读取图像获取实际尺寸
            img = cv2.imread(img_path)
            if img is None:
                print(f"  警告: 无法读取 {img_path}")
                continue

            h, w = img.shape[:2]

            # 如果图像尺寸与标定尺寸不同，缩放内参
            if w0 > 0 and h0 > 0 and (w != w0 or h != h0):
                sx, sy = w / w0, h / h0
                fx, fy = fx * sx, fy * sy
                cx, cy = cx * sx, cy * sy

            # 提取外参
            T_w2c = extract_extrinsics(cam_entry)
            if T_w2c is None:
                T_w2c = np.eye(4)

            R = T_w2c[:3, :3]
            t = T_w2c[:3, 3]
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)

            # 复制图像（移除空格）
            new_name = Path(img_path).name.replace(' ', '')
            shutil.copy2(img_path, os.path.join(images_input_dir, new_name))

            cameras_data.append({
                'id': i + 1,
                'width': w,
                'height': h,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'dist': dist,
                'quaternion': (qw, qx, qy, qz),
                'translation': (t[0], t[1], t[2]),
                'image_name': new_name
            })

        print(f"  已处理 {len(cameras_data)} 张图像")

        # 创建COLMAP稀疏模型
        sparse_dir = create_colmap_sparse(cameras_data, temp_dir)
        print(f"  已创建COLMAP稀疏模型")

        # 步骤3: pycolmap去畸变
        print(f"\n[3/4] 使用pycolmap去畸变...")
        undistorted_dir = os.path.join(temp_dir, 'undistorted')

        pycolmap.undistort_images(
            output_path=undistorted_dir,
            input_path=sparse_dir,
            image_path=images_input_dir,
            output_type="COLMAP"
        )
        print(f"  去畸变完成")

        # 步骤4: 裁剪使主点居中
        print(f"\n[4/4] 裁剪图像使主点居中...")
        unified_width, unified_height = center_principal_point(undistorted_dir, args.output_dir)

        # 清理临时目录
        print(f"\n清理临时文件...")
        shutil.rmtree(temp_dir)

        print("\n" + "="*60)
        print("处理完成！")
        print("="*60)
        print(f"输出目录: {args.output_dir}")
        print(f"  - 图像: {os.path.join(args.output_dir, 'images')}")
        print(f"  - cameras.txt: {os.path.join(args.output_dir, 'sparse', '0', 'cameras.txt')}")
        print(f"  - images.txt: {os.path.join(args.output_dir, 'sparse', '0', 'images.txt')}")
        print(f"  - points3D.txt: {os.path.join(args.output_dir, 'sparse', '0', 'points3D.txt')}")
        print(f"\n图像尺寸: {unified_width} x {unified_height}")
        print(f"主点位置: ({unified_width/2}, {unified_height/2}) [完全居中]")
        print(f"相机模型: PINHOLE (无畸变)")
        print("\n✓ 可直接用于3DGS训练")

    return 0


if __name__ == '__main__':
    exit(main())
