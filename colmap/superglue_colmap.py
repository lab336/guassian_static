"""Run COLMAP reconstruction with SuperGlue matches instead of COLMAP matcher.

The script expects a multi-camera frame layout:

  data/twopeople/images/<frame_dir>/<camera_image>.png

It creates a COLMAP database, writes SuperPoint+SuperGlue keypoints/matches to
the database, then runs COLMAP mapper/model_converter.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


MAX_IMAGE_ID = 2147483647
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
}

LOGGER = logging.getLogger("superglue_colmap")


@dataclass(frozen=True)
class ImageInfo:
    name: str
    path: Path
    width: int
    height: int
    image_id: int
    camera_id: int


@dataclass(frozen=True)
class PairMatch:
    name0: str
    name1: str
    matches: np.ndarray  # Nx4 original-image coordinates: x0, y0, x1, y1


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray
    rotmat: np.ndarray


@dataclass(frozen=True)
class ColmapPoint3D:
    point_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: list[tuple[int, int]]


@dataclass(frozen=True)
class VelocityResult:
    velocities: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    view_counts: np.ndarray


class CommandError(RuntimeError):
    pass


def natural_key(path_or_name: Path | str) -> tuple:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    stem = Path(name).stem
    if stem.isdigit():
        return (0, int(stem), name)
    parts: list[tuple[int, int | str]] = []
    chunk = ""
    is_digit = stem[:1].isdigit()
    for ch in stem:
        if ch.isdigit() == is_digit:
            chunk += ch
        else:
            parts.append((0, int(chunk)) if is_digit else (1, chunk))
            chunk = ch
            is_digit = ch.isdigit()
    if chunk:
        parts.append((0, int(chunk)) if is_digit else (1, chunk))
    return (1, parts, name)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_colmap(colmap_arg: str) -> str:
    colmap = shutil.which(colmap_arg) if os.path.basename(colmap_arg) == colmap_arg else colmap_arg
    if not colmap:
        raise FileNotFoundError("COLMAP was not found in PATH. Use --colmap to point to colmap.exe.")
    return colmap


def run_command(cmd: Sequence[str], log_path: Path | None = None) -> None:
    LOGGER.info("run: %s", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    if sys.platform == "win32":
        exe_dir = str(Path(cmd[0]).resolve().parent)
        conda_prefix = env.get("CONDA_PREFIX", "")
        prepend = [exe_dir]
        if conda_prefix:
            prepend.extend(
                str(Path(conda_prefix) / p)
                for p in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin", "Scripts")
            )
        env["PATH"] = os.pathsep.join(prepend) + os.pathsep + env.get("PATH", "")

    started = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    elapsed = time.time() - started
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            " ".join(str(x) for x in cmd)
            + "\n\n[stdout]\n"
            + result.stdout
            + "\n\n[stderr]\n"
            + result.stderr,
            encoding="utf-8",
        )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-30:])
        raise CommandError(f"command failed with exit={result.returncode}: {' '.join(cmd)}\n{tail}")
    LOGGER.info("done in %.1fs", elapsed)


def discover_frame_dirs(images_root: Path) -> list[Path]:
    if not images_root.exists():
        raise FileNotFoundError(f"images root does not exist: {images_root}")
    frames = sorted([p for p in images_root.iterdir() if p.is_dir()], key=natural_key)
    if not frames:
        raise FileNotFoundError(f"no frame folders found in {images_root}")
    return frames


def parse_frame_selection(frame_dirs: Sequence[Path], frames_arg: str | None) -> list[Path]:
    if not frames_arg:
        return list(frame_dirs)
    by_name = {p.name: p for p in frame_dirs}
    selected: list[Path] = []
    for token in [x.strip() for x in frames_arg.split(",") if x.strip()]:
        if ":" in token:
            start_s, end_s = token.split(":", 1)
            for value in range(int(start_s), int(end_s) + 1):
                key = str(value)
                if key in by_name:
                    selected.append(by_name[key])
        elif token in by_name:
            selected.append(by_name[token])
        elif token.isdigit() and 1 <= int(token) <= len(frame_dirs):
            selected.append(frame_dirs[int(token) - 1])
        else:
            raise ValueError(f"frame '{token}' not found")
    unique: list[Path] = []
    seen: set[Path] = set()
    for frame in selected:
        if frame not in seen:
            unique.append(frame)
            seen.add(frame)
    if not unique:
        raise ValueError("--frames selected no existing frame folders")
    return unique


def list_images(frame_dir: Path, max_images: int | None = None) -> list[Path]:
    images = sorted(
        [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if max_images is not None:
        images = images[:max_images]
    if len(images) < 2:
        raise ValueError(f"need at least two images in {frame_dir}, got {len(images)}")
    return images


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read image size: {path}")
    return int(img.shape[1]), int(img.shape[0])


def stage_images(src_images: Sequence[Path], image_dir: Path, copy_images: bool) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for src in src_images:
        dst = image_dir / src.name
        if dst.exists():
            dst.unlink()
        if copy_images:
            shutil.copy2(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def camera_params(model: str, width: int, height: int, focal_factor: float) -> np.ndarray:
    focal = focal_factor * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    if model == "SIMPLE_PINHOLE":
        return np.asarray([focal, cx, cy], dtype=np.float64)
    if model == "PINHOLE":
        return np.asarray([focal, focal, cx, cy], dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        return np.asarray([focal, cx, cy, 0.0], dtype=np.float64)
    if model == "RADIAL":
        return np.asarray([focal, cx, cy, 0.0, 0.0], dtype=np.float64)
    if model == "OPENCV":
        return np.asarray([focal, focal, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    raise ValueError(f"unsupported camera model: {model}")


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def array_to_blob(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def reset_database(colmap: str, database_path: Path, log_dir: Path) -> None:
    if database_path.exists():
        database_path.unlink()
    run_command([colmap, "database_creator", "--database_path", str(database_path)], log_dir / "database_creator.log")


def create_colmap_database(
    database_path: Path,
    images: Sequence[Path],
    camera_model: str,
    focal_factor: float,
    single_camera: bool,
) -> dict[str, ImageInfo]:
    con = sqlite3.connect(database_path)
    try:
        model_id = CAMERA_MODEL_IDS[camera_model]
        image_infos: dict[str, ImageInfo] = {}
        shared_camera_id: int | None = None
        shared_size: tuple[int, int] | None = None
        for image_path in images:
            width, height = image_size(image_path)
            if single_camera:
                if shared_camera_id is None:
                    shared_size = (width, height)
                    params = camera_params(camera_model, width, height, focal_factor)
                    cur = con.execute(
                        "INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                        (model_id, width, height, array_to_blob(params), 0),
                    )
                    shared_camera_id = int(cur.lastrowid)
                elif shared_size != (width, height):
                    raise ValueError("--single_camera requires all images to have identical sizes")
                camera_id = shared_camera_id
            else:
                params = camera_params(camera_model, width, height, focal_factor)
                cur = con.execute(
                    "INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                    (model_id, width, height, array_to_blob(params), 0),
                )
                camera_id = int(cur.lastrowid)

            cur = con.execute("INSERT INTO images(name, camera_id) VALUES (?, ?)", (image_path.name, camera_id))
            image_id = int(cur.lastrowid)
            image_infos[image_path.name] = ImageInfo(image_path.name, image_path, width, height, image_id, camera_id)
        con.commit()
        return image_infos
    finally:
        con.close()


def build_pairs(
    image_names: Sequence[str],
    mode: str,
    window: int,
    loop: bool,
    pairs_file: Path | None = None,
) -> list[tuple[str, str]]:
    if mode == "exhaustive":
        pairs = list(itertools.combinations(image_names, 2))
    elif mode == "sequential":
        pairs = []
        n = len(image_names)
        for i in range(n):
            for step in range(1, min(window, n - 1) + 1):
                j = i + step
                if j < n:
                    pairs.append((image_names[i], image_names[j]))
                elif loop:
                    pairs.append((image_names[i], image_names[j % n]))
    elif mode == "pairs_file":
        if not pairs_file:
            raise ValueError("--pairs_file is required when --pair_mode pairs_file")
        name_set = set(image_names)
        pairs = []
        for line in pairs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2 or parts[0] not in name_set or parts[1] not in name_set:
                raise ValueError(f"invalid pair line: {line}")
            pairs.append((parts[0], parts[1]))
    else:
        raise ValueError(f"unsupported pair mode: {mode}")

    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    order = {name: idx for idx, name in enumerate(image_names)}
    for a, b in pairs:
        if a == b:
            continue
        pair = (a, b) if order[a] < order[b] else (b, a)
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    return normalized


def process_resize(width: int, height: int, resize: Sequence[int]) -> tuple[int, int]:
    if len(resize) == 2:
        return int(resize[0]), int(resize[1])
    if len(resize) == 1 and resize[0] > 0:
        scale = float(resize[0]) / float(max(width, height))
        return int(round(width * scale)), int(round(height * scale))
    return width, height


class SuperGlueMatcher:
    def __init__(
        self,
        superglue_root: Path,
        weights: str,
        resize: Sequence[int],
        resize_float: bool,
        max_keypoints: int,
        keypoint_threshold: float,
        nms_radius: int,
        sinkhorn_iterations: int,
        match_threshold: float,
        device: str,
    ) -> None:
        self.superglue_root = superglue_root.resolve()
        if not self.superglue_root.exists():
            raise FileNotFoundError(f"SuperGlue root does not exist: {self.superglue_root}")
        sys.path.insert(0, str(self.superglue_root))

        import torch
        from models.matching import Matching

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; using CPU")
            device = "cpu"
        self.torch = torch
        self.device = device
        self.resize = list(resize)
        self.resize_float = resize_float
        self.matching = Matching(
            {
                "superpoint": {
                    "nms_radius": nms_radius,
                    "keypoint_threshold": keypoint_threshold,
                    "max_keypoints": max_keypoints,
                },
                "superglue": {
                    "weights": weights,
                    "sinkhorn_iterations": sinkhorn_iterations,
                    "match_threshold": match_threshold,
                },
            }
        ).eval().to(device)
        self.torch.set_grad_enabled(False)

    def _load_gray(self, path: Path) -> tuple[np.ndarray, tuple[float, float]]:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        width, height = image.shape[1], image.shape[0]
        new_width, new_height = process_resize(width, height, self.resize)
        scales = (float(width) / float(new_width), float(height) / float(new_height))
        if (new_width, new_height) != (width, height):
            image = cv2.resize(image.astype("float32" if self.resize_float else "uint8"), (new_width, new_height))
        return image, scales

    def match_pair(self, image0: Path, image1: Path) -> np.ndarray:
        img0, scales0 = self._load_gray(image0)
        img1, scales1 = self._load_gray(image1)
        tensor0 = self.torch.from_numpy(img0 / 255.0).float()[None, None].to(self.device)
        tensor1 = self.torch.from_numpy(img1 / 255.0).float()[None, None].to(self.device)
        pred = self.matching({"image0": tensor0, "image1": tensor1})
        pred_np = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0 = pred_np["keypoints0"]
        kpts1 = pred_np["keypoints1"]
        matches = pred_np["matches0"]
        valid = matches > -1
        if not np.any(valid):
            return np.empty((0, 4), dtype=np.float32)

        mkpts0 = kpts0[valid].astype(np.float32, copy=True)
        mkpts1 = kpts1[matches[valid]].astype(np.float32, copy=True)
        mkpts0[:, 0] *= scales0[0]
        mkpts0[:, 1] *= scales0[1]
        mkpts1[:, 0] *= scales1[0]
        mkpts1[:, 1] *= scales1[1]
        return np.concatenate([mkpts0, mkpts1], axis=1).astype(np.float32, copy=False)


def filter_corrs(
    corrs: np.ndarray,
    info0: ImageInfo,
    info1: ImageInfo,
    min_matches: int,
    ransac: bool,
    ransac_max_error: float,
    ransac_confidence: float,
) -> np.ndarray:
    if corrs.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    corrs = np.asarray(corrs, dtype=np.float32).reshape(-1, 4)
    finite = np.isfinite(corrs).all(axis=1)
    in_bounds = (
        (corrs[:, 0] >= 0)
        & (corrs[:, 0] < info0.width)
        & (corrs[:, 1] >= 0)
        & (corrs[:, 1] < info0.height)
        & (corrs[:, 2] >= 0)
        & (corrs[:, 2] < info1.width)
        & (corrs[:, 3] >= 0)
        & (corrs[:, 3] < info1.height)
    )
    corrs = corrs[finite & in_bounds]
    if len(corrs) < min_matches:
        return np.empty((0, 4), dtype=np.float32)

    quant = np.round(corrs * 4.0).astype(np.int64)
    _, unique_idx = np.unique(quant, axis=0, return_index=True)
    corrs = corrs[np.sort(unique_idx)]

    if ransac and len(corrs) >= 8:
        import cv2

        _, mask = cv2.findFundamentalMat(
            corrs[:, :2],
            corrs[:, 2:],
            method=cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_max_error,
            confidence=ransac_confidence,
        )
        if mask is not None:
            corrs = corrs[mask.ravel() > 0]
    if len(corrs) < min_matches:
        return np.empty((0, 4), dtype=np.float32)
    return corrs.astype(np.float32, copy=False)


def quantized_key(pt: Sequence[float], quantization: float) -> tuple[int, int]:
    q = max(float(quantization), 1e-6)
    return (int(round(float(pt[0]) / q)), int(round(float(pt[1]) / q)))


def export_matches_to_database(
    database_path: Path,
    image_infos: dict[str, ImageInfo],
    pair_matches: Sequence[PairMatch],
    keypoint_quantization: float,
    two_view_config: int,
) -> dict[str, int]:
    key_maps: dict[str, dict[tuple[int, int], int]] = {name: {} for name in image_infos}
    keypoints: dict[str, list[tuple[float, float]]] = {name: [] for name in image_infos}
    indexed_pairs: list[tuple[str, str, np.ndarray]] = []

    for pair in pair_matches:
        idx_matches: list[tuple[int, int]] = []
        for x0, y0, x1, y1 in pair.matches:
            key0 = quantized_key((x0, y0), keypoint_quantization)
            key1 = quantized_key((x1, y1), keypoint_quantization)
            map0 = key_maps[pair.name0]
            map1 = key_maps[pair.name1]
            if key0 not in map0:
                map0[key0] = len(keypoints[pair.name0])
                keypoints[pair.name0].append((float(x0), float(y0)))
            if key1 not in map1:
                map1[key1] = len(keypoints[pair.name1])
                keypoints[pair.name1].append((float(x1), float(y1)))
            idx_matches.append((map0[key0], map1[key1]))
        if idx_matches:
            indexed_pairs.append((pair.name0, pair.name1, np.unique(np.asarray(idx_matches, dtype=np.uint32), axis=0)))

    con = sqlite3.connect(database_path)
    try:
        con.execute("DELETE FROM keypoints")
        con.execute("DELETE FROM descriptors")
        con.execute("DELETE FROM matches")
        con.execute("DELETE FROM two_view_geometries")

        for name, pts in keypoints.items():
            image_id = image_infos[name].image_id
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            con.execute(
                "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (image_id, int(arr.shape[0]), 2, array_to_blob(arr)),
            )

        pair_count = 0
        match_count = 0
        eye3 = array_to_blob(np.eye(3, dtype=np.float64))
        qvec = array_to_blob(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        tvec = array_to_blob(np.zeros(3, dtype=np.float64))
        for name0, name1, matches in indexed_pairs:
            info0 = image_infos[name0]
            info1 = image_infos[name1]
            if info0.image_id < info1.image_id:
                image_id0, image_id1 = info0.image_id, info1.image_id
                stored = matches
            else:
                image_id0, image_id1 = info1.image_id, info0.image_id
                stored = matches[:, ::-1].copy()
            pair_id = image_ids_to_pair_id(image_id0, image_id1)
            blob = array_to_blob(stored.astype(np.uint32, copy=False))
            rows = int(stored.shape[0])
            con.execute("INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)", (pair_id, rows, 2, blob))
            con.execute(
                "INSERT OR REPLACE INTO two_view_geometries"
                "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id, rows, 2, blob, two_view_config, eye3, eye3, eye3, qvec, tvec),
            )
            pair_count += 1
            match_count += rows
        con.commit()
        return {
            "images_with_keypoints": sum(1 for pts in keypoints.values() if pts),
            "keypoints": sum(len(pts) for pts in keypoints.values()),
            "pairs": pair_count,
            "matches": match_count,
        }
    finally:
        con.close()


def model_num_registered_images(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    if images_bin.exists():
        try:
            with images_bin.open("rb") as f:
                return int(np.frombuffer(f.read(8), dtype="<u8")[0])
        except Exception:
            return 0
    images_txt = model_dir / "images.txt"
    if images_txt.exists():
        count = 0
        for line in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
        return count // 2
    return 0


def run_mapper(colmap: str, database_path: Path, image_dir: Path, sparse_dir: Path, log_dir: Path, args) -> Path:
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
        "--Mapper.ba_refine_focal_length",
        "1" if args.ba_refine_focal_length else "0",
        "--Mapper.ba_refine_principal_point",
        "1" if args.ba_refine_principal_point else "0",
        "--Mapper.min_num_matches",
        str(args.mapper_min_num_matches),
        "--Mapper.init_min_num_inliers",
        str(args.mapper_init_min_num_inliers),
        "--Mapper.init_max_error",
        str(args.mapper_init_max_error),
        "--Mapper.abs_pose_min_num_inliers",
        str(args.mapper_abs_pose_min_num_inliers),
        "--Mapper.tri_min_angle",
        str(args.mapper_tri_min_angle),
        "--Mapper.multiple_models",
        "1" if args.mapper_multiple_models else "0",
    ]
    run_command(cmd, log_dir / "mapper.log")
    models = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError(f"COLMAP mapper produced no model in {sparse_dir}")
    best = max(models, key=model_num_registered_images)
    if len(models) > 1:
        sizes = {p.name: model_num_registered_images(p) for p in sorted(models, key=natural_key)}
        LOGGER.info("mapper produced %d models %s; selected '%s'", len(models), sizes, best.name)
    return best


def export_model_txt(colmap: str, model_dir: Path, txt_dir: Path, log_dir: Path) -> None:
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_command([colmap, "model_converter", "--input_path", str(model_dir), "--output_path", str(txt_dir), "--output_type", "TXT"], log_dir / "model_converter_txt.log")


def run_dense(colmap: str, image_dir: Path, model_dir: Path, dense_dir: Path, fused_ply: Path, log_dir: Path, args) -> None:
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)
    run_command([colmap, "image_undistorter", "--image_path", str(image_dir), "--input_path", str(model_dir), "--output_path", str(dense_dir), "--output_type", "COLMAP"], log_dir / "image_undistorter.log")
    run_command([colmap, "patch_match_stereo", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.gpu_index", args.gpu_index], log_dir / "patch_match_stereo.log")
    run_command([colmap, "stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(fused_ply), "--StereoFusion.check_num_images", str(args.fusion_check_num_images), "--StereoFusion.min_num_pixels", str(args.fusion_min_num_pixels)], log_dir / "stereo_fusion.log")


def parse_points3d_stats(points3d_txt: Path) -> dict[str, float | int | None]:
    if not points3d_txt.exists():
        return {"num_sparse_points": 0, "mean_reprojection_error": None, "median_reprojection_error": None}
    errors: list[float] = []
    count = 0
    for line in points3d_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            count += 1
            try:
                errors.append(float(parts[7]))
            except ValueError:
                pass
    return {
        "num_sparse_points": count,
        "mean_reprojection_error": float(np.mean(errors)) if errors else None,
        "median_reprojection_error": float(np.median(errors)) if errors else None,
    }


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec]
    return np.asarray(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_colmap_text_model(model_txt_dir: Path) -> tuple[dict[int, ColmapCamera], dict[int, ColmapImage], list[ColmapPoint3D]]:
    cameras: dict[int, ColmapCamera] = {}
    for line in (model_txt_dir / "cameras.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=np.asarray([float(x) for x in parts[4:]], dtype=np.float64),
        )

    images: dict[int, ColmapImage] = {}
    lines = (model_txt_dir / "images.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.asarray([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(x) for x in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        points_line = lines[i].strip() if i < len(lines) else ""
        i += 1
        values = points_line.split()
        triples = len(values) // 3
        xys = np.zeros((triples, 2), dtype=np.float64)
        point3d_ids = np.full(triples, -1, dtype=np.int64)
        for j in range(triples):
            xys[j, 0] = float(values[3 * j])
            xys[j, 1] = float(values[3 * j + 1])
            point3d_ids[j] = int(values[3 * j + 2])
        images[image_id] = ColmapImage(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=xys,
            point3d_ids=point3d_ids,
            rotmat=qvec_to_rotmat(qvec),
        )

    points: list[ColmapPoint3D] = []
    for line in (model_txt_dir / "points3D.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        point_id = int(parts[0])
        xyz = np.asarray([float(x) for x in parts[1:4]], dtype=np.float64)
        rgb = np.asarray([int(x) for x in parts[4:7]], dtype=np.uint8)
        error = float(parts[7])
        track_values = parts[8:]
        track: list[tuple[int, int]] = []
        for j in range(0, len(track_values) - 1, 2):
            track.append((int(track_values[j]), int(track_values[j + 1])))
        points.append(ColmapPoint3D(point_id=point_id, xyz=xyz, rgb=rgb, error=error, track=track))
    return cameras, images, points


def camera_normalized_from_pixel(camera: ColmapCamera, u: float, v: float) -> tuple[float, float]:
    p = camera.params
    model = camera.model
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return (u - cx) / f, (v - cy) / f
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return (u - cx) / fx, (v - cy) / fy
    if model == "SIMPLE_RADIAL":
        import cv2

        f, cx, cy, k1 = p[:4]
        pts = np.asarray([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64), np.asarray([k1, 0, 0, 0], dtype=np.float64))
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    if model == "RADIAL":
        import cv2

        f, cx, cy, k1, k2 = p[:5]
        pts = np.asarray([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64), np.asarray([k1, k2, 0, 0], dtype=np.float64))
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    if model == "OPENCV":
        import cv2

        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        pts = np.asarray([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64), np.asarray([k1, k2, p1, p2], dtype=np.float64))
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    raise ValueError(f"unsupported camera model for velocity: {camera.model}")


def camera_project(camera: ColmapCamera, image: ColmapImage, xyz: np.ndarray) -> tuple[float, float, float]:
    xyz_cam = image.rotmat @ xyz + image.tvec
    depth = float(xyz_cam[2])
    if depth <= 1e-12:
        return np.nan, np.nan, depth
    x = float(xyz_cam[0] / depth)
    y = float(xyz_cam[1] / depth)
    p = camera.params
    model = camera.model
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return f * x + cx, f * y + cy, depth
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return fx * x + cx, fy * y + cy, depth
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        r2 = x * x + y * y
        d = 1.0 + k1 * r2
        return f * x * d + cx, f * y * d + cy, depth
    if model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        r2 = x * x + y * y
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        return f * x * d + cx, f * y * d + cy, depth
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return fx * xd + cx, fy * yd + cy, depth
    raise ValueError(f"unsupported camera model for velocity: {camera.model}")


def bilinear_sample_flow(flow: np.ndarray, u: float, v: float) -> tuple[float, float] | None:
    h, w = flow.shape[:2]
    if not (0 <= u <= w - 1 and 0 <= v <= h - 1):
        return None
    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    wx = u - x0
    wy = v - y0
    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]
    value = (f00 * (1.0 - wx) + f01 * wx) * (1.0 - wy) + (f10 * (1.0 - wx) + f11 * wx) * wy
    if not np.isfinite(value).all():
        return None
    return float(value[0]), float(value[1])


def flow_to_pixel_delta(
    du: float,
    dv: float,
    flow_width: int,
    flow_height: int,
    image_width: int,
    image_height: int,
    flow_format: str,
    flow_scale: float,
) -> tuple[float, float]:
    if flow_format == "norm":
        return du * image_width * flow_scale, dv * image_height * flow_scale
    if flow_format == "midnorm":
        return du * image_width * 0.5 * flow_scale, dv * image_height * 0.5 * flow_scale
    return du * (image_width / flow_width) * flow_scale, dv * (image_height / flow_height) * flow_scale


def load_flow_maps(flow_dir: Path, image_names: Sequence[str]) -> dict[str, np.ndarray]:
    flows: dict[str, np.ndarray] = {}
    if not flow_dir.exists():
        return flows
    for name in image_names:
        path = flow_dir / f"{Path(name).stem}.npy"
        if not path.exists():
            continue
        try:
            arr = np.load(str(path))
        except Exception as exc:
            LOGGER.warning("failed to load flow %s: %s", path, exc)
            continue
        if arr.ndim == 3 and arr.shape[2] == 2:
            flows[name] = arr.astype(np.float32, copy=False)
    return flows


def triangulation_angle_deg(observations: Sequence[tuple[ColmapCamera, ColmapImage, float, float]]) -> float:
    rays = []
    for camera, image, u, v in observations:
        xn, yn = camera_normalized_from_pixel(camera, u, v)
        ray_cam = np.asarray([xn, yn, 1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = image.rotmat.T @ ray_cam
        ray_world /= np.linalg.norm(ray_world)
        rays.append(ray_world)
    if len(rays) < 2:
        return 0.0
    max_angle = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            dot = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            max_angle = max(max_angle, float(np.degrees(np.arccos(dot))))
    return max_angle


def triangulate_observations(observations: Sequence[tuple[ColmapCamera, ColmapImage, float, float]]) -> np.ndarray | None:
    rows = []
    for camera, image, u, v in observations:
        xn, yn = camera_normalized_from_pixel(camera, u, v)
        rt = np.hstack([image.rotmat, image.tvec.reshape(3, 1)])
        rows.append(xn * rt[2] - rt[0])
        rows.append(yn * rt[2] - rt[1])
    if len(rows) < 4:
        return None
    a = np.asarray(rows, dtype=np.float64)
    scale = np.linalg.norm(a, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    a = a / scale
    try:
        _, _, vt = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1]
    if abs(float(h[3])) < 1e-12:
        return None
    xyz = h[:3] / h[3]
    if not np.isfinite(xyz).all():
        return None
    return xyz


def estimate_flow_velocity_for_points(
    points: Sequence[ColmapPoint3D],
    cameras: dict[int, ColmapCamera],
    images: dict[int, ColmapImage],
    flow_dir: Path,
    args,
) -> VelocityResult:
    n = len(points)
    velocities = np.zeros((n, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    confidence = np.zeros(n, dtype=np.float32)
    view_counts = np.zeros(n, dtype=np.uint8)
    flows = load_flow_maps(flow_dir, [img.name for img in images.values()])
    if not flows:
        LOGGER.info("velocity: no flow files found in %s", flow_dir)
        return VelocityResult(velocities, valid, confidence, view_counts)

    gate_schedule = [
        args.velocity_max_reproj_error * 4.0,
        args.velocity_max_reproj_error * 2.0,
        args.velocity_max_reproj_error,
    ]
    for idx, point in enumerate(points):
        observations: list[tuple[ColmapCamera, ColmapImage, float, float]] = []
        track_view_count = 0
        for image_id, point2d_idx in point.track:
            image = images.get(image_id)
            if image is None or point2d_idx < 0 or point2d_idx >= len(image.xys):
                continue
            camera = cameras[image.camera_id]
            flow = flows.get(image.name)
            if flow is None:
                continue
            track_view_count += 1
            u, v = image.xys[point2d_idx]
            if not (0 <= u < camera.width and 0 <= v < camera.height):
                continue
            flow_h, flow_w = flow.shape[:2]
            sample = bilinear_sample_flow(flow, u * flow_w / camera.width, v * flow_h / camera.height)
            if sample is None:
                continue
            du, dv = flow_to_pixel_delta(
                sample[0],
                sample[1],
                flow_w,
                flow_h,
                camera.width,
                camera.height,
                args.flow_format,
                args.flow_scale,
            )
            up = float(u + du)
            vp = float(v + dv)
            if 0 <= up < camera.width and 0 <= vp < camera.height:
                observations.append((camera, image, up, vp))

        if len(observations) < args.velocity_min_views:
            continue

        active = observations
        xyz_next = triangulate_observations(active)
        if xyz_next is None:
            continue
        for gate in gate_schedule:
            gated = []
            residuals = []
            for obs in active:
                camera, image, up, vp = obs
                pu, pv, depth = camera_project(camera, image, xyz_next)
                resid = float(np.hypot(pu - up, pv - vp)) if np.isfinite(pu) and np.isfinite(pv) else np.inf
                if depth > 1e-8 and resid <= gate:
                    gated.append(obs)
                    residuals.append(resid)
            if len(gated) < args.velocity_min_views:
                active = []
                break
            active = gated
            xyz_refined = triangulate_observations(active)
            if xyz_refined is None:
                active = []
                break
            xyz_next = xyz_refined
        if len(active) < args.velocity_min_views:
            continue

        final_residuals = []
        positive_depths = 0
        for camera, image, up, vp in active:
            pu, pv, depth = camera_project(camera, image, xyz_next)
            if depth > 1e-8:
                positive_depths += 1
            final_residuals.append(float(np.hypot(pu - up, pv - vp)) if np.isfinite(pu) and np.isfinite(pv) else np.inf)
        if positive_depths < args.velocity_min_views:
            continue
        median_resid = float(np.median(final_residuals))
        if median_resid > args.velocity_max_reproj_error:
            continue
        angle = triangulation_angle_deg(active)
        if angle < args.velocity_min_angle:
            continue

        velocity = (xyz_next - point.xyz) / args.velocity_dt
        speed = float(np.linalg.norm(velocity))
        if args.velocity_max_magnitude > 0 and speed > args.velocity_max_magnitude:
            continue

        velocities[idx] = velocity.astype(np.float32)
        valid[idx] = True
        view_counts[idx] = min(len(active), 255)
        track_ratio = len(active) / max(len(point.track), 1)
        reproj_score = np.exp(-median_resid / max(args.velocity_max_reproj_error, 1e-6))
        angle_score = min(angle / max(args.velocity_min_angle * 4.0, 1e-6), 1.0)
        confidence[idx] = float(np.clip(track_ratio * reproj_score * angle_score, 0.0, 1.0))

    LOGGER.info("velocity: %d/%d points passed multi-view flow triangulation", int(valid.sum()), n)
    return VelocityResult(velocities, valid, confidence, view_counts)


def filter_points(
    points: Sequence[ColmapPoint3D],
    min_track_len: int,
    max_reproj_error: float,
) -> list[ColmapPoint3D]:
    """Drop noisy sparse points: short tracks and high-reprojection-error points.

    A value of 0 disables the corresponding criterion.
    """
    if min_track_len <= 0 and max_reproj_error <= 0:
        return list(points)
    kept: list[ColmapPoint3D] = []
    for point in points:
        if min_track_len > 0 and len(point.track) < min_track_len:
            continue
        if max_reproj_error > 0 and point.error > max_reproj_error:
            continue
        kept.append(point)
    return kept


def write_points_ply(
    ply_path: Path,
    points: Sequence[ColmapPoint3D],
    velocity: VelocityResult | None,
) -> None:
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    if velocity is None:
        velocity = VelocityResult(
            velocities=np.zeros((n, 3), dtype=np.float32),
            valid=np.zeros(n, dtype=bool),
            confidence=np.zeros(n, dtype=np.float32),
            view_counts=np.zeros(n, dtype=np.uint8),
        )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("vx", "<f4"),
            ("vy", "<f4"),
            ("vz", "<f4"),
            ("velocity_confidence", "<f4"),
            ("velocity_valid", "u1"),
            ("velocity_views", "u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    if n:
        xyz = np.asarray([p.xyz for p in points], dtype=np.float32)
        rgb = np.asarray([p.rgb for p in points], dtype=np.uint8)
        data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        data["vx"], data["vy"], data["vz"] = velocity.velocities[:, 0], velocity.velocities[:, 1], velocity.velocities[:, 2]
        data["velocity_confidence"] = velocity.confidence.astype(np.float32, copy=False)
        data["velocity_valid"] = velocity.valid.astype(np.uint8)
        data["velocity_views"] = velocity.view_counts.astype(np.uint8, copy=False)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment velocity is estimated by multi-view optical-flow triangulation\n"
        "comment vx vy vz are in COLMAP world units per velocity_dt\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float vx\n"
        "property float vy\n"
        "property float vz\n"
        "property float velocity_confidence\n"
        "property uchar velocity_valid\n"
        "property uchar velocity_views\n"
        "end_header\n"
    )
    with ply_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def copy_sparse_model(model_dir: Path, output_root: Path, output_name: str, force: bool) -> Path:
    final_dir = output_root / "sparse" / output_name / model_dir.name
    if final_dir.exists():
        if force:
            shutil.rmtree(final_dir)
        else:
            raise FileExistsError(f"sparse output already exists: {final_dir}. Use --force to overwrite.")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, final_dir)
    return final_dir


@dataclass(frozen=True)
class RigPose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int


@dataclass(frozen=True)
class FramePaths:
    output_name: str
    frame_out: Path
    log_dir: Path
    image_dir: Path
    database_path: Path
    sparse_dir: Path
    model_txt_dir: Path
    output_ply: Path
    dense_dir: Path
    dense_ply: Path


def make_frame_paths(frame_index: int, args) -> FramePaths:
    output_name = f"frame_{frame_index:06d}"
    frame_out = args.output_root / "_workspace" / output_name
    return FramePaths(
        output_name=output_name,
        frame_out=frame_out,
        log_dir=frame_out / "logs",
        image_dir=frame_out / "images",
        database_path=frame_out / "database.db",
        sparse_dir=frame_out / "sparse",
        model_txt_dir=frame_out / "model_txt",
        output_ply=args.output_root / "points_cloud" / f"{output_name}.ply",
        dense_dir=frame_out / "dense",
        dense_ply=args.output_root / "dense_points_cloud" / f"{output_name}.ply",
    )


def prepare_frame_workspace(paths: FramePaths, args) -> None:
    for folder in (paths.frame_out, paths.log_dir, paths.output_ply.parent):
        folder.mkdir(parents=True, exist_ok=True)
    if args.force and paths.frame_out.exists():
        for child in (paths.image_dir, paths.sparse_dir, paths.model_txt_dir, paths.dense_dir):
            if child.exists():
                shutil.rmtree(child)
        if paths.database_path.exists():
            paths.database_path.unlink()
        final_sparse_root = args.output_root / "sparse" / paths.output_name
        if final_sparse_root.exists():
            shutil.rmtree(final_sparse_root)
        if paths.output_ply.exists():
            paths.output_ply.unlink()


def make_matcher(args) -> SuperGlueMatcher:
    return SuperGlueMatcher(
        superglue_root=args.superglue_root,
        weights=args.superglue,
        resize=args.resize,
        resize_float=args.resize_float,
        max_keypoints=args.max_keypoints,
        keypoint_threshold=args.keypoint_threshold,
        nms_radius=args.nms_radius,
        sinkhorn_iterations=args.sinkhorn_iterations,
        match_threshold=args.match_threshold,
        device=args.device,
    )


def match_frame_pairs(
    image_names: Sequence[str],
    image_infos: dict[str, ImageInfo],
    matcher: SuperGlueMatcher,
    args,
    output_name: str,
) -> tuple[list[PairMatch], dict]:
    pairs = build_pairs(image_names, args.pair_mode, args.pair_window, args.loop_pairs, args.pairs_file)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    LOGGER.info("%s: %d image pairs", output_name, len(pairs))

    pair_matches: list[PairMatch] = []
    raw_total = 0
    kept_total = 0
    skipped_pairs = 0
    for idx, (name0, name1) in enumerate(pairs, start=1):
        if idx == 1 or idx % args.log_every == 0:
            LOGGER.info("%s: matching pair %d/%d (%s, %s)", output_name, idx, len(pairs), name0, name1)
        raw = matcher.match_pair(image_infos[name0].path, image_infos[name1].path)
        raw_total += len(raw)
        kept = filter_corrs(raw, image_infos[name0], image_infos[name1], args.min_matches, args.ransac, args.ransac_max_error, args.ransac_confidence)
        kept_total += len(kept)
        if len(kept) >= args.min_matches:
            pair_matches.append(PairMatch(name0=name0, name1=name1, matches=kept))
        else:
            skipped_pairs += 1

    match_stats = {
        "num_candidate_pairs": len(pairs),
        "num_valid_pairs": len(pair_matches),
        "num_skipped_pairs": skipped_pairs,
        "raw_matches": raw_total,
        "filtered_matches": kept_total,
    }
    return pair_matches, match_stats


def finalize_frame(
    model_dir: Path,
    paths: FramePaths,
    frame_dir: Path,
    args,
    colmap: str,
    extra_stats: dict,
    num_images: int,
) -> tuple[dict, dict[int, ColmapCamera], dict[int, ColmapImage]]:
    export_model_txt(colmap, model_dir, paths.model_txt_dir, paths.log_dir)
    cameras, images, points = read_colmap_text_model(paths.model_txt_dir)
    raw_num_points = len(points)
    points = filter_points(points, args.ply_min_track_len, args.ply_max_reproj_error)
    LOGGER.info("%s: %d/%d points kept after track/error filtering", paths.output_name, len(points), raw_num_points)

    velocity_result = None
    if args.compute_velocity:
        flow_dir = args.flows_root / frame_dir.name
        if not flow_dir.exists():
            flow_dir = args.flows_root / paths.output_name
        if not flow_dir.exists():
            flow_dir = args.flows_root
        velocity_result = estimate_flow_velocity_for_points(points, cameras, images, flow_dir, args)
    write_points_ply(paths.output_ply, points, velocity_result)
    final_model_dir = copy_sparse_model(model_dir, args.output_root, paths.output_name, args.force)

    if args.dense:
        paths.dense_ply.parent.mkdir(parents=True, exist_ok=True)
        run_dense(colmap, paths.image_dir, model_dir, paths.dense_dir, paths.dense_ply, paths.log_dir, args)

    stats = {
        "frame": frame_dir.name,
        "output_name": paths.output_name,
        "sparse_model_dir": str(final_model_dir),
        "sparse_ply": str(paths.output_ply),
        "dense_ply": str(paths.dense_ply) if args.dense else None,
        "num_images": num_images,
        **extra_stats,
        **parse_points3d_stats(paths.model_txt_dir / "points3D.txt"),
        "num_filtered_out_points": raw_num_points - len(points),
        "num_ply_points": len(points),
        "velocity_valid_points": int(velocity_result.valid.sum()) if velocity_result is not None else 0,
    }
    if args.keep_workspace:
        (paths.frame_out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        shutil.rmtree(paths.frame_out, ignore_errors=True)
        try:
            paths.frame_out.parent.rmdir()
        except OSError:
            pass
    return stats, cameras, images


def reconstruct_frame(
    frame_dir: Path,
    frame_index: int,
    args,
    colmap: str,
    matcher: SuperGlueMatcher,
) -> tuple[dict, dict[int, ColmapCamera], dict[int, ColmapImage]]:
    paths = make_frame_paths(frame_index, args)
    prepare_frame_workspace(paths, args)

    src_images = list_images(frame_dir, args.max_images)
    staged_images = stage_images(src_images, paths.image_dir, copy_images=args.copy_images)
    LOGGER.info("%s: %d images", paths.output_name, len(staged_images))

    reset_database(colmap, paths.database_path, paths.log_dir)
    image_infos = create_colmap_database(paths.database_path, staged_images, args.camera_model, args.focal_factor, args.single_camera)

    image_names = [p.name for p in staged_images]
    pair_matches, match_stats = match_frame_pairs(image_names, image_infos, matcher, args, paths.output_name)
    if not pair_matches:
        raise RuntimeError(f"{paths.output_name}: SuperGlue produced no valid pairs")

    db_stats = export_matches_to_database(paths.database_path, image_infos, pair_matches, args.keypoint_quantization, args.two_view_config)
    LOGGER.info("%s: exported %s", paths.output_name, db_stats)

    model_dir = run_mapper(colmap, paths.database_path, paths.image_dir, paths.sparse_dir, paths.log_dir, args)
    extra_stats = {**match_stats, "database_export": db_stats, "mode": "incremental_sfm"}
    return finalize_frame(model_dir, paths, frame_dir, args, colmap, extra_stats, len(staged_images))


def create_rig_database(
    database_path: Path,
    staged_images: Sequence[Path],
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
) -> dict[str, ImageInfo]:
    """Create a database whose camera/image ids match the reference reconstruction.

    Only images that were registered in the reference rig solve are inserted, with
    their reference intrinsics and explicit ids so ``point_triangulator`` can map the
    database keypoints onto the fixed poses.
    """
    con = sqlite3.connect(database_path)
    try:
        image_infos: dict[str, ImageInfo] = {}
        inserted_cameras: set[int] = set()
        for src in staged_images:
            name = src.name
            pose = ref_pose_by_name.get(name)
            if pose is None:
                continue
            cam = ref_cameras[pose.camera_id]
            if pose.camera_id not in inserted_cameras:
                con.execute(
                    "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
                    (pose.camera_id, CAMERA_MODEL_IDS[cam.model], cam.width, cam.height, array_to_blob(np.asarray(cam.params, dtype=np.float64)), 1),
                )
                inserted_cameras.add(pose.camera_id)
            con.execute("INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)", (pose.image_id, name, pose.camera_id))
            image_infos[name] = ImageInfo(name, src, cam.width, cam.height, pose.image_id, pose.camera_id)
        con.commit()
        return image_infos
    finally:
        con.close()


def write_skeleton_model(
    skeleton_dir: Path,
    image_infos: dict[str, ImageInfo],
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
) -> None:
    """Write a COLMAP text model with known poses but no 3D points for triangulation."""
    if skeleton_dir.exists():
        shutil.rmtree(skeleton_dir)
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    used_camera_ids = sorted({info.camera_id for info in image_infos.values()})
    cam_lines = ["# Camera list", f"# Number of cameras: {len(used_camera_ids)}"]
    for cid in used_camera_ids:
        cam = ref_cameras[cid]
        params = " ".join(repr(float(x)) for x in cam.params)
        cam_lines.append(f"{cid} {cam.model} {int(cam.width)} {int(cam.height)} {params}")
    (skeleton_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")

    img_lines = ["# Image list with two lines of data per image"]
    for name, info in sorted(image_infos.items(), key=lambda kv: kv[1].image_id):
        pose = ref_pose_by_name[name]
        qstr = " ".join(repr(float(x)) for x in pose.qvec)
        tstr = " ".join(repr(float(x)) for x in pose.tvec)
        img_lines.append(f"{pose.image_id} {qstr} {tstr} {pose.camera_id} {name}")
        img_lines.append("")  # empty POINTS2D line (filled in by point_triangulator)
    (skeleton_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    (skeleton_dir / "points3D.txt").write_text("# 3D point list\n", encoding="utf-8")


def run_point_triangulator(
    colmap: str,
    database_path: Path,
    image_dir: Path,
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    args,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap,
        "point_triangulator",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--input_path",
        str(input_dir),
        "--output_path",
        str(output_dir),
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
        "--Mapper.tri_min_angle",
        str(args.mapper_tri_min_angle),
        "--Mapper.min_num_matches",
        str(args.mapper_min_num_matches),
    ]
    run_command(cmd, log_dir / "point_triangulator.log")
    if not (output_dir / "points3D.bin").exists() and not (output_dir / "points3D.txt").exists():
        raise RuntimeError(f"point_triangulator produced no model in {output_dir}")
    return output_dir


def triangulate_frame_with_rig(
    frame_dir: Path,
    frame_index: int,
    args,
    colmap: str,
    matcher: SuperGlueMatcher,
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
) -> tuple[dict, dict[int, ColmapCamera], dict[int, ColmapImage]]:
    paths = make_frame_paths(frame_index, args)
    prepare_frame_workspace(paths, args)

    src_images = list_images(frame_dir, args.max_images)
    staged_images = stage_images(src_images, paths.image_dir, copy_images=args.copy_images)

    reset_database(colmap, paths.database_path, paths.log_dir)
    image_infos = create_rig_database(paths.database_path, staged_images, ref_cameras, ref_pose_by_name)
    if len(image_infos) < 2:
        raise RuntimeError(f"{paths.output_name}: fewer than 2 rig cameras available for triangulation")
    LOGGER.info("%s: %d rig cameras with known pose", paths.output_name, len(image_infos))

    image_names = [p.name for p in staged_images if p.name in image_infos]
    pair_matches, match_stats = match_frame_pairs(image_names, image_infos, matcher, args, paths.output_name)
    if not pair_matches:
        raise RuntimeError(f"{paths.output_name}: SuperGlue produced no valid pairs")

    db_stats = export_matches_to_database(paths.database_path, image_infos, pair_matches, args.keypoint_quantization, args.two_view_config)
    LOGGER.info("%s: exported %s", paths.output_name, db_stats)

    skeleton_dir = paths.frame_out / "rig_input"
    write_skeleton_model(skeleton_dir, image_infos, ref_cameras, ref_pose_by_name)
    model_dir = run_point_triangulator(colmap, paths.database_path, paths.image_dir, skeleton_dir, paths.sparse_dir / "0", paths.log_dir, args)
    extra_stats = {**match_stats, "database_export": db_stats, "mode": "rig_triangulation"}
    return finalize_frame(model_dir, paths, frame_dir, args, colmap, extra_stats, len(image_infos))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replace COLMAP matcher with SuperPoint+SuperGlue matches.")
    parser.add_argument("--images_root", type=Path, default=Path("data/twopeople/images"))
    parser.add_argument("--output_root", type=Path, default=Path("output/twopeople_superglue"))
    parser.add_argument("--superglue_root", type=Path, default=Path("colmap/SuperGluePretrainedNetwork"))
    parser.add_argument("--frames", type=str, default=None, help="Frame folder names or ranges, e.g. '1' or '1:3'.")
    parser.add_argument("--max_images", type=int, default=None, help="Debug limit for images per frame.")
    parser.add_argument("--max_pairs", type=int, default=None, help="Debug limit for image pairs per frame.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing frame workspace.")
    parser.add_argument("--keep_workspace", action="store_true", help="Keep temporary images, database, logs, and text model.")
    parser.add_argument("--copy_images", action="store_true", help="Copy images instead of using hard links when possible.")

    parser.add_argument("--static_rig", action=argparse.BooleanOptionalAction, default=False,
                        help="Solve camera poses once on a reference frame, then reuse them to triangulate every "
                             "frame (point_triangulator). Use when the cameras do not move between frames.")
    parser.add_argument("--rig_ref_frame", type=str, default=None,
                        help="Frame folder name used for the reference pose solve in --static_rig "
                             "(default: first selected frame).")

    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--camera_model", choices=sorted(CAMERA_MODEL_IDS), default="OPENCV")
    parser.add_argument("--focal_factor", type=float, default=1.2)
    parser.add_argument("--single_camera", action="store_true")

    parser.add_argument("--pair_mode", choices=["sequential", "exhaustive", "pairs_file"], default="exhaustive")
    parser.add_argument("--pair_window", type=int, default=5)
    parser.add_argument("--loop_pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pairs_file", type=Path, default=None)

    parser.add_argument("--superglue", choices=["indoor", "outdoor"], default="outdoor")
    parser.add_argument("--resize", type=int, nargs="+", default=[2560], help="SuperGlue input resize: max_dim, width height, or -1 for full resolution.")
    parser.add_argument("--resize_float", action="store_true")
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    parser.add_argument("--compute_velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flows_root", type=Path, default=Path("data/twopeople/flows"))
    parser.add_argument("--flow_format", choices=["norm", "midnorm", "pixel"], default="norm")
    parser.add_argument("--flow_scale", type=float, default=1.0)
    parser.add_argument("--velocity_dt", type=float, default=1.0)
    parser.add_argument("--velocity_min_views", type=int, default=3)
    parser.add_argument("--velocity_max_reproj_error", type=float, default=3.0)
    parser.add_argument("--velocity_min_angle", type=float, default=1.0)
    parser.add_argument("--velocity_max_magnitude", type=float, default=0.0, help="0 disables speed clipping.")

    parser.add_argument("--min_matches", type=int, default=30)
    parser.add_argument("--keypoint_quantization", type=float, default=0.25)
    parser.add_argument("--two_view_config", type=int, default=3)
    parser.add_argument("--ransac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ransac_max_error", type=float, default=1.5)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)

    parser.add_argument("--ply_min_track_len", type=int, default=3,
                        help="Drop sparse points seen by fewer than this many images (0 disables).")
    parser.add_argument("--ply_max_reproj_error", type=float, default=2.0,
                        help="Drop sparse points whose reprojection error exceeds this (0 disables).")

    parser.add_argument("--ba_refine_focal_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ba_refine_principal_point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mapper_min_num_matches", type=int, default=15)
    parser.add_argument("--mapper_init_min_num_inliers", type=int, default=30)
    parser.add_argument("--mapper_init_max_error", type=float, default=4.0)
    parser.add_argument("--mapper_abs_pose_min_num_inliers", type=int, default=20)
    parser.add_argument("--mapper_tri_min_angle", type=float, default=2.0)
    parser.add_argument("--mapper_multiple_models", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--dense", action="store_true", help="Run image_undistorter, patch_match_stereo, stereo_fusion.")
    parser.add_argument("--gpu_index", default="0")
    parser.add_argument("--fusion_check_num_images", type=int, default=2)
    parser.add_argument("--fusion_min_num_pixels", type=int, default=3)

    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    args.images_root = args.images_root.resolve()
    args.output_root = args.output_root.resolve()
    args.superglue_root = args.superglue_root.resolve()
    args.flows_root = args.flows_root.resolve()
    if args.velocity_dt <= 0:
        raise ValueError("--velocity_dt must be positive")
    if args.pairs_file is not None:
        args.pairs_file = args.pairs_file.resolve()

    colmap = resolve_colmap(args.colmap)
    frame_dirs = discover_frame_dirs(args.images_root)
    selected_frames = parse_frame_selection(frame_dirs, args.frames)
    args.output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("selected frames: %s", ", ".join(p.name for p in selected_frames))
    matcher = make_matcher(args)
    all_stats = []
    name_to_index = {p.name: idx + 1 for idx, p in enumerate(frame_dirs)}

    if args.static_rig:
        by_name = {p.name: p for p in frame_dirs}
        ref_name = args.rig_ref_frame or selected_frames[0].name
        if ref_name not in by_name:
            raise ValueError(f"--rig_ref_frame '{ref_name}' not found among frame folders")
        ref_frame = by_name[ref_name]
        LOGGER.info("static rig: solving reference poses from frame '%s'", ref_name)
        ref_stats, ref_cameras, ref_images = reconstruct_frame(ref_frame, name_to_index[ref_name], args, colmap, matcher)
        all_stats.append(ref_stats)
        ref_pose_by_name = {
            img.name: RigPose(img.image_id, img.qvec, img.tvec, img.camera_id)
            for img in ref_images.values()
        }
        LOGGER.info("static rig: reference registered %d/%d cameras", len(ref_pose_by_name), ref_stats["num_images"])
        for frame_dir in selected_frames:
            if frame_dir.name == ref_name:
                continue
            stats, _, _ = triangulate_frame_with_rig(
                frame_dir, name_to_index[frame_dir.name], args, colmap, matcher, ref_cameras, ref_pose_by_name
            )
            all_stats.append(stats)
    else:
        for frame_dir in selected_frames:
            stats, _, _ = reconstruct_frame(frame_dir, name_to_index[frame_dir.name], args, colmap, matcher)
            all_stats.append(stats)

    for stats in all_stats:
        LOGGER.info(
            "%s: sparse=%s ply=%s velocity=%d/%d",
            stats["output_name"],
            stats["sparse_model_dir"],
            stats["sparse_ply"],
            stats["velocity_valid_points"],
            stats["num_sparse_points"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
