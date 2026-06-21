"""Use A2PM-MESA matches in a COLMAP reconstruction pipeline.

The script builds a COLMAP database, replaces the matcher stage with A2PM
correspondences, then runs COLMAP mapper/model_converter. It is designed for a
fixed multi-camera matrix layout like:

  data/twopeople/images/<frame_dir>/<camera_image>.png

Run from the repository root, preferably inside the A2PM conda environment:

  conda run -n A2PM-new python colmap/a2pm_colmap.py \
      --images_root data/twopeople/images \
      --output_root output/twopeople_a2pm \
      --frames 1 \
      --point_matcher mast3r
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
from typing import Iterable, Sequence

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


LOGGER = logging.getLogger("a2pm_colmap")


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
    matches: np.ndarray  # Nx4, original-image coordinates: x0, y0, x1, y1


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
            start = int(start_s)
            end = int(end_s)
            for value in range(start, end + 1):
                key = str(value)
                if key in by_name:
                    selected.append(by_name[key])
        elif token in by_name:
            selected.append(by_name[token])
        elif token.isdigit():
            idx = int(token)
            # Prefer exact frame folder name. If it is absent, treat as 1-based index.
            if 1 <= idx <= len(frame_dirs):
                selected.append(frame_dirs[idx - 1])
            else:
                raise ValueError(f"frame '{token}' not found")
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
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError
        return int(img.shape[1]), int(img.shape[0])
    except Exception:
        try:
            from PIL import Image

            with Image.open(path) as img:
                return int(img.width), int(img.height)
        except Exception as exc:
            raise RuntimeError(f"failed to read image size: {path}") from exc


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
    run_command(
        [colmap, "database_creator", "--database_path", str(database_path)],
        log_dir / "database_creator.log",
    )


def create_colmap_database(
    database_path: Path,
    images: Sequence[Path],
    camera_model: str,
    focal_factor: float,
    single_camera: bool,
) -> dict[str, ImageInfo]:
    if camera_model not in CAMERA_MODEL_IDS:
        raise ValueError(f"unsupported --camera_model {camera_model}")

    con = sqlite3.connect(database_path)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        model_id = CAMERA_MODEL_IDS[camera_model]
        image_infos: dict[str, ImageInfo] = {}
        shared_camera_id: int | None = None
        shared_size: tuple[int, int] | None = None

        for idx, image_path in enumerate(images, start=1):
            width, height = image_size(image_path)
            if single_camera:
                if shared_camera_id is None:
                    shared_size = (width, height)
                    params = camera_params(camera_model, width, height, focal_factor)
                    cur = con.execute(
                        "INSERT INTO cameras(model, width, height, params, prior_focal_length) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (model_id, width, height, array_to_blob(params), 0),
                    )
                    shared_camera_id = int(cur.lastrowid)
                elif shared_size != (width, height):
                    raise ValueError(
                        "--single_camera requires identical image sizes; "
                        f"{image_path.name} has {(width, height)}, expected {shared_size}"
                    )
                camera_id = shared_camera_id
            else:
                params = camera_params(camera_model, width, height, focal_factor)
                cur = con.execute(
                    "INSERT INTO cameras(model, width, height, params, prior_focal_length) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (model_id, width, height, array_to_blob(params), 0),
                )
                camera_id = int(cur.lastrowid)

            cur = con.execute(
                "INSERT INTO images(name, camera_id) VALUES (?, ?)",
                (image_path.name, camera_id),
            )
            image_id = int(cur.lastrowid)
            image_infos[image_path.name] = ImageInfo(
                name=image_path.name,
                path=image_path,
                width=width,
                height=height,
                image_id=image_id,
                camera_id=camera_id,
            )
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
            max_step = min(window, n - 1)
            for step in range(1, max_step + 1):
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
            if len(parts) != 2:
                raise ValueError(f"invalid pair line: {line}")
            if parts[0] not in name_set or parts[1] not in name_set:
                raise ValueError(f"pair references missing image: {line}")
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


def load_a2pm_config(a2pm_root: Path, group: str, name: str):
    from omegaconf import OmegaConf

    path = a2pm_root / "conf" / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"A2PM config not found: {path}")
    return OmegaConf.load(path)


def patch_a2pm_checkpoint_paths(cfg, a2pm_root: Path) -> None:
    for key in ("weight_path", "mast3r_weight_path"):
        if key not in cfg:
            continue
        value = str(cfg[key])
        if not value:
            continue
        path = Path(value)
        if path.exists():
            continue
        candidate = a2pm_root / "checkpoints" / path.name
        if candidate.exists():
            LOGGER.info("patch A2PM %s: %s -> %s", key, value, candidate)
            cfg[key] = str(candidate)


def instantiate_a2pm(cfg):
    import hydra

    return hydra.utils.instantiate(cfg)


class LocalPairLoader:
    """Minimal A2PM dataloader for full A2PM area matching mode."""

    def __init__(
        self,
        abstract_base,
        root_path: str,
        image0: Path,
        image1: Path,
        sem_dir: Path,
        intrinsics_dir: Path | None,
    ) -> None:
        class _Loader(abstract_base):  # type: ignore[misc, valid-type]
            def __init__(self, outer: "LocalPairLoader") -> None:
                self.outer = outer
                super().__init__(
                    root_path=outer.root_path,
                    scene_name="colmap_pair",
                    image_name0=outer.image0.stem,
                    image_name1=outer.image1.stem,
                )
                self._name = "LocalPairLoader"
                self.img0_path = str(outer.image0)
                self.img1_path = str(outer.image1)
                self.sem0_path = str(outer.sem_dir / f"{outer.image0.stem}.npy")
                self.sem1_path = str(outer.sem_dir / f"{outer.image1.stem}.npy")
                self.K0_path = str(outer.intrinsics_dir / f"{outer.image0.stem}.txt") if outer.intrinsics_dir else ""
                self.K1_path = str(outer.intrinsics_dir / f"{outer.image1.stem}.txt") if outer.intrinsics_dir else ""

            def _path_assemble(self):
                return None

            def load_Ks(self, scale0=1.0, scale1=1.0):
                if not self.K0_path or not self.K1_path:
                    return None, None
                return outer_load_K(self.K0_path, scale0), outer_load_K(self.K1_path, scale1)

            def load_depths(self):
                return None, None

            def load_semantics(self):
                return np.load(self.sem0_path, allow_pickle=True), np.load(self.sem1_path, allow_pickle=True)

            def load_poses(self):
                return None, None

            def get_eval_info(self):
                raise NotImplementedError

            def get_sem_paths(self):
                return self.sem0_path, self.sem1_path

            def tune_corrs_size_to_eval(self, corrs, match_W, match_H, eval_W, eval_H):
                return corrs

        def outer_load_K(path: str, scale) -> np.ndarray:
            values = [float(x) for x in Path(path).read_text(encoding="utf-8").split()]
            if len(values) == 4:
                fx, fy, cx, cy = values
            elif len(values) == 9:
                fx, fy, cx, cy = values[0], values[4], values[2], values[5]
            else:
                raise ValueError(f"intrinsics file must contain fx fy cx cy or 3x3 K: {path}")
            if isinstance(scale, list):
                sx, sy = float(scale[0]), float(scale[1])
            else:
                sx = sy = float(scale)
            return np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float64)

        self.root_path = root_path
        self.image0 = image0
        self.image1 = image1
        self.sem_dir = sem_dir
        self.intrinsics_dir = intrinsics_dir
        self.loader = _Loader(self)


class A2PMMatcher:
    def __init__(
        self,
        a2pm_root: Path,
        mode: str,
        point_matcher_name: str,
        area_matcher_name: str,
        geo_matcher_name: str,
        match_width: int,
        match_height: int,
        match_num: int,
        device: str,
        sem_root: Path | None,
        intrinsics_root: Path | None,
        debug_root: Path,
    ) -> None:
        self.a2pm_root = a2pm_root.resolve()
        self.mode = mode
        self.match_width = match_width
        self.match_height = match_height
        self.match_num = match_num
        self.sem_root = sem_root
        self.intrinsics_root = intrinsics_root
        self.debug_root = debug_root

        if not self.a2pm_root.exists():
            raise FileNotFoundError(f"A2PM root does not exist: {self.a2pm_root}")
        sys.path.insert(0, str(self.a2pm_root))

        self.point_cfg = load_a2pm_config(self.a2pm_root, "point_matcher", point_matcher_name)
        patch_a2pm_checkpoint_paths(self.point_cfg, self.a2pm_root)
        if "device" in self.point_cfg:
            self.point_cfg.device = device
        self.point_matcher = instantiate_a2pm(self.point_cfg)
        self.point_matcher.set_corr_num_init(match_num)

        self.area_matcher = None
        self.geo_matcher = None
        self.abstract_loader_base = None
        if mode == "full":
            if sem_root is None:
                raise ValueError("--sem_root is required for --a2pm_mode full")
            area_cfg = load_a2pm_config(self.a2pm_root, "area_matcher", area_matcher_name)
            geo_cfg = load_a2pm_config(self.a2pm_root, "geo_area_matcher", geo_matcher_name)
            patch_a2pm_checkpoint_paths(area_cfg, self.a2pm_root)
            self._patch_full_cfg(area_cfg, geo_cfg)
            self.area_matcher = instantiate_a2pm(area_cfg)
            self.geo_matcher = instantiate_a2pm(geo_cfg)
            from dataloader.abstract_dataloader import AbstractDataloader

            self.abstract_loader_base = AbstractDataloader

    def _patch_full_cfg(self, area_cfg, geo_cfg) -> None:
        # Treat arbitrary user data as "demo" so A2PM keeps original image sizes
        # internally instead of assuming ScanNet resize conventions.
        if "datasetName" in area_cfg:
            area_cfg.datasetName = "MegaDepth"
        if "W" in area_cfg:
            area_cfg.W = min(self.match_width, 1024)
        if "H" in area_cfg:
            area_cfg.H = min(self.match_height, 1024)
        if "datasetName" in geo_cfg:
            geo_cfg.datasetName = "demo"
        for key in ("crop_size_W", "crop_size_H", "std_match_num"):
            if key == "crop_size_W" and key in geo_cfg:
                geo_cfg[key] = self.match_width
            elif key == "crop_size_H" and key in geo_cfg:
                geo_cfg[key] = self.match_height
            elif key == "std_match_num" and key in geo_cfg:
                geo_cfg[key] = self.match_num
        if "valid_inside_area_match_num" in geo_cfg:
            geo_cfg.valid_inside_area_match_num = min(int(geo_cfg.valid_inside_area_match_num), 20)
        if "verbose" in geo_cfg:
            geo_cfg.verbose = 0
        if "draw_verbose" in area_cfg:
            area_cfg.draw_verbose = 0

    def match_pair(self, image0: Path, image1: Path, pair_debug_dir: Path) -> np.ndarray:
        if self.mode == "full":
            return self._match_pair_full(image0, image1, pair_debug_dir)
        return self._match_pair_point(image0, image1)

    def _read_and_resize(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image: {path}")
        orig_h, orig_w = img.shape[:2]
        resized = cv2.resize(img, (self.match_width, self.match_height), interpolation=cv2.INTER_AREA)
        return resized, (orig_w, orig_h)

    def _scale_corrs_to_original(
        self,
        corrs: Sequence[Sequence[float]],
        size0: tuple[int, int],
        size1: tuple[int, int],
    ) -> np.ndarray:
        if len(corrs) == 0:
            return np.empty((0, 4), dtype=np.float32)
        arr = np.asarray(corrs, dtype=np.float32).reshape(-1, 4)
        w0, h0 = size0
        w1, h1 = size1
        arr[:, 0] *= w0 / self.match_width
        arr[:, 1] *= h0 / self.match_height
        arr[:, 2] *= w1 / self.match_width
        arr[:, 3] *= h1 / self.match_height
        return arr

    def _match_pair_point(self, image0: Path, image1: Path) -> np.ndarray:
        img0, size0 = self._read_and_resize(image0)
        img1, size1 = self._read_and_resize(image1)
        self.point_matcher.set_corr_num_init(self.match_num)
        corrs = self.point_matcher.match(img0, img1)
        return self._scale_corrs_to_original(corrs, size0, size1)

    def _match_pair_full(self, image0: Path, image1: Path, pair_debug_dir: Path) -> np.ndarray:
        assert self.area_matcher is not None
        assert self.geo_matcher is not None
        assert self.abstract_loader_base is not None
        assert self.sem_root is not None

        pair_debug_dir.mkdir(parents=True, exist_ok=True)
        sem_dir = self.sem_root
        intrinsics_dir = self.intrinsics_root if self.intrinsics_root and self.intrinsics_root.exists() else None
        loader = LocalPairLoader(
            self.abstract_loader_base,
            root_path=str(image0.parent),
            image0=image0,
            image1=image1,
            sem_dir=sem_dir,
            intrinsics_dir=intrinsics_dir,
        ).loader

        # Initial whole-image matches are still produced at match_width/height.
        initial_corrs = self._match_pair_point(image0, image1).tolist()
        area0, area1 = self.area_matcher.area_matching(loader, str(pair_debug_dir / "area"))
        self.geo_matcher.init_gam(loader, self.point_matcher, initial_corrs, str(pair_debug_dir / "geo"))
        alpha_corrs_dict, _, sampled = self.geo_matcher.geo_area_matching_refine(area0, area1)
        if sampled:
            alpha_corrs_dict = sampled
        if not alpha_corrs_dict:
            return np.asarray(initial_corrs, dtype=np.float32)
        best_alpha = max(alpha_corrs_dict.keys(), key=lambda k: sum(len(x) for x in alpha_corrs_dict[k]))
        corrs = flatten_corrs(alpha_corrs_dict[best_alpha])
        if not corrs:
            return np.asarray(initial_corrs, dtype=np.float32)
        return np.asarray(corrs, dtype=np.float32).reshape(-1, 4)


def flatten_corrs(value) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    if not value:
        return []
    if len(value[0]) == 4 and all(isinstance(x, (int, float, np.number)) for x in value[0]):
        return value
    out: list[list[float]] = []
    for item in value:
        out.extend(flatten_corrs(item))
    return out


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

    # Remove exact/near-exact duplicate rows before RANSAC.
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
            unique = np.unique(np.asarray(idx_matches, dtype=np.uint32), axis=0)
            indexed_pairs.append((pair.name0, pair.name1, unique))

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
            con.execute(
                "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, rows, 2, blob),
            )
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


def run_mapper(colmap: str, database_path: Path, image_dir: Path, sparse_dir: Path, log_dir: Path, args) -> Path:
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
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
    run_command(mapper_cmd, log_dir / "mapper.log")
    models = sorted([p for p in sparse_dir.iterdir() if p.is_dir()], key=natural_key)
    if not models:
        raise RuntimeError(f"COLMAP mapper produced no model in {sparse_dir}")
    return models[0]


def export_model(colmap: str, model_dir: Path, txt_dir: Path, ply_path: Path, log_dir: Path) -> None:
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(txt_dir),
            "--output_type",
            "TXT",
        ],
        log_dir / "model_converter_txt.log",
    )
    run_command(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(ply_path),
            "--output_type",
            "PLY",
        ],
        log_dir / "model_converter_ply.log",
    )


def run_dense(colmap: str, image_dir: Path, model_dir: Path, dense_dir: Path, fused_ply: Path, log_dir: Path, args) -> None:
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(model_dir),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ],
        log_dir / "image_undistorter.log",
    )
    (dense_dir / "stereo" / "depth_maps").mkdir(parents=True, exist_ok=True)
    (dense_dir / "stereo" / "normal_maps").mkdir(parents=True, exist_ok=True)
    run_command(
        [
            colmap,
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
            "--PatchMatchStereo.gpu_index",
            args.gpu_index,
        ],
        log_dir / "patch_match_stereo.log",
    )
    run_command(
        [
            colmap,
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(fused_ply),
            "--StereoFusion.check_num_images",
            str(args.fusion_check_num_images),
            "--StereoFusion.min_num_pixels",
            str(args.fusion_min_num_pixels),
        ],
        log_dir / "stereo_fusion.log",
    )


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
    if errors:
        return {
            "num_sparse_points": count,
            "mean_reprojection_error": float(np.mean(errors)),
            "median_reprojection_error": float(np.median(errors)),
        }
    return {"num_sparse_points": count, "mean_reprojection_error": None, "median_reprojection_error": None}


def reconstruct_frame(frame_dir: Path, frame_index: int, args, colmap: str) -> dict:
    output_name = f"frame_{frame_index:06d}"
    frame_out = args.output_root / "debug" / "scenes" / "frames" / output_name
    log_dir = args.output_root / "debug" / "logs" / output_name
    stats_dir = args.output_root / "debug" / "stats"
    image_dir = frame_out / "images"
    database_path = frame_out / "database.db"
    sparse_dir = frame_out / "sparse"
    model_txt_dir = frame_out / "model_txt"
    output_ply = args.output_root / "points_cloud" / f"{output_name}.ply"
    dense_dir = frame_out / "dense"
    dense_ply = args.output_root / "dense_points_cloud" / f"{output_name}.ply"

    for folder in (frame_out, log_dir, stats_dir, output_ply.parent):
        folder.mkdir(parents=True, exist_ok=True)
    if args.force and frame_out.exists():
        for child in (image_dir, sparse_dir, model_txt_dir, dense_dir):
            if child.exists():
                shutil.rmtree(child)
        if database_path.exists():
            database_path.unlink()

    src_images = list_images(frame_dir, args.max_images)
    staged_images = stage_images(src_images, image_dir, copy_images=args.copy_images)
    LOGGER.info("%s: %d images", output_name, len(staged_images))

    reset_database(colmap, database_path, log_dir)
    image_infos = create_colmap_database(
        database_path,
        staged_images,
        args.camera_model,
        args.focal_factor,
        args.single_camera,
    )

    image_names = [p.name for p in staged_images]
    pairs = build_pairs(image_names, args.pair_mode, args.pair_window, args.loop_pairs, args.pairs_file)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    LOGGER.info("%s: %d image pairs", output_name, len(pairs))

    matcher = A2PMMatcher(
        a2pm_root=args.a2pm_root,
        mode=args.a2pm_mode,
        point_matcher_name=args.point_matcher,
        area_matcher_name=args.area_matcher,
        geo_matcher_name=args.geo_matcher,
        match_width=args.match_width,
        match_height=args.match_height,
        match_num=args.match_num,
        device=args.device,
        sem_root=args.sem_root,
        intrinsics_root=args.intrinsics_root,
        debug_root=frame_out / "a2pm_debug",
    )

    pair_matches: list[PairMatch] = []
    raw_total = 0
    kept_total = 0
    skipped_pairs = 0
    for idx, (name0, name1) in enumerate(pairs, start=1):
        if idx == 1 or idx % args.log_every == 0:
            LOGGER.info("%s: matching pair %d/%d (%s, %s)", output_name, idx, len(pairs), name0, name1)
        raw = matcher.match_pair(
            image_infos[name0].path,
            image_infos[name1].path,
            frame_out / "a2pm_debug" / f"{Path(name0).stem}_{Path(name1).stem}",
        )
        raw_total += len(raw)
        kept = filter_corrs(
            raw,
            image_infos[name0],
            image_infos[name1],
            args.min_matches,
            args.ransac,
            args.ransac_max_error,
            args.ransac_confidence,
        )
        kept_total += len(kept)
        if len(kept) >= args.min_matches:
            pair_matches.append(PairMatch(name0=name0, name1=name1, matches=kept))
        else:
            skipped_pairs += 1

    if not pair_matches:
        raise RuntimeError(f"{output_name}: A2PM produced no valid pairs")

    db_stats = export_matches_to_database(
        database_path,
        image_infos,
        pair_matches,
        args.keypoint_quantization,
        args.two_view_config,
    )
    LOGGER.info("%s: exported %s", output_name, db_stats)

    model_dir = run_mapper(colmap, database_path, image_dir, sparse_dir, log_dir, args)
    export_model(colmap, model_dir, model_txt_dir, output_ply, log_dir)
    if args.dense:
        dense_ply.parent.mkdir(parents=True, exist_ok=True)
        run_dense(colmap, image_dir, model_dir, dense_dir, dense_ply, log_dir, args)

    stats = {
        "frame": frame_dir.name,
        "output_name": output_name,
        "image_dir": str(image_dir),
        "database": str(database_path),
        "model_dir": str(model_dir),
        "model_txt_dir": str(model_txt_dir),
        "sparse_ply": str(output_ply),
        "dense_ply": str(dense_ply) if args.dense else None,
        "num_images": len(staged_images),
        "num_candidate_pairs": len(pairs),
        "num_valid_pairs": len(pair_matches),
        "num_skipped_pairs": skipped_pairs,
        "raw_matches": raw_total,
        "filtered_matches": kept_total,
        "database_export": db_stats,
        **parse_points3d_stats(model_txt_dir / "points3D.txt"),
    }
    (stats_dir / f"{output_name}.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replace COLMAP matcher with A2PM-MESA matches.")
    parser.add_argument("--images_root", type=Path, default=Path("data/twopeople/images"))
    parser.add_argument("--output_root", type=Path, default=Path("output/twopeople_a2pm"))
    parser.add_argument("--a2pm_root", type=Path, default=Path("colmap/A2PM-MESA"))
    parser.add_argument("--frames", type=str, default=None, help="Frame folder names or ranges, e.g. '1' or '1:3'.")
    parser.add_argument("--max_images", type=int, default=None, help="Debug limit for images per frame.")
    parser.add_argument("--max_pairs", type=int, default=None, help="Debug limit for image pairs per frame.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing frame workspace.")
    parser.add_argument("--copy_images", action="store_true", help="Copy images instead of using hard links when possible.")

    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--camera_model", choices=sorted(CAMERA_MODEL_IDS), default="SIMPLE_PINHOLE")
    parser.add_argument("--focal_factor", type=float, default=1.2)
    parser.add_argument("--single_camera", action="store_true")

    parser.add_argument("--pair_mode", choices=["sequential", "exhaustive", "pairs_file"], default="sequential")
    parser.add_argument("--pair_window", type=int, default=5)
    parser.add_argument("--loop_pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pairs_file", type=Path, default=None)

    parser.add_argument("--a2pm_mode", choices=["point", "full"], default="point")
    parser.add_argument("--point_matcher", default="mast3r", help="Name under A2PM-MESA/conf/point_matcher.")
    parser.add_argument("--area_matcher", default="dmesa", help="Name under A2PM-MESA/conf/area_matcher for full mode.")
    parser.add_argument("--geo_matcher", default="gam", help="Name under A2PM-MESA/conf/geo_area_matcher for full mode.")
    parser.add_argument("--sem_root", type=Path, default=None, help="SAM .npy folder for full A2PM mode.")
    parser.add_argument("--intrinsics_root", type=Path, default=None, help="Optional fx fy cx cy txt folder for full A2PM mode.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--match_width", type=int, default=512)
    parser.add_argument("--match_height", type=int, default=512)
    parser.add_argument("--match_num", type=int, default=4000)
    parser.add_argument("--min_matches", type=int, default=30)
    parser.add_argument("--keypoint_quantization", type=float, default=0.25)
    parser.add_argument("--two_view_config", type=int, default=3)
    parser.add_argument("--ransac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ransac_max_error", type=float, default=4.0)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)

    parser.add_argument("--ba_refine_focal_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ba_refine_principal_point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mapper_min_num_matches", type=int, default=15)
    parser.add_argument("--mapper_init_min_num_inliers", type=int, default=30)
    parser.add_argument("--mapper_init_max_error", type=float, default=8.0)
    parser.add_argument("--mapper_abs_pose_min_num_inliers", type=int, default=20)
    parser.add_argument("--mapper_tri_min_angle", type=float, default=1.0)
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
    args.a2pm_root = args.a2pm_root.resolve()
    if args.sem_root is not None:
        args.sem_root = args.sem_root.resolve()
    if args.intrinsics_root is not None:
        args.intrinsics_root = args.intrinsics_root.resolve()
    if args.pairs_file is not None:
        args.pairs_file = args.pairs_file.resolve()

    colmap = resolve_colmap(args.colmap)
    frame_dirs = discover_frame_dirs(args.images_root)
    selected_frames = parse_frame_selection(frame_dirs, args.frames)
    args.output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("selected frames: %s", ", ".join(p.name for p in selected_frames))
    all_stats = []
    name_to_index = {p.name: idx + 1 for idx, p in enumerate(frame_dirs)}
    for frame_dir in selected_frames:
        stats = reconstruct_frame(frame_dir, name_to_index[frame_dir.name], args, colmap)
        all_stats.append(stats)

    summary = {
        "images_root": str(args.images_root),
        "output_root": str(args.output_root),
        "a2pm_root": str(args.a2pm_root),
        "frames": all_stats,
    }
    summary_path = args.output_root / "debug" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("summary saved to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
