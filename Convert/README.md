# Convert —— 标定参数转换 / 图像去畸变

把不同来源的相机标定（**calib.io / libCalib 的 `calib.json`**、**Agisoft Metashape 的 `cameras.xml`**）转换成
**COLMAP** 能识别的参数，并对图像做**去畸变 + 裁黑边 + 主点居中**，最终产出可直接用于 **3D Gaussian Splatting (3DGS)** 训练的数据。

> 适用场景：固定多相机阵列拍摄（每个相机一段视频 / 一组照片），先标定，再把所有帧统一去畸变。

---

## 0. 数据来源约定

| 标定来源 | 文件 | 由谁解析 |
|---|---|---|
| calib.io / libCalib | `calib.json` | `undistort_from_calib.py`、`calib_to_colmap.py`、`calib_to_agisoft_cameras.py` |
| Agisoft Metashape（导出 cameras.xml） | `cameras.xml` | `batch_undistort_frames.py`、`xml_undistort_gs_params.py` |

`calib.json` 内外参约定：旋转/平移为 **W2C（world-to-camera）**，旋转用 Rodrigues 向量表示。

图像目录两种常见组织方式（由 `preprocess/batch_cut_video.py` 产出）：

- **单目录**：`images/` 下直接是 `1.png 2.png ...`（一张图对应一个相机）。
- **按帧分组（by_frame）**：`images/1/ 2/ 3/ ...`，每个子目录是同一帧的所有相机图。

---

## 1. `undistort_from_calib.py` ★主力推荐

**功能**：读 `calib.json` → 用 OpenCV/`FULL_OPENCV`（含 k3，避免鱼眼残留）去畸变 → 自动裁切使**主点严格居中** → 统一输出尺寸 → 输出 **PINHOLE 无畸变** 的 COLMAP 参数。
自动判断模式：`--images_dir` 含子目录 = **批量(by_frame)模式**（不需要 pycolmap）；否则 = **单目录模式**（需要 `pip install pycolmap`）。

**数据准备**
- `calib.json`
- 图像目录（单目录或 by_frame 子目录结构）

**命令**
```bash
# 批量（by_frame，推荐）
python Convert/undistort_from_calib.py \
  --calib data/two/calib.json \
  --images_dir data/two/images/by_frame \
  --output_dir output/two \
  --workers 16

# 单目录（需 pycolmap）
python Convert/undistort_from_calib.py --calib calib.json --images_dir images --output_dir undistorted
```

**产出**（`output_dir/`）
```
images/                 # 去畸变后图像，子目录结构与输入一致（by_frame 时为 1/ 2/ ...）
sparse/0/cameras.txt    # PINHOLE，无畸变
sparse/0/images.txt     # 各相机位姿（W2C 四元数 + 平移）
sparse/0/points3D.txt   # 空文件（占位）
```
主点位于图像正中心、相机模型 PINHOLE，可直接喂给 3DGS。

---

## 2. `batch_undistort_frames.py`

**功能**：读 **Agisoft `cameras.xml`** → 对 `images_group/` 下**每一帧文件夹**批量去畸变 → 统一裁黑边 → 主点居中 → 输出每帧图像 + 一份所有帧共享的 COLMAP 参数。多线程。

**数据准备**
- Agisoft 导出的 `cameras.xml`
- `images_group/`，结构为 `images_group/{帧号}/{相机号}.png`（如 `1/1.png 1/2.png ... 2/1.png ...`）

**命令**
```bash
python Convert/batch_undistort_frames.py \
  --xml cameras.xml \
  --images_group images_group \
  --output_dir output \
  --workers 8
# --no_unified_intrinsics（默认）每相机独立内参；--unified_intrinsics 全部统一内参
```

**产出**（`output_dir/`）
```
{帧号}/images/...       # 每帧去畸变图像
sparse/0/{cameras,images,points3D}.txt   # 所有帧共享一份
```

---

## 3. `xml_undistort_gs_params.py`

**功能**：`batch_undistort_frames.py` 的**单目录**版本。读 Agisoft `cameras.xml`，对一个图像目录去畸变、裁黑边、主点居中，并可同时导出**更新后的 cameras.xml** 和 **COLMAP 参数**。

**数据准备**：`cameras.xml` + 单个图像目录（`*.png`）。

**命令**
```bash
python Convert/xml_undistort_gs_params.py \
  --xml cameras.xml \
  --images_dir images \
  --output_dir output \
  --output_xml output/cameras_undistorted.xml
# 可选开关（默认均开启，加 no_ 前缀关闭）：
#   --unified_roi / --no_unified_roi          统一裁切区域
#   --unified_intrinsics / --no_unified_intrinsics  统一内参
#   --export_colmap / --no_export_colmap      导出 COLMAP sparse/0
```

**产出**：`output_dir/` 下去畸变图像 + 更新后的 `cameras.xml` + （默认）`sparse/0/` COLMAP 参数。

---

## 4. `calib_to_colmap.py`

**功能**：纯参数转换，**不做去畸变**。把 `calib.json` 转成 COLMAP 文本格式，相机模型用 **OPENCV（保留 k1,k2,p1,p2 畸变）**。适合让 COLMAP 自己去畸变 / 后续做 MVS 的场景。

**数据准备**：`calib.json`（+ 可选图像目录，仅用于取文件名）。

**命令**
```bash
python Convert/calib_to_colmap.py \
  --calib output/calib_undistorted.json \
  --images output \
  --output colmap_sparse \
  --image-ext .png --validate
```

**产出**（`--output` 目录）：`cameras.txt`（OPENCV）、`images.txt`、`points3D.txt`（空）、`database_info.txt`（统计信息）。把该目录改名为 `sparse/0/` 即可被 COLMAP 导入。

---

## 5. `calib_to_agisoft_cameras.py`

**功能**：把 `calib.json` 转成 **Agisoft Metashape 的 `cameras.xml`**（严格对齐 Metashape 结构，含 sensors/components/cameras）。用于把 libCalib 标定导入 Agisoft 工程。注意 Metashape 的 cx/cy 是相对图像中心的偏移，脚本已自动换算。

**数据准备**：`calib.json`。

**命令**
```bash
python Convert/calib_to_agisoft_cameras.py \
  --input output/calib_undistorted.json \
  --output output/cameras_generated.xml
```

**产出**：一份 Agisoft 可导入的 `cameras.xml`。

---

## 6. `merge_images.py`

**功能**：目录重组——把**以相机为单位**的目录结构（`相机A/img0.png img1.png...`）转成**以帧为单位**（`1/1.png 2.png ... 48.png`）。相机文件夹按名称排序决定相机编号，文件夹内图片按名称排序决定帧号。会自动清理 `.` 开头的隐藏文件。

**数据准备**：`images/` 下若干相机子文件夹，每个含等数量、同顺序的图片。

**用法**：无命令行参数，直接改文件末尾的 `source_directory` / `output_directory` 后运行：
```bash
python Convert/merge_images.py
```

**产出**：`output_directory/{帧号}/{相机号}.png`，即可作为前面去畸变脚本的 `images_group` / by_frame 输入。

---

## 7. `Projection_one_image.py`（调试/验证）

**功能**：把一个 `.ply` 点云投影到**单个相机视角**，画出投影叠加图，用来**肉眼验证内外参是否正确**。参数（R/t、fx/fy/cx/cy、点云路径）写死在脚本顶部，需手动改。

**用法**
```bash
python Convert/Projection_one_image.py   # 先在文件里改好相机参数与 data/*.ply 路径
```

**产出**：一张投影叠加 PNG（默认 `agisoft/output/projection_overlay2.png`，点投到画面内的数量会打印出来）。

---

## 推荐流程（calib.json 路线）

```
preprocess/batch_cut_video.py        # 视频 → by_frame 图像
        │
        ▼
Convert/undistort_from_calib.py      # 去畸变 + 主点居中 + COLMAP(PINHOLE)
        │
        ▼
output/<name>/{images, sparse/0}     # 直接用于 3DGS 训练
```

Agisoft 路线则用 `batch_undistort_frames.py`（多帧）或 `xml_undistort_gs_params.py`（单目录）替换中间一步。
