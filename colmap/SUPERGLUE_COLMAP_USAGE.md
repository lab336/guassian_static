# SuperGlue COLMAP 使用说明

脚本：`colmap/superglue_colmap.py`

作用：用 `colmap/SuperGluePretrainedNetwork` 替代 COLMAP 自带 matcher，估计每帧相机的内参、外参、稀疏点云，并把光流估计出的速度写入 PLY。

> 大多数参数已经设了适合本数据集（约 100 路相机、4K 图像）的默认值，**正常情况下只需要几个参数**，见第 3 节。需要调的参数在第 5 节有详细解释。

---

## 1. 数据放法

图像目录（每个子目录是一个时间帧，里面是同一时刻各路相机的图）：

```text
data/twopeople/images/1/1.png      # 帧 1，相机 1
data/twopeople/images/1/2.png      # 帧 1，相机 2
data/twopeople/images/2/1.png      # 帧 2，相机 1
```

光流目录（文件名和相机编号一致，`.npy` 形状为 `(H, W, 2)`）：

```text
data/twopeople/flows/1/1.npy
data/twopeople/flows/1/2.npy
data/twopeople/flows/2/1.npy
```

`--images_root`、`--flows_root`、`--output_root` 都有默认值（分别指向上面这些路径和 `output/twopeople_superglue`），**只要在仓库根目录运行脚本，就不用手动写这三个路径**。

---

## 2. 先跑一个小测试

第一次用先确认环境和流程没问题：只取第 1 帧的前 12 张图、20 对匹配。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --max_images 12 --max_pairs 20 --output_root output/twopeople_superglue_test --force
```

如果 Windows 上 `conda run` 报临时文件占用，直接用环境里的 python：

```powershell
C:\ProgramData\miniconda3\envs\A2PM-new\python.exe colmap/superglue_colmap.py --frames 1 --max_images 12 --max_pairs 20 --output_root output/twopeople_superglue_test --force
```

---

## 3. 正式运行（推荐用法）

相机阵列是**固定不动**的（各帧之间相机位置不变），所以推荐加 `--static_rig`：先用一帧把所有相机的内外参解算出来，再用同一套相机位姿去三角化每一帧。这样：

- **所有帧共享同一套内参 + 外参**（在参考帧上解算并用对极约束清洗后重解一次更新），帧与帧之间天然对齐、同一尺度；
- 每一帧只用已知位姿清洗匹配并三角化点云，**不再改动内外参**（`--rig_fix_poses` 默认开启，位姿全程锁死）；
- 位姿只解一次，更准、更稳，点云噪声更小。

> 参考帧默认用所选的第一帧，可用 `--rig_ref_frame <帧名>` 指定，建议选**人物清晰、遮挡少、相机覆盖好**的一帧——它的位姿质量决定所有帧。

```powershell
# 处理第 1~3 帧（推荐）
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --force

# 处理全部帧
conda run -n A2PM-new python colmap/superglue_colmap.py --static_rig --force

# 只处理单帧（此时 static_rig 等同于普通重建）
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --force
```

如果相机会移动（不是固定阵列），就**不要**加 `--static_rig`，让每帧各自独立重建：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --force
```

> 运行时间提示：**参考帧**用 `exhaustive`（两两全配对）+ 全分辨率解算，最准但最慢（只跑一次）。**后续帧**会自动提速：用参考帧的内外参挑出真正重叠的相机对（共视选对，约 4950→600~900 对），匹配分辨率默认降到 2560（`--rig_resize`），并且每张图的 SuperPoint 特征只算一次复用。整体后续帧通常比参考帧快约 10 倍量级，点云质量基本不变。相关开关见第 5 节。

---

## 4. 输出在哪里

输出已经精简成**直接可喂 3DGS** 的三样东西：

```text
output/twopeople1/
  undistorted/
    images/
      1/                 # 第 1 帧：去畸变 + 主点居中后的图像
        1.png 2.png ...
      2/                 # 第 2 帧 ...
    sparse/
      0/                 # 所有帧共享的一组相机（PINHOLE，主点已居中）+ 外参 + points3D
        cameras.bin images.bin points3D.bin
  points/
    1.ply 2.ply ...      # 每帧稀疏点云（带速度属性）
```

关键点：
- **`undistorted/sparse/0/` 是所有帧共用的一组内外参**（静态机位），相机模型是 PINHOLE 且**主点已居中**（cx=W/2、cy=H/2），所以原版 3DGS 不会再因为忽略 cx/cy 而发糊。
- **`undistorted/images/<帧>/`** 是对应的去畸变+居中裁剪图像。同一台相机在所有帧里尺寸一致；不同相机尺寸可以不同（都已在 `cameras.bin` 里各自记好）。
- **`points/<帧>.ply`** 是每帧点云（带 `vx vy vz` 等速度属性，**仅供你自己的 4D 流程用**；不要拿它当 3DGS 的 input.ply，3DGS 用 `sparse/0/points3D.bin` 初始化）。

输出帧名与输入帧文件夹名一致：输入 `images/1` → `images/1/`、`points/1.ply`。

喂 3DGS 时，把某一帧组成标准布局即可：`images/` 用 `undistorted/images/<帧>/`，`sparse/0/` 用 `undistorted/sparse/0/`。

> 去畸变+居中是**默认行为**，无需参数。不需要这套导出就加 `--no-undistort`（那样不产生 `undistorted/`）；想压缩图像尺寸可用 `--undistort_max_image_size 2000` 限制长边。

`points/1.ply` 每个点包含：

```text
x y z                  # 坐标
red green blue         # 颜色
vx vy vz               # 速度（COLMAP 世界单位 / velocity_dt）
velocity_confidence    # 速度置信度 0~1
velocity_valid         # 该点速度是否有效 0/1
velocity_views         # 参与速度三角化的相机数
```

> `.ply` **默认是 ASCII（文本）格式**，可直接用记事本/编辑器打开查看，CloudCompare、Open3D、3DGS 也都能读。想要更小、读写更快的二进制 PLY 就加 `--no-ply_ascii`（稀疏点云体积差别很小）。

---

## 5. 需要了解的参数

下面是**实际可能要改**的参数，其它的保持默认即可。

### 选帧 / 路径 / 覆盖

| 参数 | 说明 |
| --- | --- |
| `--frames 1` | 只处理第 1 帧。 |
| `--frames 1:3` | 处理第 1 到第 3 帧（含两端）。也支持逗号，如 `--frames 1,3,5`。不写则处理全部帧。 |
| `--output_root <目录>` | 输出目录，默认 `output/twopeople_superglue`。 |
| `--force` | 覆盖该帧已有的旧结果。重复跑同一目录时一般都要加。 |

### 静态相机阵列

| 参数 | 说明 |
| --- | --- |
| `--static_rig` | **核心开关**。相机固定不动时加上：用一帧解出的相机位姿三角化所有帧，保证跨帧对齐、降低噪声。相机会动则不要加。 |
| `--rig_ref_frame <帧名>` | `--static_rig` 时用哪一帧解算参考位姿，默认用所选的第一帧。建议选**人物清晰、遮挡少、相机覆盖好**的一帧，参考帧的位姿质量决定了所有帧的质量。 |

### 精度 / 显存（按机器情况调）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--resize 2560` | 2560 | SuperGlue 处理时把图像长边缩放到的像素数。**越大特征点定位越准、点云越细，但越吃显存、越慢。** 显存不够就调小（如 `--resize 2048`）；想要极致精度且显存足够可用 `--resize 3200` 或 `--resize -1`（原图）。 |
| `--camera_model OPENCV` | OPENCV | 相机模型。默认 OPENCV 会估计镜头畸变，能明显减少边缘噪声。若某些相机点太少导致畸变估计不稳，改成 `--camera_model SIMPLE_RADIAL`（更简单更稳）。 |

### 点云清洗（结果太脏或太空时调）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--ply_min_track_len 3` | 3 | 只保留被**至少这么多相机**看到的点。值越大越干净但点越少；设 `0` 关闭该过滤。 |
| `--ply_max_reproj_error 2.0` | 2.0 | 丢弃重投影误差大于该值（像素）的点。值越小越干净；设 `0` 关闭该过滤。点太少时可放宽到 `3.0~4.0`。 |

### 提速（--static_rig 后续帧，默认已开启）

参考帧之后的每一帧都会用已解出的内外参来加速，质量基本不变。一般不用动，跑得太慢/太空时再调：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--rig_pair_mode` | covisibility | 后续帧只匹配**真正重叠**的相机对（共视 + 视角相邻），约 4950→600~900 对。想退回每帧全配对用 `--rig_pair_mode same`。 |
| `--rig_resize 2560` | 2560 | 后续帧的匹配分辨率（位姿已固定，降一点更快）。想后续帧也用全分辨率：`--rig_resize -1`（更慢更细）。 |
| `--rig_covis_top_k 12` | 12 | 每台相机保留多少个共视最强的邻居。点云偏空就调大（如 18~24，更稳但更慢）。 |
| `--rig_geo_neighbors 6` | 6 | 每台相机额外按视角相邻补几个邻居，保证物理相邻相机一定匹配（应对人物走动）。 |

> SuperPoint 特征缓存（每张图只检测一次、跨相机对复用）是自动的，参考帧和后续帧都生效，无需开关。

### 压榨硬件（GPU/CPU 重叠，默认已开启）

下面这些只影响**速度和数值精度**，不改变算法/选对策略，目的是把 GPU 和 CPU 同时喂满：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--pipeline` | 开 | `--static_rig` 下，把第 N 帧的三角化+去畸变导出（CPU/磁盘）和第 N+1 帧的匹配（GPU）**重叠**起来。各帧相互独立，输出完全一致。墙钟时间常能接近砍半。想串行排错用 `--no-pipeline`。 |
| `--decode_workers 4` | 4 | 后台预解码图片的线程数，让磁盘/CPU 的 4K 解码和 GPU 检测重叠。输出不变。机器核多可调大；设 `1` 关闭预取。 |
| `--fp16` | 开 | SuperPoint+SuperGlue 用 fp16 混合精度（Ampere/Ada 上约 1.5~2×）。**关键点坐标仍保持 fp32**，匹配点集只有极微差异（后面有 RANSAC + 对极清洗兜底）。要逐位一致的全精度匹配用 `--no-fp16`。 |
| `--tf32` | 开 | 允许 Ampere+ 上的 TF32 矩阵/卷积（小幅加速，数值差异比 fp16 更小）。要绝对精确用 `--no-tf32`。 |

> 说明：`--fp16` / `--tf32` 是「近似无损」——实测对 SfM 质量基本无影响，但不是逐位相同。若你要求严格可复现，用 `--no-fp16 --no-tf32`，仍可享受 `--pipeline` + 预解码（这两者**逐位一致**）带来的提速。COLMAP 三角化/去畸变本身已默认吃满所有 CPU 核。

### 其它默认已开启（一般不用动）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--epipolar_filter` | 开 | 用估出来的位姿做对极约束清洗匹配：参考帧会清洗后**重解一次**（更新内外参），其余每帧三角化前也清洗一遍。明显降噪。极少数情况想关用 `--no-epipolar_filter`。 |
| `--epipolar_max_error 1.5` | 1.5 | 对极清洗的像素阈值。越小越严格、越干净；匹配被删太多就放宽到 `2.0~3.0`。 |
| `--rig_fix_poses` | 开 | `--static_rig` 下把每帧位姿锁死为参考帧的位姿（保证全程共享一套外参）。需要 COLMAP ≥ 3.7；若报错会自动退回不锁。 |
| `--undistort` | 开 | 输出去畸变 + **主点居中**的图像和 PINHOLE 内参（3DGS-ready），见第 4 节。不要就 `--no-undistort`。 |

### 其它常用开关

| 参数 | 说明 |
| --- | --- |
| `--no-compute_velocity` | 不估计速度，只输出点云（更快）。 |
| `--keep_workspace` | 保留临时图片、database、日志和 TXT 模型，排错时用。 |
| `--single_camera` | 所有相机共用一套内参。本数据集是 100 路**不同**相机，**不要**加。 |

---

## 6. 结果不理想时怎么办

**点云太空 / COLMAP 报 `no initial pair` / 匹配太少**：放宽匹配和建图门槛。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --static_rig --min_matches 15 --mapper_min_num_matches 10 --mapper_init_min_num_inliers 15 --force
```

**COLMAP mapper 报 `ba_config.NumImages() >= 2` / `At least two images must be registered for global bundle-adjustment`**：这是 COLMAP 多模型重建时偶发的全局 BA 断言。脚本会自动改用单模型重试；如果你想手动避开，也可以直接加：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:201 --static_rig --no-mapper_multiple_models --force
```

**点云还是有噪声**：收紧清洗门槛（更干净）。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --ply_min_track_len 4 --ply_max_reproj_error 1.5 --force
```

**后续帧还是太慢**：把后续帧分辨率再降一点、共视邻居数再少一点。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --rig_resize 2048 --rig_covis_top_k 10 --force
```

**后续帧点云比参考帧偏空**（共视选对漏了相机对）：调大邻居数，或退回每帧全配对。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --rig_covis_top_k 20 --rig_geo_neighbors 10 --force
# 或彻底退回（最稳最慢）：
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --rig_pair_mode same --force
```

**速度有效点太少**：放宽速度门槛。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --static_rig --velocity_min_views 2 --velocity_max_reproj_error 6.0 --force
```

**速度数值明显偏大 / 偏小**：优先换光流格式（`norm` / `midnorm` / `pixel`）。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --static_rig --flow_format pixel --force
```

**想看中间文件排错**：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --keep_workspace --force
```

---

## 7. 速度是怎么估计的

不是单相机直接投影，而是多视角光流三角化：

1. 读取 COLMAP 中每个 3D 点被哪些相机真实看到；
2. 在这些相机的光流图里采样它到下一帧的像素位移；
3. 用多个相机的“下一帧像素”射线重新三角化出下一时刻的 3D 点；
4. 用重投影误差逐步剔除错误光流和遮挡视角；
5. 速度 = （下一时刻位置 − 当前位置）/ `velocity_dt`，写入 `vx/vy/vz`。

这样比单视角投影更稳，可以减少镂空点或错误表面投影造成的伪速度。
