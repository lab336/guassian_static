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

- 所有帧的点云在**同一个坐标系、同一个尺度**下，帧与帧之间天然对齐；
- 相机位姿只解一次，更准、更稳，噪声更小。

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

> 运行时间提示：默认是 `exhaustive`（两两全配对）匹配，100 路相机每帧约 4950 对，质量最好但较慢。嫌慢看第 6 节的提速办法。

---

## 4. 输出在哪里

成功后默认只保留两个结果目录：

```text
output/twopeople_superglue/
  sparse/
    frame_000001/
      0/
        cameras.bin      # 内参（含畸变）
        images.bin       # 外参（位姿）
        points3D.bin
  points_cloud/
    frame_000001.ply     # 带速度属性的稀疏点云
```

`points_cloud/frame_000001.ply` 每个点包含：

```text
x y z                  # 坐标
red green blue         # 颜色
vx vy vz               # 速度（COLMAP 世界单位 / velocity_dt）
velocity_confidence    # 速度置信度 0~1
velocity_valid         # 该点速度是否有效 0/1
velocity_views         # 参与速度三角化的相机数
```

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

**点云还是有噪声**：收紧清洗门槛（更干净）。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --ply_min_track_len 4 --ply_max_reproj_error 1.5 --force
```

**嫌 exhaustive 太慢**：改成顺序匹配并加大窗口（前提是相机编号大致按空间相邻排列）。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --pair_mode sequential --pair_window 12 --force
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
