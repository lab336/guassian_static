# A2PM 替代 COLMAP Matcher 使用文档

本文档对应脚本：`colmap/a2pm_colmap.py`。

## 目标

脚本保留 COLMAP 的建库、`mapper`、模型导出和可选稠密重建，但不运行 COLMAP 的 `exhaustive_matcher/sequential_matcher`。图像对匹配由 `colmap/A2PM-MESA` 里的 A2PM matcher 产生，再写入 COLMAP database 的：

- `keypoints`
- `matches`
- `two_view_geometries`

随后 COLMAP 用这些匹配估计相机内参、外参和稀疏点云。

## 数据结构

当前支持这种多相机矩阵数据：

```text
data/twopeople/images/
  1/
    1.png
    2.png
    ...
    100.png
  2/
    1.png
    ...
```

脚本会按数字顺序读取帧目录和相机图片。输出帧名会标准化为 `frame_000001`、`frame_000002`。

## 推荐命令

在仓库根目录运行：

```powershell
conda run -n A2PM-new python colmap/a2pm_colmap.py `
  --images_root data/twopeople/images `
  --output_root output/twopeople_a2pm `
  --frames 1 `
  --point_matcher mast3r `
  --pair_mode sequential `
  --pair_window 5 `
  --loop_pairs `
  --force
```

如果先小规模验证：

```powershell
conda run -n A2PM-new python colmap/a2pm_colmap.py `
  --images_root data/twopeople/images `
  --output_root output/twopeople_a2pm_test `
  --frames 1 `
  --max_images 12 `
  --max_pairs 20 `
  --point_matcher mast3r `
  --force
```

## 主要输出

```text
output/twopeople_a2pm/
  debug/
    scenes/frames/frame_000001/
      images/
      database.db
      sparse/0/
      model_txt/
        cameras.txt
        images.txt
        points3D.txt
      a2pm_debug/
    logs/frame_000001/
    stats/frame_000001.json
    summary.json
  points_cloud/
    frame_000001.ply
```

重点看：

- `model_txt/cameras.txt`：内参
- `model_txt/images.txt`：外参
- `model_txt/points3D.txt`：稀疏点云
- `points_cloud/frame_000001.ply`：稀疏点云 PLY
- `debug/stats/frame_000001.json`：匹配数、点云数量、重投影误差统计

## 匹配模式

默认是 `--a2pm_mode point`，使用 A2PM-MESA 的 point matcher 直接生成点匹配。这个模式不需要 SAM 分割结果，适合先替换 COLMAP matcher。

完整 A2PM/MESA 区域到点匹配需要每张图的 SAM `.npy` 分割结果。准备好后可用：

```powershell
conda run -n A2PM-new python colmap/a2pm_colmap.py `
  --images_root data/twopeople/images `
  --output_root output/twopeople_a2pm_full `
  --frames 1 `
  --a2pm_mode full `
  --point_matcher mast3r `
  --area_matcher dmesa `
  --geo_matcher gam `
  --sem_root path/to/samres `
  --force
```

## 图像对策略

多相机环形矩阵通常只匹配相邻相机更稳：

- `--pair_mode sequential --pair_window 5 --loop_pairs`：默认，匹配每台相机前后窗口内的相机，并首尾相连。
- `--pair_mode exhaustive`：所有图片两两匹配，100 张图会有 4950 对，A2PM 会很慢。
- `--pair_mode pairs_file --pairs_file pairs.txt`：手写图片对，每行两个文件名。

`pairs.txt` 示例：

```text
1.png 2.png
2.png 3.png
99.png 100.png
100.png 1.png
```

## 稠密点云

默认只输出 COLMAP mapper 的稀疏点云。需要稠密点云时加：

```powershell
--dense
```

这会继续运行：

1. `image_undistorter`
2. `patch_match_stereo`
3. `stereo_fusion`

稠密输出在：

```text
output/twopeople_a2pm/dense_points_cloud/frame_000001.ply
```

## 常用参数

- `--point_matcher mast3r`：读取 `colmap/A2PM-MESA/conf/point_matcher/mast3r.yaml`。
- `--match_width 512 --match_height 512`：A2PM 输入尺寸，MASt3R 通常用 512x512。
- `--match_num 4000`：每对图最多保留的 A2PM 原始匹配数。
- `--ransac --ransac_max_error 4.0`：写入 COLMAP 前做 Fundamental Matrix 几何过滤。
- `--min_matches 30`：过滤后少于该数量的图像对不写入数据库。
- `--camera_model SIMPLE_PINHOLE`：COLMAP 初始相机模型。
- `--single_camera`：所有图片共用同一个相机模型；多相机矩阵通常不建议，除非图片尺寸和内参确实一致。

## 注意事项

1. `points/A2PM-MESA` 当前磁盘上不存在，实际 A2PM 路径是 `colmap/A2PM-MESA`，脚本默认使用这个路径。若以后移动目录，用 `--a2pm_root` 指定。
2. A2PM 依赖应在 `A2PM-new` 环境里可用，尤其是 `hydra-core`、`omegaconf`、`torch`、`opencv-python` 和对应 matcher 的权重文件。
3. `full` 模式必须提供 SAM `.npy`，否则无法运行 MESA/DMESA 的区域匹配。
4. 如果 `mapper` 报 `no initial pair`，优先尝试增大 `--pair_window`、放宽 `--ransac_max_error`，或临时加 `--no-ransac` 检查 A2PM 原始匹配是否足够。
