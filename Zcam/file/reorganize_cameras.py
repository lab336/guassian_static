import os
import shutil
from pathlib import Path

def reorganize_camera_data():
    """
    重新组织相机数据文件夹结构
    从: data/timestamp/xxx.png
    到: reorganized/Camera_XX/1.png, 2.png, ...
    """
    
    base_path = Path("data")
    output_path = Path("reorganized")
    
    # 创建输出目录
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()
    
    # 获取所有时间戳文件夹
    timestamp_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    if not timestamp_folders:
        print("没有找到时间戳文件夹")
        return
    
    # 读取第一个文件夹的CamerasSN.ini来获取相机信息
    first_folder = timestamp_folders[0]
    ini_path = first_folder / "CamerasSN.ini"
    
    if not ini_path.exists():
        print(f"在 {first_folder} 中找不到 CamerasSN.ini 文件")
        return
    
    # 解析相机序列号
    cameras = []
    with open(ini_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('Camera') and '=' in line:
            key, value = line.split('=', 1)
            if value.strip():  # 只处理非空的相机序列号
                camera_num = int(key.replace('Camera', ''))
                cameras.append((camera_num, value.strip()))
    
    cameras.sort()  # 按相机编号排序
    
    print(f"找到 {len(cameras)} 台相机")
    print(f"找到 {len(timestamp_folders)} 组图片")
    
    # 为每台相机创建文件夹
    for camera_num, camera_sn in cameras:
        camera_folder = output_path / f"Camera_{camera_num:02d}"
        camera_folder.mkdir()
        print(f"创建相机文件夹: {camera_folder.name}")
        
        # 复制该相机的所有图片
        for i, timestamp_folder in enumerate(timestamp_folders, 1):
            source_image = timestamp_folder / f"{camera_num:03d}.png"
            if source_image.exists():
                dest_image = camera_folder / f"{i}.png"
                shutil.copy2(source_image, dest_image)
                print(f"  复制 {source_image} -> {dest_image}")
            else:
                print(f"  警告: 找不到图片 {source_image}")
    
    print("\n重新组织完成!")
    print(f"新的文件结构已创建在 '{output_path}' 文件夹中")

if __name__ == "__main__":
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    reorganize_camera_data()