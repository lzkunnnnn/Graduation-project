import os
import torch
import numpy as np
import lpips
from PIL import Image

# 初始化 LPIPS 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='alex').to(device)  # 使用 AlexNet 作为骨干网络

def calculate_lpips(image1, image2):
    """计算两张图片的 LPIPS 距离"""
    # 转换为 tensor 并归一化到 [0, 1]

    image1 = image1.resize((224, 224), Image.BILINEAR)
    image2 = image2.resize((224, 224), Image.BILINEAR)
    image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 将数据移动到 GPU
    image1_tensor = image1_tensor.to(device)
    image2_tensor = image2_tensor.to(device)

    # 计算 LPIPS 距离
    distance = lpips_model(image1_tensor, image2_tensor)
    print (distance)
    return distance.item()

def find_most_similar_image(input_image_path, vehicle_images_dir):
    """在车辆照片目录中找到与输入图片最相似的图片"""
    # 加载输入图片
    input_image = Image.open(input_image_path).convert('RGB')

    # 初始化最小距离和对应的图片信息
    min_distance = float('inf')
    most_similar_image_path = None

    # 遍历车辆照片目录，仅处理以 "ID" 开头的文件
    for root, _, files in os.walk(vehicle_images_dir):
        for file in files:
            if file.startswith('ID') and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 构建完整路径
                vehicle_image_path = os.path.join(root, file)

                try:
                    # 加载车辆照片
                    vehicle_image = Image.open(vehicle_image_path).convert('RGB')

                    # 计算 LPIPS 距离
                    distance = calculate_lpips(input_image, vehicle_image)

                    # 更新最小距离和对应的图片路径
                    if distance < min_distance:
                        min_distance = distance
                        most_similar_image_path = vehicle_image_path
                except Exception as e:
                    print(f"无法加载图片 {vehicle_image_path}: {e}")
                    continue

    return most_similar_image_path, min_distance

# 示例使用
if __name__ == "__main__":
    # 输入图片路径
    input_image_path = "../output/query_image/0.png"  # 替换为你的输入图片路径

    # 车辆照片目录（从视频中截取的车辆照片存放目录）
    vehicle_images_dir = "../output/video"  # 替换为你的车辆照片目录

    # 找到最相似的图片
    most_similar_image_path, min_distance = find_most_similar_image(input_image_path, vehicle_images_dir)

    if most_similar_image_path:
        print(f"最相似的车辆照片是: {most_similar_image_path}")
        print(f"LPIPS 距离: {min_distance:.4f}")
    else:
        print("未找到车辆照片！")