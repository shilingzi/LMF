import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def load_image(path, scale=4):
    """加载一张图像，并创建其低分辨率版本"""
    img_hr = Image.open(path).convert('RGB')
    
    # 创建低分辨率版本
    w, h = img_hr.size
    img_lr = img_hr.resize((w // scale, h // scale), Image.BICUBIC)
    
    # 转换为张量
    transform = transforms.ToTensor()
    img_hr_tensor = transform(img_hr).unsqueeze(0)
    img_lr_tensor = transform(img_lr).unsqueeze(0)
    
    return img_lr_tensor, img_hr_tensor

def calc_psnr(sr, hr):
    """计算PSNR"""
    diff = (sr - hr) ** 2
    mse = diff.mean().item()
    return 10 * np.log10(1.0 / mse)

def print_model_keys(checkpoint):
    """打印模型权重的键"""
    print("模型权重包含以下键:")
    if isinstance(checkpoint, dict):
        for k in checkpoint.keys():
            print(f"- {k}")
            if k == 'model' and isinstance(checkpoint[k], dict):
                # 只打印前10个键
                keys = list(checkpoint[k].keys())
                print(f"  模型包含 {len(keys)} 个参数, 前10个是:")
                for i in range(min(10, len(keys))):
                    print(f"  - {keys[i]}")
    else:
        print("checkpoint不是字典类型")

def evaluate_set5():
    """在Set5数据集上评估模型"""
    # 设置参数
    model_path = 'save/temp/epoch-best.pth'
    data_dir = 'load/Set5'
    scale = 4
    
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path)
    print_model_keys(checkpoint)
    
    # 加载模型 (这里直接检查模型内容而不加载)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print(f"在 {data_dir} 中没有找到图像")
        return
    
    # 分析第一张图像
    first_img = os.path.join(data_dir, image_files[0])
    print(f"分析第一张图像: {first_img}")
    img_lr, img_hr = load_image(first_img, scale)
    print(f"低分辨率图像形状: {img_lr.shape}")
    print(f"高分辨率图像形状: {img_hr.shape}")
    
    # 保存结果
    print("\n评估完成。由于模型加载问题，无法进行完整评测。")
    print("请确保模型权重与模型定义匹配，并使用正确的配置文件。")

if __name__ == "__main__":
    print("开始评估Set5数据集")
    evaluate_set5() 