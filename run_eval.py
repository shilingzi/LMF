import os
import sys
import torch
import yaml
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def load_model(model_path, config_path):
    """加载模型和配置"""
    print(f"正在加载模型: {model_path}")
    print(f"配置文件: {config_path}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 导入make_model函数
    sys.path.insert(0, '.')
    from models import make as make_model
    
    # 创建模型
    model = make_model(config['model'])
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 加载权重到模型
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("模型加载完成")
    return model, config

def load_image(path):
    """加载图像并转换为张量"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

def evaluate_model(model, data_dir, scale=4):
    """评估模型在数据集上的表现"""
    device = next(model.parameters()).device
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"找到 {len(image_files)} 张图像")
    
    # 如果没有图像，返回
    if not image_files:
        print(f"在 {data_dir} 中没有找到图像文件")
        return
    
    # 从eval_simple.py导入评估函数
    sys.path.insert(0, '.')
    
    try:
        from eval_simple import evaluate_on_image
        eval_fn = evaluate_on_image
    except ImportError:
        # 尝试从test.py导入
        try:
            from test import eval_psnr
            eval_fn = eval_psnr
        except ImportError:
            print("无法导入评估函数，请检查项目结构")
            return
    
    # 评估
    total_psnr = 0
    count = 0
    
    # 对每个图像评估
    for img_file in tqdm(image_files, desc=f"评估 {os.path.basename(data_dir)}"):
        try:
            img_path = os.path.join(data_dir, img_file)
            img = load_image(img_path).to(device)
            
            # 调用评估函数
            psnr = evaluate_on_image(model, img, scale=scale)
            total_psnr += psnr
            count += 1
            
            print(f"{img_file}: PSNR = {psnr:.2f} dB")
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {str(e)}")
    
    # 计算平均PSNR
    if count > 0:
        avg_psnr = total_psnr / count
        print(f"\n数据集 {os.path.basename(data_dir)} 的平均 PSNR: {avg_psnr:.2f} dB")
    else:
        print(f"\n未能成功评估任何图像")

def main():
    # 设置参数
    model_path = 'save/temp/epoch-best.pth'
    config_path = 'save/temp/config.yaml'
    data_dir = 'load/Set5'
    scale = 4
    
    # 加载模型
    model, config = load_model(model_path, config_path)
    
    # 评估模型
    evaluate_model(model, data_dir, scale)

if __name__ == "__main__":
    main() 