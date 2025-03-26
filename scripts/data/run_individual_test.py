"""
Set5数据集上的简单测试脚本
"""

import os
import torch
from PIL import Image
from torchvision import transforms
import models
from utils import make_coord, calc_psnr
from tqdm import tqdm

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

model_path = 'save/edsr-b_lm-lmlte/epoch-best.pth'
model_spec = torch.load(model_path, map_location='cpu')['model']
model = models.make(model_spec, load_sd=True).to(device)
model.eval()

# 设置数据路径和配置
data_dir = './load/Set5'
scale_factor = 4
scale_max = 4

# 创建输出目录
output_dir = './outputs/Set5'
os.makedirs(output_dir, exist_ok=True)

# 处理所有图像
image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
psnr_values = []

for img_file in tqdm(image_files, desc="处理图像"):
    img_path = os.path.join(data_dir, img_file)
    
    # 加载高分辨率图像为Ground Truth
    hr_img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
    
    # 降采样创建低分辨率输入
    h_lr = hr_img.shape[1] // scale_factor
    w_lr = hr_img.shape[2] // scale_factor
    lr_img = transforms.Resize((h_lr, w_lr), antialias=True)(hr_img)
    
    # 超分辨率
    with torch.no_grad():
        # 计算目标尺寸
        h = hr_img.shape[1]
        w = hr_img.shape[2]
        
        # 准备坐标和单元格
        coord = make_coord((h, w)).to(device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        
        # 进行超分辨率处理
        cell_factor = max(scale_factor/scale_max, 1)
        
        # 打印形状信息进行调试
        print(f"输入形状: {lr_img.shape}")
        print(f"坐标形状: {coord.shape}, 坐标类型: {coord.dtype}")
        print(f"单元格形状: {cell.shape}, 单元格类型: {cell.dtype}")
        
        # 使用demo.py中的方式处理图像
        pred = model(((lr_img - 0.5) / 0.5).unsqueeze(0),
                    coord.unsqueeze(0), cell_factor * cell.unsqueeze(0))[0]
        
        # 转换回正常范围
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1)
        
        # 计算PSNR
        psnr = calc_psnr(pred, hr_img)
        psnr_values.append(psnr)
        
        print(f"{img_file}: PSNR = {psnr:.2f} dB")
        
        # 保存超分辨率结果
        output_path = os.path.join(output_dir, f"sr_{img_file}")
        transforms.ToPILImage()(pred.cpu()).save(output_path)
        
        # 同时保存低分辨率输入作为对比
        lr_output_path = os.path.join(output_dir, f"lr_{img_file}")
        transforms.ToPILImage()(lr_img.cpu()).save(lr_output_path)

# 输出平均PSNR
if psnr_values:
    avg_psnr = sum(psnr_values) / len(psnr_values)
    print(f"\n平均PSNR: {avg_psnr:.2f} dB")
    
    # 将结果写入文件
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"模型: {model_path}\n")
        f.write(f"测试数据集: Set5\n")
        f.write(f"放大倍数: {scale_factor}x\n")
        f.write(f"平均PSNR: {avg_psnr:.2f} dB\n\n")
        for i, img_file in enumerate(image_files):
            f.write(f"{img_file}: PSNR = {psnr_values[i]:.2f} dB\n")
else:
    print("未能处理任何图像") 