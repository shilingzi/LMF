import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import time

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

def create_model(checkpoint):
    """从特殊格式的权重创建模型"""
    # 检查是否包含必要的字段
    if 'model' not in checkpoint:
        print("错误：模型权重中没有'model'字段")
        return None
    
    model_data = checkpoint['model']
    if not isinstance(model_data, dict) or 'name' not in model_data or 'args' not in model_data or 'sd' not in model_data:
        print("错误：模型数据格式不正确，缺少必要字段")
        return None
    
    print(f"模型类型: {model_data['name']}")
    
    # 导入make_model函数
    sys.path.insert(0, '.')
    try:
        from models import make as make_model
        
        # 尝试创建模型
        model_args = {
            'name': model_data['name'],
            'args': model_data['args']
        }
        
        model = make_model(model_args)
        print("模型创建成功")
        
        # 加载模型状态字典
        model.load_state_dict(model_data['sd'])
        print("模型权重加载成功")
        
        return model
    except Exception as e:
        print(f"创建或加载模型时出错: {str(e)}")
        return None

def evaluate_dataset(model, data_dir, scale=4, device='cuda'):
    """评估模型在指定数据集上的性能"""
    print(f"\n开始评估数据集: {os.path.basename(data_dir)}")
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print(f"在 {data_dir} 中没有找到图像")
        return 0
    
    try:
        # 评估每张图像
        total_psnr = 0
        total_time = 0
        for img_file in tqdm(image_files, desc=f"评估 {os.path.basename(data_dir)}"):
            img_path = os.path.join(data_dir, img_file)
            img_lr, img_hr = load_image(img_path, scale)
            
            # 移动到设备
            img_lr = img_lr.to(device)
            img_hr = img_hr.to(device)
            
            # 创建坐标
            h, w = img_hr.shape[-2], img_hr.shape[-1]
            coord = make_coord((h, w)).to(device)
            coord = coord.unsqueeze(0)  # 添加批次维度
            
            # 创建cell
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / h
            cell[:, :, 1] *= 2 / w
            
            # 前向传播
            with torch.no_grad():
                start_time = time.time()
                pred = model(img_lr, coord, cell)
                end_time = time.time()
                inference_time = end_time - start_time
                total_time += inference_time
                
                pred = pred.view(1, h, w, 3).permute(0, 3, 1, 2)
            
            # 计算PSNR
            psnr = calc_psnr(pred, img_hr)
            total_psnr += psnr
            print(f"{img_file}: PSNR = {psnr:.2f} dB, 推理时间: {inference_time:.2f}秒")
        
        # 计算平均PSNR和推理时间
        avg_psnr = total_psnr / len(image_files)
        avg_time = total_time / len(image_files)
        print(f"\n{os.path.basename(data_dir)}数据集结果:")
        print(f"平均PSNR: {avg_psnr:.2f} dB")
        print(f"平均推理时间: {avg_time:.2f}秒/图像")
        return avg_psnr
    except Exception as e:
        print(f"评估数据集时出错: {str(e)}")
        return 0

def make_coord(shape, flatten=True):
    """生成坐标网格"""
    h, w = shape
    coords_h = torch.linspace(-1, 1, h)
    coords_w = torch.linspace(-1, 1, w)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
    if flatten:
        coords = coords.reshape(-1, 2)
    return coords

def main():
    """主函数"""
    # 设置参数
    model_path = 'save/temp/epoch-best.pth'
    scale = 4
    
    # 数据集目录
    datasets = [
        'load/Set5',
        'load/Set14',
        'load/U100',
        'load/DIV2K_valid_HR'
        # 添加更多数据集
    ]
    
    # 检查哪些数据集存在
    available_datasets = []
    for dataset in datasets:
        if os.path.exists(dataset):
            available_datasets.append(dataset)
        else:
            print(f"警告: 数据集 {dataset} 不存在，将被跳过")
    
    if not available_datasets:
        print("错误: 没有找到可用的数据集")
        return
    
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型
    model = create_model(checkpoint)
    if model is None:
        print("无法创建模型，评估终止")
        return
    
    # 设置模型为评估模式
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 创建结果目录
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 评估所有数据集
    results = {}
    start_time = time.time()
    for dataset in available_datasets:
        dataset_name = os.path.basename(dataset)
        avg_psnr = evaluate_dataset(model, dataset, scale, device)
        results[dataset_name] = avg_psnr
    
    # 保存评估结果
    end_time = time.time()
    total_time = end_time - start_time
    
    # 写入评估报告
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f"evaluation_report_{timestamp}.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 超分辨率模型评估报告\n\n")
        f.write(f"## 模型: {os.path.basename(model_path)}\n\n")
        
        f.write("## 评估设置\n\n")
        f.write(f"- 模型路径: `{model_path}`\n")
        f.write(f"- 超分辨率比例: {scale}x\n")
        f.write(f"- 评估设备: {device}\n")
        f.write(f"- 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 评估结果\n\n")
        f.write("| 数据集 | PSNR (dB) |\n")
        f.write("|--------|----------|\n")
        
        total_psnr = 0
        valid_count = 0
        
        for dataset_name, psnr in results.items():
            if psnr > 0:
                f.write(f"| {dataset_name} | {psnr:.2f} |\n")
                total_psnr += psnr
                valid_count += 1
            else:
                f.write(f"| {dataset_name} | 评估失败 |\n")
        
        if valid_count > 0:
            avg_psnr = total_psnr / valid_count
            f.write(f"| **平均** | **{avg_psnr:.2f}** |\n\n")
        
        f.write("## 结论\n\n")
        if valid_count > 0:
            f.write(f"模型在 {valid_count} 个数据集上的平均PSNR为 **{avg_psnr:.2f} dB**。\n\n")
        else:
            f.write("所有评估均失败，无法得出结论。\n\n")
        
        f.write(f"总评估时间: {total_time:.2f}秒\n")
    
    print(f"\n评估完成！报告已保存到: {report_path}")

if __name__ == "__main__":
    main() 