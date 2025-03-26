import os
import yaml
import torch
import numpy as np
import argparse
import traceback
from tqdm import tqdm
from models import make as make_model
from utils import make_coord
from utils import calc_psnr
from PIL import Image
import torchvision.transforms as transforms
import math

def load_image(path):
    """加载一张图像并转换为张量"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # 添加批次维度

def resize_fn(img, size):
    """调整图像大小"""
    return transforms.Resize(size, antialias=True)(img)

def make_coord(shape, flatten=True):
    """生成坐标网格"""
    h, w = shape
    coords_h = torch.linspace(-1, 1, h)
    coords_w = torch.linspace(-1, 1, w)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
    if flatten:
        coords = coords.reshape(-1, 2)
    return coords

def evaluate_on_image(model, img_hr, scale=4, debug=False):
    """在单个高分辨率图像上评估模型"""
    device = next(model.parameters()).device
    
    # 调整为低分辨率
    h, w = img_hr.shape[-2], img_hr.shape[-1]
    img_lr = resize_fn(img_hr, (h // scale, w // scale))
    
    # 准备坐标
    coord = make_coord((h, w)).to(device)
    coord = coord.unsqueeze(0)  # 添加批次维度
    
    # 准备cell
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / h
    cell[:, :, 1] *= 2 / w
    
    if debug:
        print(f"高分辨率图像形状: {img_hr.shape}")
        print(f"低分辨率图像形状: {img_lr.shape}")
        print(f"坐标形状: {coord.shape}")
        print(f"单元格形状: {cell.shape}")
    
    # 移动到设备
    img_lr = img_lr.to(device)
    
    try:
        # LMLTE模型期望输入格式为(B, h*w, 3)，但SwinIR编码器期望(B, 3, h, w)
        # 因此，我们保持img_lr的4D格式不变，然后在模型内部处理转换
        
        # 添加特征坐标
        feat_h, feat_w = img_lr.shape[-2], img_lr.shape[-1]
        feat_coord = make_coord((feat_h, feat_w), flatten=False).to(device)
        feat_coord = feat_coord.unsqueeze(0)
        
        if debug:
            print(f"特征坐标形状: {feat_coord.shape}")
        
        # 前向传播
        with torch.no_grad():
            # 创建一个额外的维度来匹配模型期望的cell格式
            cell_reshaped = cell.view(1, h * w, 1, 2)
            if debug:
                print(f"重排后的cell形状: {cell_reshaped.shape}")
            
            # 使用辅助函数处理模型调用
            pred = model_forward(model, img_lr, coord, cell_reshaped, feat_coord, debug)
            
        # 计算PSNR
        pred = pred.view(1, h, w, 3).permute(0, 3, 1, 2)
        mse = ((pred - img_hr) ** 2).mean().item()
        psnr = -10 * np.log10(mse)
        return psnr
    except Exception as e:
        if debug:
            print(f"模型前向传播时出错: {str(e)}")
            traceback.print_exc()
        raise e

def model_forward(model, img_lr, coord, cell, feat_coord, debug=False):
    """辅助函数，处理模型的前向传播"""
    # 我们检查模型的forward函数，确定所需的输入格式
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'check_image_size'):
        # 这是LMLTE模型，需要特殊处理
        if debug:
            print("检测到LMLTE模型，使用特殊处理")
        
        # 对于LMLTE模型，我们需要将输入正确地格式化
        # 1. 保持img_lr为4D张量(B,C,H,W)供编码器使用
        # 2. 在模型内部，编码器会处理4D张量
        # 3. 然后在需要时将编码器输出与坐标和cell一起处理
        
        # 对LMLTE定制的forward调用
        return custom_lmlte_forward(model, img_lr, coord, cell, feat_coord, debug)
    else:
        # 默认情况，直接调用模型
        return model(img_lr, coord, cell)

def custom_lmlte_forward(model, img_lr, coord, cell, feat_coord, debug=False):
    """为LMLTE模型定制的前向传播函数"""
    device = img_lr.device
    
    # 获取编码器的特征
    with torch.no_grad():
        # 1. 使用编码器处理4D张量
        feat = model.encoder(img_lr)
        if debug:
            print(f"编码器输出特征形状: {feat.shape}")
        
        # 2. 设置模型的内部状态
        # 将img_lr转换为(B, h*w, 3)格式，这是模型期望的输入格式
        bs, c, h, w = img_lr.shape
        model.inp = img_lr
        model.feat = feat
        
        # 初始化query_bsize
        if not hasattr(model, 'query_bsize') or model.query_bsize is None:
            # 使用默认值
            model.query_bsize = int(2160 * 3840 * 0.5)
            model.query_bsize = math.ceil(coord.shape[1] / math.ceil(coord.shape[1] / model.query_bsize))
            if debug:
                print(f"已初始化query_bsize: {model.query_bsize}")
        
        # 处理feat_coord，确保尺寸匹配
        # feat坐标应该与编码器输出特征的形状匹配
        feat_h, feat_w = feat.shape[-2], feat.shape[-1]
        model.feat_coord = make_coord((feat_h, feat_w), flatten=False).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        if debug:
            print(f"重新生成的feat_coord形状: {model.feat_coord.shape}")
        
        # 处理cell参数 - 需要将[B, HW, 1, 2]变为[B, HW, 2]
        # gen_modulations期望的cell格式是[B, HW, 2]
        cell_processed = cell.squeeze(2)  # 移除额外的维度
        if debug:
            print(f"处理后的cell形状: {cell_processed.shape}")
        
        # 初始化非局部特征 - 这是模型在gen_modulations中需要的
        # 检查模型是否启用了非局部注意力
        if model.non_local_attn:
            # 创建与特征相同大小的非局部特征张量
            bs, in_c, in_h, in_w = feat.shape
            # 注意: 如果CrossScaleAttention不可用，我们只创建一个空张量
            model.non_local_feat = torch.zeros(bs, model.non_local_attn_dim, in_h, in_w).to(device)
            
            # 如果调试模式开启，打印相关信息
            if debug:
                print(f"已初始化non_local_feat属性，形状为: {model.non_local_feat.shape}")
        
        # 3. 手动调用必要的函数来生成结果
        # 生成调制信号
        mod = model.gen_modulations(feat, cell_processed)
        if debug:
            print(f"调制信号形状: {mod.shape}")
        
        # 设置系数和频率
        if model.mod_input:
            model.coeff = mod[:, model.mod_dim:model.mod_dim + model.mod_coef_dim, :, :]
            model.freqq = mod[:, model.mod_dim + model.mod_coef_dim:, :, :]
            if debug:
                print(f"系数形状: {model.coeff.shape}")
                print(f"频率形状: {model.freqq.shape}")
        
        # 进行查询 - 这里使用原始的cell
        # 确保query_bsize已设置
        out = model.batched_query_rgb(model.coeff, model.freqq, mod, coord, cell_processed, model.query_bsize)
        if debug:
            print(f"输出形状: {out.shape}")
        
        return out

def evaluate_on_folder(model, folder_path, scale=4, debug=False):
    """在文件夹中的所有图像上评估模型"""
    device = next(model.parameters()).device
    model.eval()
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if debug:
        print(f"找到 {len(image_files)} 张图像")
    
    total_psnr = 0
    count = 0
    
    # 对每个图像评估
    with torch.no_grad():
        for img_file in tqdm(image_files, desc=f"评估 {os.path.basename(folder_path)}"):
            try:
                if debug:
                    print(f"处理图像: {img_file}")
                img_path = os.path.join(folder_path, img_file)
                img = load_image(img_path).to(device)
                
                psnr = evaluate_on_image(model, img, scale, debug)
                total_psnr += psnr
                count += 1
                
                if debug:
                    print(f"{img_file}: PSNR = {psnr:.2f} dB")
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {str(e)}")
    
    # 计算平均PSNR
    avg_psnr = total_psnr / count if count > 0 else 0
    return avg_psnr

def main():
    parser = argparse.ArgumentParser(description='简单模型评估工具')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--config_path', type=str, required=True, help='模型配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./load/Set5', help='测试数据集目录')
    parser.add_argument('--scale', type=int, default=4, help='超分辨率缩放因子')
    parser.add_argument('--save_results', action='store_true', help='是否保存结果')
    parser.add_argument('--debug', action='store_true', help='是否打印调试信息')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在: {args.config_path}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return
    
    # 加载配置
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        print(f"配置文件加载成功: {args.config_path}")
        
        if args.debug:
            print(f"模型配置: {config['model']}")
    except Exception as e:
        print(f"加载配置文件时出错: {str(e)}")
        return
    
    # 加载模型
    try:
        model = make_model(config['model']).cuda()
        
        if args.debug:
            print(f"模型类型: {type(model).__name__}")
        
        # 检查模型权重
        state_dict = torch.load(args.model_path, map_location='cuda')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        if args.debug:
            print(f"权重键数量: {len(state_dict.keys())}")
            if len(state_dict.keys()) > 0:
                print(f"权重键示例: {list(state_dict.keys())[:3]}")
        
        # 尝试非严格加载
        try:
            model.load_state_dict(state_dict)
            print("模型权重成功加载 (严格模式)")
        except Exception as e:
            print(f"严格加载模式失败, 尝试非严格加载: {str(e)}")
            model.load_state_dict(state_dict, strict=False)
            print("模型权重成功加载 (非严格模式)")
        
        model.eval()
        print(f"模型加载成功: {args.model_path}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        traceback.print_exc()
        return
    
    # 评估
    try:
        avg_psnr = evaluate_on_folder(model, args.data_dir, args.scale, args.debug)
        print(f"\n数据集 {os.path.basename(args.data_dir)}: 平均 PSNR = {avg_psnr:.2f} dB")
        
        # 保存结果
        if args.save_results:
            results_dir = 'evaluation_results'
            os.makedirs(results_dir, exist_ok=True)
            
            model_name = os.path.basename(os.path.dirname(args.model_path))
            dataset_name = os.path.basename(args.data_dir)
            
            with open(os.path.join(results_dir, f"{model_name}_results.txt"), 'a', encoding='utf-8') as f:
                f.write(f"{dataset_name}: {avg_psnr:.2f} dB\n")
                
            print(f"结果已保存到 {os.path.join(results_dir, f'{model_name}_results.txt')}")
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main() 