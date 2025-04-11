import os
import yaml
import torch
import numpy as np
import traceback
from tqdm import tqdm
from models import make as make_model
from utils import make_coord
from datasets import make as make_data_loader
from utils import calc_psnr

def evaluate_model(model_path, config_path, dataset_path, scale=4):
    """
    在指定数据集上评估LO模型性能，并添加调试信息
    """
    try:
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        print(f"配置文件加载成功: {config_path}")
        
        # 修改验证集路径
        config['val_dataset']['dataset']['args']['root_path'] = dataset_path
        
        print(f"数据集路径设置为: {dataset_path}")
        print(f"验证数据集配置: {config['val_dataset']}")
        
        # 创建数据加载器
        val_loader = make_data_loader(config['val_dataset'])
        
        print(f"数据加载器创建成功")
        
        # 加载模型
        model = make_model(config['model']).cuda()
        model.load_state_dict(torch.load(model_path)['model'])
        model.eval()
        
        print(f"模型加载成功: {model_path}")
        print(f"模型类型: {type(model).__name__}")
        
        # 评估
        psnr_list = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader, desc=f'评估 {os.path.basename(dataset_path)}')):
                # 打印当前处理的图像信息
                print(f"\n处理图像 {idx}:")
                
                inp = batch['inp'].cuda()
                gt = batch['gt'].cuda()
                
                print(f"输入尺寸: {inp.shape}")
                print(f"目标尺寸: {gt.shape}")
                
                # 生成坐标
                coord = make_coord((inp.shape[-2] * scale, inp.shape[-1] * scale)).cuda()
                cell = torch.ones_like(coord)
                cell[:, 0] *= 2 / (inp.shape[-2] * scale)
                cell[:, 1] *= 2 / (inp.shape[-1] * scale)
                
                print(f"坐标尺寸: {coord.shape}")
                print(f"单元尺寸: {cell.shape}")
                
                try:
                    # 前向传播
                    pred = model(inp, coord, cell)
                    print(f"初始预测尺寸: {pred.shape}")
                    
                    # 重塑预测结果
                    pred = pred.view(-1, scale, scale, 3).permute(0, 3, 1, 2)
                    print(f"重塑后预测尺寸: {pred.shape}")
                    
                    # 计算PSNR
                    psnr = calc_psnr(pred, gt, scale=scale)
                    psnr_value = psnr.item()
                    psnr_list.append(psnr_value)
                    print(f"PSNR: {psnr_value:.2f} dB")
                    
                    # 检查是否有异常值
                    if not (20 <= psnr_value <= 50):
                        print(f"警告: 检测到异常PSNR值: {psnr_value}")
                        
                except Exception as e:
                    print(f"处理图像 {idx} 时出错: {str(e)}")
                    traceback.print_exc()
                    continue
        
        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            print(f"\n平均PSNR: {avg_psnr:.2f} dB")
            print(f"最小PSNR: {np.min(psnr_list):.2f} dB")
            print(f"最大PSNR: {np.max(psnr_list):.2f} dB")
            return avg_psnr
        else:
            print("没有成功计算任何PSNR值")
            return 0.0
            
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        traceback.print_exc()
        raise e

def main():
    # 模型路径
    model_path = 'save/liif_lm-lmlte_liifb_lo/epoch-best.pth'
    
    # 配置文件路径
    config_path = 'configs/train-lmf/train_liifb-liif_plus_lo.yaml'
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    # 数据集路径列表
    dataset_paths = [
        './load/Set5',
        './load/Set14',
        './load/U100',
        './load/DIV2K_valid_HR'
    ]
    
    # 检查数据集是否存在
    valid_datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            valid_datasets.append(path)
        else:
            print(f"警告: 数据集路径不存在: {path}")
    
    if not valid_datasets:
        print("错误: 没有有效的数据集可用于评估")
        return
    
    # 评估结果
    results = {}
    model_name = os.path.basename(os.path.dirname(model_path))
    results[model_name] = {}
    
    # 对每个数据集进行评估
    for dataset_path in valid_datasets:
        dataset_name = os.path.basename(dataset_path)
        try:
            print(f"\n开始评估数据集: {dataset_name}")
            psnr = evaluate_model(model_path, config_path, dataset_path)
            results[model_name][dataset_name] = psnr
            print(f"数据集 {dataset_name}: PSNR = {psnr:.2f} dB")
        except Exception as e:
            print(f"评估数据集 {dataset_name} 时出错: {str(e)}")
    
    # 打印结果
    if results[model_name]:
        print("\n评估结果汇总:")
        print("-" * 50)
        for dataset_name, psnr in results[model_name].items():
            print(f"{dataset_name}: {psnr:.2f} dB")
        print("-" * 50)
        
        # 保存结果到文件
        with open('lo_evaluation_results.txt', 'w', encoding='utf-8') as f:
            f.write("LO模型评估结果:\n")
            f.write("-" * 50 + "\n")
            for dataset_name, psnr in results[model_name].items():
                f.write(f"{dataset_name}: {psnr:.2f} dB\n")
            f.write("-" * 50 + "\n")
    else:
        print("没有成功评估任何数据集")

if __name__ == '__main__':
    main() 