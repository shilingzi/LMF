import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from models import make as make_model
from utils import make_coord
from datasets import make as make_data_loader
from utils import calc_psnr

def evaluate_model(model_path, config_path, dataset_path, scale=4):
    """
    在指定数据集上评估模型性能
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 修改验证集路径
    config['val_dataset']['dataset']['args']['root_path'] = dataset_path
    
    # 创建数据加载器
    val_loader = make_data_loader(config['val_dataset'])
    
    # 加载模型
    model = make_model(config['model']).cuda()
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    
    # 评估
    psnr_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'评估 {os.path.basename(dataset_path)}'):
            inp = batch['inp'].cuda()
            gt = batch['gt'].cuda()
            
            # 生成坐标
            coord = make_coord((inp.shape[-2] * scale, inp.shape[-1] * scale)).cuda()
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / (inp.shape[-2] * scale)
            cell[:, 1] *= 2 / (inp.shape[-1] * scale)
            
            # 前向传播
            pred = model(inp, coord, cell)
            pred = pred.view(-1, scale, scale, 3).permute(0, 3, 1, 2)
            
            # 计算PSNR
            psnr = calc_psnr(pred, gt, scale=scale)
            psnr_list.append(psnr.item())
    
    return np.mean(psnr_list)

def main():
    # 模型路径
    model_path = 'save/swinir-b_lm-lmlte_new/epoch-best.pth'
    
    # 配置文件路径
    config_path = 'configs/train-lmf/train_swinir-baseline-lmlte_small.yaml'
    
    # 数据集路径列表
    dataset_paths = [
        './load/Set5',
        './load/Set14',
        './load/U100',
        './load/DIV2K_valid_HR'
    ]
    
    # 评估结果
    results = {}
    model_name = os.path.basename(os.path.dirname(model_path))
    results[model_name] = {}
    
    # 对每个数据集进行评估
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        try:
            psnr = evaluate_model(model_path, config_path, dataset_path)
            results[model_name][dataset_name] = psnr
            print(f"数据集 {dataset_name}: PSNR = {psnr:.2f} dB")
        except Exception as e:
            print(f"评估数据集 {dataset_name} 时出错: {str(e)}")
    
    # 打印结果
    print("\n评估结果汇总:")
    print("-" * 50)
    for dataset_name, psnr in results[model_name].items():
        print(f"{dataset_name}: {psnr:.2f} dB")
    print("-" * 50)
    
    # 保存结果到文件
    with open('swinir_evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("SwinIR-LTE 模型评估结果:\n")
        f.write("-" * 50 + "\n")
        for dataset_name, psnr in results[model_name].items():
            f.write(f"{dataset_name}: {psnr:.2f} dB\n")
        f.write("-" * 50 + "\n")

if __name__ == '__main__':
    main() 