import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from models import make
from utils import make_coord
from datasets import make_data_loader
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
    model = make(config['model']).cuda()
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    
    # 评估
    psnr_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Evaluating on {os.path.basename(dataset_path)}'):
            inp = batch['inp'].cuda()
            gt = batch['gt'].cuda()
            
            # 生成坐标
            coord = make_coord((inp.shape[-2] * scale, inp.shape[-1] * scale))
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
    # 模型路径列表
    model_paths = [
        'save/edsr-b_lm-lmlte_new/epoch-best.pth',
        'save/rdn-b_lm-lmlte_new/epoch-best.pth',
        'save/swinir-b_lm-lmlte_new/epoch-best.pth'
    ]
    
    # 配置文件路径列表
    config_paths = [
        'configs/train-lmf/train_edsr-baseline-lmlte_small.yaml',
        'configs/train-lmf/train_rdn-baseline-lmlte_small.yaml',
        'configs/train-lmf/train_swinir-baseline-lmlte_small.yaml'
    ]
    
    # 数据集路径列表
    dataset_paths = [
        './load/Set5',
        './load/Set14',
        './load/DIV2K100'
    ]
    
    # 评估结果
    results = {}
    
    # 对每个模型进行评估
    for model_path, config_path in zip(model_paths, config_paths):
        model_name = os.path.basename(os.path.dirname(model_path))
        results[model_name] = {}
        
        # 对每个数据集进行评估
        for dataset_path in dataset_paths:
            dataset_name = os.path.basename(dataset_path)
            psnr = evaluate_model(model_path, config_path, dataset_path)
            results[model_name][dataset_name] = psnr
    
    # 打印结果
    print("\n评估结果:")
    print("-" * 50)
    for model_name, dataset_results in results.items():
        print(f"\n模型: {model_name}")
        for dataset_name, psnr in dataset_results.items():
            print(f"{dataset_name}: {psnr:.2f} dB")
        print("-" * 50)
    
    # 保存结果到文件
    with open('evaluation_results.txt', 'w') as f:
        f.write("评估结果:\n")
        f.write("-" * 50 + "\n")
        for model_name, dataset_results in results.items():
            f.write(f"\n模型: {model_name}\n")
            for dataset_name, psnr in dataset_results.items():
                f.write(f"{dataset_name}: {psnr:.2f} dB\n")
            f.write("-" * 50 + "\n")

if __name__ == '__main__':
    main() 