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

def evaluate_model(model_path, config_path, dataset_path, scale=4, strict_load=False, debug_weights=False):
    """
    在指定数据集上评估模型性能
    
    Args:
        model_path: 模型权重文件路径
        config_path: 配置文件路径
        dataset_path: 数据集路径
        scale: 放大比例
        strict_load: 是否严格加载模型权重
        debug_weights: 是否打印权重信息进行调试
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
        
        # 创建数据加载器 - 使用完整的数据集配置
        val_dataset = make_data_loader(config['val_dataset']['dataset'])
        
        # 应用wrapper
        if 'wrapper' in config['val_dataset']:
            from datasets.wrappers import make as make_wrapper
            val_dataset = make_wrapper(config['val_dataset']['wrapper'], val_dataset)
        
        # 创建dataloader
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config['val_dataset'].get('batch_size', 1),
            num_workers=0
        )
        
        print(f"数据加载器创建成功")
        
        # 加载模型
        model = make_model(config['model']).cuda()
        
        # 加载模型权重
        state_dict = torch.load(model_path)['model']
        
        # 调试模式 - 打印权重信息
        if debug_weights:
            print(f"模型权重键名数量: {len(state_dict.keys())}")
            print(f"模型权重键名前10个: {list(state_dict.keys())[:10]}")
            
            # 获取模型当前参数键名
            model_keys = set(model.state_dict().keys())
            weights_keys = set(state_dict.keys())
            
            # 分析缺失的键
            missing_keys = model_keys - weights_keys
            unexpected_keys = weights_keys - model_keys
            
            print(f"模型期望的键但权重文件中缺失的数量: {len(missing_keys)}")
            if len(missing_keys) > 0:
                print(f"缺失的键前5个: {list(missing_keys)[:5]}")
            
            print(f"权重文件中有但模型不需要的键数量: {len(unexpected_keys)}")
            if len(unexpected_keys) > 0:
                print(f"多余的键前5个: {list(unexpected_keys)[:5]}")
        
        # 非严格加载模式
        model.load_state_dict(state_dict, strict=strict_load)
        model.eval()
        
        print(f"模型加载成功: {model_path}")
        print(f"模型类型: {type(model).__name__}")
        
        # 评估
        psnr_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f'评估 {os.path.basename(dataset_path)}')):
                try:
                    # 检查输入类型
                    if isinstance(batch, dict) and 'inp' in batch and isinstance(batch['inp'], torch.Tensor):
                        inp = batch['inp'].cuda()
                        
                        if not isinstance(inp, torch.Tensor):
                            print(f"警告: 第{batch_idx}个batch的输入不是张量, 类型为: {type(inp)}")
                            continue
                        
                        # 检查是否有ground truth
                        if 'gt' in batch and isinstance(batch['gt'], torch.Tensor):
                            gt = batch['gt'].cuda()
                        else:
                            print(f"警告: 第{batch_idx}个batch没有ground truth或类型不是张量")
                            continue
                        
                        # 生成坐标
                        if 'coord' in batch and isinstance(batch['coord'], torch.Tensor):
                            coord = batch['coord'].cuda()
                        else:
                            coord = make_coord((inp.shape[-2] * scale, inp.shape[-1] * scale)).cuda()
                        
                        # 生成单元格
                        if 'cell' in batch and isinstance(batch['cell'], torch.Tensor):
                            cell = batch['cell'].cuda()
                        else:
                            cell = torch.ones_like(coord)
                            cell[:, 0] *= 2 / (inp.shape[-2] * scale)
                            cell[:, 1] *= 2 / (inp.shape[-1] * scale)
                        
                        # 前向传播
                        pred = model(inp, coord, cell)
                        
                        # 重塑预测结果
                        if pred.dim() == 3:  # B x HW x C
                            pred = pred.view(-1, scale, scale, 3).permute(0, 3, 1, 2)
                        
                        # 打印输入和输出形状进行调试
                        print(f"输入形状: {inp.shape}, 预测形状: {pred.shape}, GT形状: {gt.shape}")
                        
                        # 计算PSNR
                        psnr = calc_psnr(pred, gt, scale=scale)
                        psnr_list.append(psnr.item())
                    else:
                        print(f"警告: 第{batch_idx}个batch格式不正确")
                        continue
                except Exception as e:
                    print(f"处理第{batch_idx}个batch时出错: {str(e)}")
                    traceback.print_exc()
                    continue
        
        if psnr_list:
            avg_psnr = np.mean(psnr_list)
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
    model_path = 'save/swinir-b_lm-lmlte_new/epoch-best.pth'
    
    # 配置文件路径
    config_path = 'configs/train-lmf/train_swinir-baseline-lmlte_small.yaml'
    
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
    
    # 首先打印一次模型权重信息进行调试
    print("正在进行模型权重调试...")
    try:
        evaluate_model(model_path, config_path, valid_datasets[0], debug_weights=True, strict_load=False)
        print("模型权重调试完成")
    except Exception as e:
        print(f"模型权重调试出错: {str(e)}")
    
    # 评估结果
    results = {}
    model_name = os.path.basename(os.path.dirname(model_path))
    results[model_name] = {}
    
    # 对每个数据集进行评估 - 使用非严格加载模式
    for dataset_path in valid_datasets:
        dataset_name = os.path.basename(dataset_path)
        try:
            print(f"\n开始评估数据集: {dataset_name}")
            psnr = evaluate_model(model_path, config_path, dataset_path, strict_load=False)
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
        with open('swinir_evaluation_results.txt', 'w', encoding='utf-8') as f:
            f.write("SwinIR-LTE 模型评估结果:\n")
            f.write("-" * 50 + "\n")
            for dataset_name, psnr in results[model_name].items():
                f.write(f"{dataset_name}: {psnr:.2f} dB\n")
            f.write("-" * 50 + "\n")
    else:
        print("没有成功评估任何数据集")

if __name__ == '__main__':
    main() 