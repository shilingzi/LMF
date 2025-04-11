import os
import argparse
import time
import yaml
from datetime import datetime
import re
import importlib.util
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_module_from_file(file_path, module_name):
    """从文件加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_evaluation(model_path, config_path, data_dir, scale=4, debug=False):
    """运行单个评估任务"""
    print(f"评估数据集: {os.path.basename(data_dir)}")
    print(f"模型路径: {model_path}")
    print(f"配置文件: {config_path}")
    
    # 加载eval_simple模块
    eval_module = load_module_from_file("eval_simple.py", "eval_simple")
    
    # 直接在当前进程中运行评估
    start_time = time.time()
    
    # 设置命令行参数
    sys.argv = [
        "eval_simple.py",
        "--model_path", model_path,
        "--config_path", config_path,
        "--data_dir", data_dir,
        "--scale", str(scale)
    ]
    
    if debug:
        sys.argv.append("--debug")
    
    # 捕获输出
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            # 运行评估
            if hasattr(eval_module, 'main'):
                eval_module.main()
            else:
                print("评估模块中没有main函数")
                return os.path.basename(data_dir), 0.0, "评估模块中没有main函数"
        except Exception as e:
            error_msg = f"评估过程中出错: {str(e)}"
            print(error_msg)
            return os.path.basename(data_dir), 0.0, error_msg
    
    output = f.getvalue()
    end_time = time.time()
    
    # 从输出中提取PSNR值
    psnr_match = re.search(r"平均\s*PSNR\s*=\s*([\d.]+)", output)
    
    if psnr_match:
        psnr = float(psnr_match.group(1))
        dataset_name = os.path.basename(data_dir)
        print(f"数据集 {dataset_name} 评估完成，PSNR: {psnr:.2f} dB，耗时: {end_time - start_time:.2f}秒")
        return dataset_name, psnr, output
    else:
        print(f"警告: 在输出中未找到PSNR值")
        return os.path.basename(data_dir), 0.0, output

def compare_encoders(encoders, datasets, scale=4, debug=False):
    """比较不同编码器的性能"""
    results = {}
    all_results = {}
    
    # 创建结果目录
    results_dir = "comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取当前时间作为评估ID
    eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 对每个编码器进行评估
    for encoder_name, model_path, config_path in encoders:
        print(f"\n正在评估编码器: {encoder_name}")
        encoder_results = {}
        
        # 检查模型和配置文件
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            continue
        
        if not os.path.exists(config_path):
            print(f"错误: 配置文件 {config_path} 不存在")
            continue
        
        # 在每个数据集上评估
        for dataset in datasets:
            if not os.path.exists(dataset):
                print(f"警告: 数据集目录 {dataset} 不存在，跳过")
                continue
            
            dataset_name, psnr, output = run_evaluation(
                model_path, config_path, dataset, scale, debug
            )
            
            encoder_results[dataset_name] = psnr
            
            # 保存详细输出
            output_file = os.path.join(results_dir, f"{encoder_name}_{dataset_name}_output_{eval_id}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
        
        results[encoder_name] = encoder_results
        
    # 生成对比报告
    summary_file = os.path.join(results_dir, f"comparison_summary_{eval_id}.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# 不同编码器的LMLTE模型性能对比\n\n")
        
        # 添加评估设置
        f.write("## 评估设置\n\n")
        f.write(f"- 超分辨率比例: {scale}x\n")
        f.write(f"- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 添加PSNR结果表
        f.write("## 评估结果 (PSNR in dB)\n\n")
        f.write("| 数据集 |")
        for encoder_name, _, _ in encoders:
            f.write(f" {encoder_name} |")
        f.write("\n|--------|")
        for _ in encoders:
            f.write("----------|")
        f.write("\n")
        
        # 计算每个编码器的平均性能
        encoder_averages = {encoder_name: 0.0 for encoder_name, _, _ in encoders}
        encoder_counts = {encoder_name: 0 for encoder_name, _, _ in encoders}
        
        # 添加每个数据集的结果
        for dataset in datasets:
            dataset_name = os.path.basename(dataset)
            if all(dataset_name in results[encoder_name] for encoder_name, _, _ in encoders):
                f.write(f"| {dataset_name} |")
                for encoder_name, _, _ in encoders:
                    psnr = results[encoder_name][dataset_name]
                    f.write(f" {psnr:.2f} |")
                    encoder_averages[encoder_name] += psnr
                    encoder_counts[encoder_name] += 1
                f.write("\n")
        
        # 添加平均性能
        f.write("| **平均** |")
        for encoder_name, _, _ in encoders:
            if encoder_counts[encoder_name] > 0:
                avg_psnr = encoder_averages[encoder_name] / encoder_counts[encoder_name]
                f.write(f" **{avg_psnr:.2f}** |")
            else:
                f.write(" N/A |")
        f.write("\n\n")
        
        # 添加结论
        f.write("## 结论\n\n")
        
        # 寻找最佳编码器
        best_encoder = None
        best_psnr = 0.0
        
        for encoder_name, _, _ in encoders:
            if encoder_counts[encoder_name] > 0:
                avg_psnr = encoder_averages[encoder_name] / encoder_counts[encoder_name]
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_encoder = encoder_name
        
        if best_encoder:
            f.write(f"在本次比较中，**{best_encoder}**表现最好，平均PSNR达到了**{best_psnr:.2f} dB**。\n\n")
        
        # 添加详细分析
        f.write("### 编码器对比分析\n\n")
        f.write("各编码器的特点：\n\n")
        f.write("- **EDSR**：深度残差网络，通过大量残差块和跳跃连接提取特征，结构简单但效果强大。\n")
        f.write("- **SwinIR**：基于Transformer的架构，通过自注意力机制捕捉长距离依赖，全局建模能力更强。\n\n")
        
        # 添加详细输出文件链接
        f.write("## 详细输出\n\n")
        for encoder_name, _, _ in encoders:
            f.write(f"### {encoder_name} 编码器\n\n")
            for dataset in datasets:
                dataset_name = os.path.basename(dataset)
                output_file = f"{encoder_name}_{dataset_name}_output_{eval_id}.txt"
                if os.path.exists(os.path.join(results_dir, output_file)):
                    f.write(f"- [{dataset_name}]({output_file})\n")
            f.write("\n")
    
    print(f"\n评估完成！对比摘要已保存到: {summary_file}")
    
    # 绘制对比图
    plot_comparison(results, os.path.join(results_dir, f"comparison_plot_{eval_id}.png"))
    
    return results

def plot_comparison(results, save_path):
    """绘制不同编码器的性能对比图"""
    datasets = set()
    for encoder_results in results.values():
        datasets.update(encoder_results.keys())
    datasets = sorted(list(datasets))
    
    encoder_names = list(results.keys())
    x = np.arange(len(datasets))
    width = 0.8 / len(encoder_names)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, encoder_name in enumerate(encoder_names):
        psnr_values = [results[encoder_name].get(dataset, 0) for dataset in datasets]
        ax.bar(x + i * width - 0.4 + width/2, psnr_values, width, label=encoder_name)
    
    ax.set_xlabel('数据集')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('不同编码器的LMLTE模型性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    # 添加数值标签
    for i, encoder_name in enumerate(encoder_names):
        for j, dataset in enumerate(datasets):
            if dataset in results[encoder_name]:
                psnr = results[encoder_name][dataset]
                ax.annotate(f'{psnr:.2f}', 
                            xy=(j + i * width - 0.4 + width/2, psnr), 
                            xytext=(0, 3),
                            textcoords="offset points", 
                            ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"对比图已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="比较不同编码器在LMLTE模型中的性能")
    parser.add_argument("--edsr_model", default="save/edsr-b_lm-lmlte/epoch-best.pth", help="EDSR模型路径")
    parser.add_argument("--swinir_model", default="save/swinir-b_lm-lmlte_new/epoch-best.pth", help="SwinIR模型路径")
    parser.add_argument("--scale", type=int, default=4, help="超分辨率比例")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    # 定义编码器列表：(名称, 模型路径, 配置文件路径)
    encoders = [
        ("EDSR", args.edsr_model, "configs/train-lmf/train_edsr-baseline-lmlte_small.yaml"),
        ("SwinIR", args.swinir_model, "configs/train-lmf/train_swinir-baseline-lmlte_small.yaml")
    ]
    
    # 定义要评估的数据集
    datasets = [
        "./load/Set5",
        "./load/Set14",
        "./load/DIV2K_valid_HR"
    ]
    
    # 检查数据集是否存在
    valid_datasets = []
    for dataset in datasets:
        if os.path.exists(dataset):
            valid_datasets.append(dataset)
        else:
            print(f"警告: 数据集目录 {dataset} 不存在，跳过")
    
    if not valid_datasets:
        print("错误: 没有找到有效的数据集目录")
        return
    
    # 比较编码器性能
    compare_encoders(encoders, valid_datasets, args.scale, args.debug)

if __name__ == "__main__":
    main() 