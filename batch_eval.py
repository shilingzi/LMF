import os
import argparse
import time
import yaml
from datetime import datetime
import re
import importlib.util
import sys

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

def main():
    parser = argparse.ArgumentParser(description="批量评估超分辨率模型")
    parser.add_argument("--model_path", default="save/swinir-b_lm-lmlte_new/epoch-best.pth", help="模型权重路径")
    parser.add_argument("--config_path", default="configs/train-lmf/train_swinir-baseline-lmlte_small.yaml", help="模型配置文件路径")
    parser.add_argument("--scale", type=int, default=4, help="超分辨率比例")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    # 检查模型和配置文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件 {args.config_path} 不存在")
        return
    
    # 定义要评估的数据集
    datasets = [
        "./load/Set5",
        "./load/Set14",
        "./load/B100",  # BSD100
        "./load/Urban100",
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
    
    print(f"将在以下数据集上评估模型:")
    for dataset in valid_datasets:
        print(f"- {dataset}")
    
    # 创建结果目录
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取当前时间作为评估ID
    eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(results_dir, f"evaluation_summary_{eval_id}.md")
    
    # 运行评估
    results = []
    for dataset in valid_datasets:
        dataset_name, psnr, output = run_evaluation(
            args.model_path, args.config_path, dataset, args.scale, args.debug
        )
        
        results.append({
            "dataset": dataset_name,
            "psnr": psnr,
            "output": output
        })
        
        # 保存详细输出
        output_file = os.path.join(results_dir, f"{dataset_name}_output_{eval_id}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
    
    # 生成评估摘要
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# 超分辨率模型评估摘要\n\n")
        
        # 添加模型和配置信息
        f.write("## 评估设置\n\n")
        f.write(f"- 模型路径: `{args.model_path}`\n")
        f.write(f"- 配置文件: `{args.config_path}`\n")
        f.write(f"- 超分辨率比例: {args.scale}x\n")
        f.write(f"- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        try:
            # 尝试读取配置文件内容
            with open(args.config_path, 'r') as cf:
                config = yaml.safe_load(cf)
                f.write("## 模型配置\n\n")
                f.write("```yaml\n")
                f.write(f"model: {config.get('model', {})}\n")
                f.write("```\n\n")
        except Exception as e:
            f.write(f"无法读取配置文件: {str(e)}\n\n")
        
        # 添加PSNR结果表
        f.write("## 评估结果\n\n")
        f.write("| 数据集 | PSNR (dB) |\n")
        f.write("|--------|----------|\n")
        
        total_psnr = 0
        valid_count = 0
        
        for result in results:
            psnr = result["psnr"]
            if psnr > 0:
                f.write(f"| {result['dataset']} | {psnr:.2f} |\n")
                total_psnr += psnr
                valid_count += 1
            else:
                f.write(f"| {result['dataset']} | 评估失败 |\n")
        
        if valid_count > 0:
            avg_psnr = total_psnr / valid_count
            f.write(f"| **平均** | **{avg_psnr:.2f}** |\n\n")
        
        # 添加结论
        f.write("## 结论\n\n")
        if valid_count > 0:
            f.write(f"模型在 {valid_count} 个数据集上的平均PSNR为 **{avg_psnr:.2f} dB**。\n\n")
        else:
            f.write("所有评估均失败，无法得出结论。\n\n")
        
        # 添加详细输出文件链接
        f.write("## 详细输出\n\n")
        for result in results:
            output_file = f"{result['dataset']}_output_{eval_id}.txt"
            f.write(f"- [{result['dataset']}]({output_file})\n")
    
    print(f"\n评估完成！摘要已保存到: {summary_file}")

if __name__ == "__main__":
    main() 