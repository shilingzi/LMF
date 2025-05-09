import os
import subprocess
import sys

def main():
    # 配置训练参数
    config_file = 'configs/train-lmf/train_swinir-baseline-lmlte_small.yaml'
    gpu_id = '0'
    save_path = 'save/swinir-b_lm-lmlte_new'
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 构建训练命令
    cmd = []
    
    # 如果当前环境不是lian，则激活lian环境
    if not os.environ.get("CONDA_PREFIX", "").endswith("lian"):
        # 在Windows上使用conda的激活脚本
        if sys.platform == "win32":
            cmd = [
                "conda", "run", "-n", "lian",
                "python", "train_windows.py",
                "--config", config_file,
                "--gpu", gpu_id,
                "--save_path", save_path
            ]
        else:
            # 在Linux/Mac上的命令
            cmd = [
                "/bin/bash", "-c",
                f"conda activate lian && python train_windows.py --config {config_file} --gpu {gpu_id} --save_path {save_path}"
            ]
    else:
        # 已经在lian环境中
        cmd = [
            "python", "train_windows.py",
            "--config", config_file,
            "--gpu", gpu_id,
            "--save_path", save_path
        ]
    
    # 执行训练
    print(f"开始训练 SwinIR-LTE 模型...")
    print(f"配置文件: {config_file}")
    print(f"GPU ID: {gpu_id}")
    print(f"保存路径: {save_path}")
    print(f"使用环境: lian")
    
    subprocess.run(cmd)
    
    print("训练完成!")

if __name__ == "__main__":
    main()
