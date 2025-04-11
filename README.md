# LMF (Local Modulation Framework)

这是一个基于深度学习的超分辨率图像处理项目，主要实现了基于SwinIR的LMLTE (Local Modulation Lite Transformer Enhancement) 模型。

## 项目结构

```
.
├── configs/             # 配置文件目录
│   ├── train-lmf/       # 训练配置
│   └── test-lmf/        # 测试配置
├── datasets/            # 数据集处理模块
├── models/              # 模型定义
│   ├── arch_ciaosr/     # CIAOSR架构相关模块
│   ├── lmlte.py         # LMLTE模型实现
│   ├── swinir.py        # SwinIR模型实现
│   └── ...              # 其他模型文件
├── scripts/             # 辅助脚本
├── load/                # 数据集目录（需自行下载）
│   ├── Set5/            # Set5测试集
│   ├── Set14/           # Set14测试集
│   └── DIV2K_valid_HR/  # DIV2K验证集
├── save/                # 模型保存目录
├── eval_simple.py       # 单数据集评估脚本
├── batch_eval.py        # 批量评估脚本
├── train_windows.py     # Windows版训练脚本
└── ...
```

## 环境配置

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 模型评估

### 单数据集评估

使用`eval_simple.py`脚本可以对单个数据集进行评估：

```bash
python eval_simple.py --model_path save/swinir-b_lm-lmlte_new/epoch-best.pth --config_path configs/train-lmf/train_swinir-baseline-lmlte_small.yaml --data_dir ./load/Set5 --scale 4
```

参数说明：
- `--model_path`: 模型权重文件路径
- `--config_path`: 模型配置文件路径
- `--data_dir`: 评估数据集目录
- `--scale`: 超分辨率比例（默认为4）
- `--debug`: 启用调试模式（可选）

### 批量评估

使用`batch_eval.py`脚本可以对多个数据集进行批量评估：

```bash
python batch_eval.py
```

该脚本将自动评估`./load/`目录下的所有可用数据集，包括Set5、Set14、DIV2K_valid_HR等。

参数说明（可选）：
- `--model_path`: 模型权重文件路径
- `--config_path`: 模型配置文件路径
- `--scale`: 超分辨率比例（默认为4）
- `--debug`: 启用调试模式（可选）

评估完成后，结果将保存在`evaluation_results`目录中。

## 训练模型

```bash
python train_windows.py --config configs/train-lmf/train_swinir-baseline-lmlte_small.yaml
```

或使用批处理脚本：

```bash
train_swinir.bat
```

## 常见问题

1. 如果遇到CUDA相关错误，请检查您的GPU驱动和CUDA版本是否兼容。

2. 如果遇到内存不足错误，可以尝试减小批量大小或使用更小的模型。

3. 如果在评估过程中出现模型加载错误，请确保使用的配置文件与训练时使用的配置文件匹配。

## 许可证

请参见LICENSE文件。 