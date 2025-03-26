# LMF - 本地建模场（Local Modeling Field）

LMF是一个用于图像超分辨率的深度学习框架，它基于隐式神经表示（Implicit Neural Representation）技术，通过本地建模场方法实现高质量的图像超分辨率处理。

## 项目特点

- 基于隐式神经表示的超分辨率方法
- 支持任意比例的超分辨率处理
- 高质量的超分辨率重建效果
- 提供了基于Flask的Web应用界面
- 支持多种backbone网络（EDSR、RDN、RCAN、SwinIR等）
- 提供了多种超分辨率模型（LIIF、LTE、CiaoSR等及其改进版本）

## 安装与环境配置

### 依赖项

```
torch>=1.9.1
numpy>=1.19
tensorboardX>=2.1
pyyaml>=5.3
tqdm>=4.46
torchvision>=0.9
Pillow>=8.0
imageio>=2.9
timm>=0.3.2
```

### 安装步骤

1. 克隆此仓库：
```bash
git clone https://github.com/your-username/lmf.git
cd lmf
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

将训练和验证数据集放置于`load/`目录下：

- DIV2K训练集：`load/DIV2K_train_HR/`
- DIV2K验证集：`load/DIV2K_valid_HR/`

## 模型训练

使用以下命令开始训练模型：

```bash
python train.py --config configs/train-lmf/train_edsr-baseline-lmlte.yaml --gpu 0
```

配置文件允许您设置不同的训练参数，如backbone网络、批量大小、学习率等。

## 模型测试

测试模型性能：

```bash
python test.py --config configs/test-lmf/test-div2k.yaml --model save/edsr-b_lm-lmlte/epoch-best.pth --gpu 0
```

## 演示

使用以下命令对单张图像进行超分辨率处理：

```bash
python demo.py --input input.png --model save/edsr-b_lm-lmlte/epoch-best.pth --scale 4 --output output.png --gpu 0
```

参数说明：
- `--input`：输入低分辨率图像的路径
- `--model`：训练好的模型路径
- `--scale`：放大倍数
- `--output`：输出图像的保存路径
- `--gpu`：指定使用的GPU ID
- `--fast`：是否使用快速模式（默认为True）

## Web应用

项目提供了基于Flask的Web应用，可以通过浏览器上传图像并进行超分辨率处理：

```bash
python app.py
```

启动后，访问 http://localhost:5000 即可使用Web界面。

## 项目结构

```
├── app.py                   # Flask Web应用
├── configs/                 # 配置文件目录
│   ├── train-lmf/           # LMF训练配置
│   ├── train-original/      # 原始模型训练配置
│   ├── test-lmf/            # LMF测试配置
│   └── test-original/       # 原始模型测试配置
├── datasets/                # 数据集处理模块
├── demo.py                  # 单图像演示脚本
├── init_cmsr.py             # CMSR初始化脚本
├── models/                  # 模型定义
│   ├── edsr.py              # EDSR模型
│   ├── rdn.py               # RDN模型
│   ├── rcan.py              # RCAN模型
│   ├── swinir.py            # SwinIR模型
│   ├── liif.py              # LIIF模型
│   ├── lte.py               # LTE模型
│   ├── ciaosr.py            # CiaoSR模型
│   ├── lmliif.py            # LMF-LIIF模型
│   ├── lmlte.py             # LMF-LTE模型
│   └── lmciaosr.py          # LMF-CiaoSR模型
├── requirements.txt         # 依赖项列表
├── save/                    # 保存的模型目录
├── scripts/                 # 实用脚本
├── templates/               # Web应用模板
├── test.py                  # 测试脚本
├── train.py                 # 训练脚本
└── utils.py                 # 工具函数
```

## 引用

如果您在研究中使用了本项目，请考虑引用相关论文。

## 许可证

本项目基于[LICENSE](LICENSE)许可证开源。 