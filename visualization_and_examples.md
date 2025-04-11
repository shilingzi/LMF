# LMF 模型可视化与实例

## 1. 模型架构可视化

### 1.1 整体架构图

```
+----------------------+      +-------------------------+      +-----------------------+
|                      |      |                         |      |                       |
|   特征编码器         |      |  低辨率高维空间         |      |  高辨率低维空间       |
|  (Feature Encoder)   +----->+  (LR-HD)               +----->+  (HR-LD)              |
|                      |      |  潜在解码器             |      |  渲染解码器           |
|                      |      |                         |      |                       |
+----------------------+      +-------------------------+      +-----------------------+
        |                               |                               |
        v                               v                               v
+----------------------+      +-------------------------+      +-----------------------+
| • 提取特征            |      | • 生成潜在调制          |      | • 渲染高分辨率图像    |
| • SwinIR/EDSR/RDN    |      | • 跨尺度注意力机制      |      | • MLP网络             |
| • 无上采样操作        |      | • 局部集成              |      | • 调制机制            |
+----------------------+      +-------------------------+      +-----------------------+
```

### 1.2 SwinIR编码器结构

```
+-------------------+
| 输入低分辨率图像   |
+--------+----------+
         |
         v
+--------+----------+
| 浅层特征提取       |
| (3×3卷积)         |
+--------+----------+
         |
         v
+--------+----------+
| 残差Swin           |
| Transformer块 1    |
+--------+----------+
         |
         v
+--------+----------+
| 残差Swin           |
| Transformer块 2    |
+--------+----------+
         |
         v
       ...
         |
         v
+--------+----------+
| 残差Swin           |
| Transformer块 n    |
+--------+----------+
         |
         v
+--------+----------+
| 特征合并           |
| (跳跃连接)         |
+--------+----------+
         |
         v
+--------+----------+
| 输出特征图         |
+-------------------+
```

### 1.3 CrossScaleAttention示意图

```
                     +-------------------+
                     | 输入特征X          |
                     +--------+----------+
                              |
              +---------------+----------------+
              |                                |
              v                                v
+-------------+-------------+    +-------------+-------------+
| 下采样 (尺度1)            |    | 下采样 (尺度2)            |
+-------------+-------------+    +-------------+-------------+
              |                                |
              v                                v
+-------------+-------------+    +-------------+-------------+
| 特征提取                  |    | 特征提取                  |
+-------------+-------------+    +-------------+-------------+
              |                                |
              +----------------+----------------+
                               |
                               v
                  +------------+------------+
                  | 查询 (Query) 生成       |
                  +------------+------------+
                               |
                               v
                  +------------+------------+
                  | 键-值 (Key-Value) 生成  |
                  +------------+------------+
                               |
                               v
                  +------------+------------+
                  | 注意力权重计算          |
                  | softmax(Q·K^T/√d)       |
                  +------------+------------+
                               |
                               v
                  +------------+------------+
                  | 特征聚合                |
                  | (Attention·V)          |
                  +------------+------------+
                               |
                               v
                  +------------+------------+
                  | 输出增强特征            |
                  +---------------------------+
```

### 1.4 渲染解码器结构

```
+-------------------------+    +------------------------+
| 调制信号                 |    | 输入坐标               |
+------------+------------+    +-----------+------------+
             |                              |
             +------------------------------+
                            |
                            v
               +------------+------------+
               | 特征合并                |
               +------------+------------+
                            |
                            v
     +------------------------+------------------------+
     |                        |                        |
     v                        v                        v
+----+----+            +-----+-----+            +-----+-----+
| 尺度调制 |            | 偏移调制  |            | 基础MLP    |
+----+----+            +-----+-----+            +-----+-----+
     |                        |                        |
     +------------------------+------------------------+
                            |
                            v
               +------------+------------+
               | 多层MLP网络             |
               +------------+------------+
                            |
                            v
               +------------+------------+
               | 输出RGB像素值           |
               +---------------------------+
```

## 2. 数据流程图

### 2.1 训练流程

```
+------------------+    +------------------+    +------------------+
| 训练数据集       |    | 数据增强         |    | 随机裁剪         |
| (DIV2K)          +--->+ (翻转/旋转)      +--->+ (48×48)          |
+------------------+    +------------------+    +--------+---------+
                                                         |
                                                         v
+------------------+    +------------------+    +--------+---------+
| 损失计算         |    | 模型输出         |    | 模型前向传播     |
| (L1+感知损失)    |<---+ (SR图像)         |<---+ (LMLTE)          |
+--------+---------+    +------------------+    +------------------+
         |
         v
+--------+---------+    +------------------+
| 反向传播         |    | 优化器更新       |
| (梯度计算)       +--->+ (Adam)           |
+------------------+    +------------------+
```

### 2.2 推理流程

```
+------------------+
| 输入低分辨率图像 |
+--------+---------+
         |
         v
+--------+---------+
| SwinIR编码器     |
| 特征提取         |
+--------+---------+
         |
         v
+--------+---------+    +------------------+
| 生成目标坐标网格 |    | 生成像素单元格   |
| (高分辨率空间)   +--->+ (cell)           |
+--------+---------+    +--------+---------+
                                 |
                                 v
                        +--------+---------+
                        | 跨尺度注意力     |
                        | 特征增强         |
                        +--------+---------+
                                 |
                                 v
                        +--------+---------+
                        | 生成调制信号     |
                        | (超网络)         |
                        +--------+---------+
                                 |
                                 v
                        +--------+---------+
                        | MLP渲染器        |
                        | 图像重建         |
                        +--------+---------+
                                 |
                                 v
                        +--------+---------+
                        | 输出高分辨率图像 |
                        +------------------+
```

## 3. 使用示例

### 3.1 命令行使用示例

#### 评估模型性能

```bash
# 在单个数据集上评估
python eval_simple.py --model_path save/swinir-b_lm-lmlte_new/epoch-best.pth \
                     --config_path configs/train-lmf/train_swinir-baseline-lmlte_small.yaml \
                     --data_dir ./load/Set5 \
                     --scale 4

# 在多个数据集上批量评估
python batch_eval.py --model_path save/swinir-b_lm-lmlte_new/epoch-best.pth \
                    --config_path configs/train-lmf/train_swinir-baseline-lmlte_small.yaml
```

#### 训练模型

```bash
# 使用SwinIR编码器训练
python train_windows.py --config configs/train-lmf/train_swinir-baseline-lmlte_small.yaml

# 使用EDSR编码器训练
python train_edsr_lmlte.py --config configs/train-lmf/train_edsr-baseline-lmlte.yaml

# 使用RDN编码器训练
python train_rdn_lmlte.py --config configs/train-lmf/train_rdn-baseline-lmlte.yaml
```

### 3.2 Python API使用示例

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# 加载模型
def load_model(model_path, config_path):
    from models import make_model
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
    model = make_model(config['model'])
    
    # 加载权重
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.eval()
    
    return model

# 超分辨率处理函数
def super_resolve(model, img_path, scale=4, device='cuda'):
    # 加载图像
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    lr_tensor = transform(img).unsqueeze(0).to(device)
    
    # 生成坐标网格
    from utils import make_coord
    h, w = lr_tensor.shape[-2:]
    coord = make_coord((h*scale, w*scale)).to(device)
    coord = coord.unsqueeze(0)
    
    # 生成cell
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / (h * scale)
    cell[:, :, 1] *= 2 / (w * scale)
    
    # 执行超分辨率
    with torch.no_grad():
        sr_tensor = model(lr_tensor, coord, cell)
        sr_tensor = sr_tensor.view(1, h*scale, w*scale, 3).permute(0, 3, 1, 2)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
    
    # 转换为PIL图像
    to_pil = transforms.ToPILImage()
    sr_img = to_pil(sr_tensor.squeeze(0).cpu())
    
    return sr_img

# 示例用法
if __name__ == "__main__":
    # 配置参数
    model_path = "save/swinir-b_lm-lmlte_new/epoch-best.pth"
    config_path = "configs/train-lmf/train_swinir-baseline-lmlte_small.yaml"
    img_path = "load/Set5/butterfly.png"
    output_path = "butterfly_sr.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model = load_model(model_path, config_path).to(device)
    
    # 超分辨率处理
    sr_img = super_resolve(model, img_path, scale=4, device=device)
    
    # 保存结果
    sr_img.save(output_path)
    print(f"超分辨率图像已保存至 {output_path}")
```

### 3.3 批处理示例

处理文件夹中的所有图像：

```python
import os
import glob
from PIL import Image
import torch
from torchvision import transforms

def process_folder(model, input_dir, output_dir, scale=4, device='cuda'):
    """处理文件夹中的所有图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    img_paths = glob.glob(os.path.join(input_dir, "*.png")) + \
               glob.glob(os.path.join(input_dir, "*.jpg")) + \
               glob.glob(os.path.join(input_dir, "*.bmp"))
    
    # 转换函数
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 处理每张图像
    for img_path in img_paths:
        # 获取文件名
        filename = os.path.basename(img_path)
        print(f"处理: {filename}")
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        lr_tensor = transform(img).unsqueeze(0).to(device)
        
        # 生成坐标和cell
        from utils import make_coord
        h, w = lr_tensor.shape[-2:]
        coord = make_coord((h*scale, w*scale)).to(device)
        coord = coord.unsqueeze(0)
        
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / (h * scale)
        cell[:, :, 1] *= 2 / (w * scale)
        
        # 执行超分辨率
        with torch.no_grad():
            sr_tensor = model(lr_tensor, coord, cell)
            sr_tensor = sr_tensor.view(1, h*scale, w*scale, 3).permute(0, 3, 1, 2)
            sr_tensor = torch.clamp(sr_tensor, 0, 1)
        
        # 保存结果
        to_pil = transforms.ToPILImage()
        sr_img = to_pil(sr_tensor.squeeze(0).cpu())
        output_path = os.path.join(output_dir, f"sr_{filename}")
        sr_img.save(output_path)
    
    print(f"所有图像处理完成，结果已保存至 {output_dir}")

# 示例用法
if __name__ == "__main__":
    # 配置参数
    model_path = "save/swinir-b_lm-lmlte_new/epoch-best.pth"
    config_path = "configs/train-lmf/train_swinir-baseline-lmlte_small.yaml"
    input_dir = "load/Set5"
    output_dir = "results/Set5_x4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model = load_model(model_path, config_path).to(device)
    
    # 处理文件夹
    process_folder(model, input_dir, output_dir, scale=4, device=device)
```

## 4. 输出效果展示

### 4.1 超分辨率效果对比

| 原始低分辨率图像 | 双三次插值 | LMLTE (SwinIR) | LMLTE (EDSR) | LMLTE (RDN) | 原始高分辨率图像 |
|----------------|------------|---------------|-------------|------------|----------------|
| ![LR](butterfly_lr.png) | ![Bicubic](butterfly_bicubic.png) | ![SwinIR](butterfly_swinir.png) | ![EDSR](butterfly_edsr.png) | ![RDN](butterfly_rdn.png) | ![HR](butterfly_hr.png) |

### 4.2 特征可视化

特征图可视化:
![Feature Maps](feature_visualization.png)

注意力权重可视化:
![Attention Weights](attention_visualization.png)

### 4.3 不同缩放比例效果

| 原始图像 | ×2缩放 | ×3缩放 | ×4缩放 |
|---------|--------|--------|--------|
| ![Original](original.png) | ![x2](x2.png) | ![x3](x3.png) | ![x4](x4.png) |

## 5. Web演示界面设计

### 5.1 界面布局

```
+-----------------------------------------------------------------------+
|                           LMF 超分辨率演示                              |
+-----------------------------------------------------------------------+
|                                                                       |
|  +------------------+                      +----------------------+   |
|  |                  |                      |                      |   |
|  |                  |                      |                      |   |
|  |                  |                      |                      |   |
|  |   输入图像        |                      |   输出图像           |   |
|  |                  |                      |                      |   |
|  |                  |                      |                      |   |
|  |                  |                      |                      |   |
|  +------------------+                      +----------------------+   |
|                                                                       |
|  +------------------------------------------------------------------+ |
|  |                          参数设置                                 | |
|  |                                                                  | |
|  |  编码器:  [SwinIR ▼]   缩放比例: [×4 ▼]   模型: [standard ▼]     | |
|  |                                                                  | |
|  |  [上传图像]                                    [开始处理]         | |
|  +------------------------------------------------------------------+ |
|                                                                       |
|  +------------------------------------------------------------------+ |
|  |                          进阶选项                                 | |
|  |                                                                  | |
|  |  □ 保存中间特征     □ 显示注意力图     □ 批量处理                 | |
|  +------------------------------------------------------------------+ |
|                                                                       |
+-----------------------------------------------------------------------+
```

### 5.2 简易 HTML/JS 实现

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LMF 超分辨率演示</title>
    <style>
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
            color: #1d1d1f;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 18px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        h1 {
            text-align: center;
            font-weight: 600;
            margin-bottom: 30px;
            color: #1d1d1f;
        }
        
        .image-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
        }
        
        .image-box {
            flex: 1;
            margin: 0 15px;
            text-align: center;
        }
        
        .image-box img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .image-box h3 {
            margin-top: 15px;
            font-weight: 500;
        }
        
        .panel {
            background-color: #f5f5f7;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .panel h2 {
            margin-top: 0;
            font-size: 18px;
            font-weight: 500;
        }
        
        .form-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        select, button {
            background-color: white;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 16px;
            color: #1d1d1f;
        }
        
        button {
            background-color: #0071e3;
            color: white;
            border: none;
            padding: 12px 25px;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        button:hover {
            background-color: #0062cc;
        }
        
        .checkbox-group {
            display: flex;
            gap: 20px;
        }
        
        .checkbox-label {
            display: flex;
            align-items: center;
            cursor: pointer;
        }
        
        .checkbox-label input {
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>LMF 超分辨率演示</h1>
        
        <div class="image-container">
            <div class="image-box">
                <img id="input-image" src="placeholder.jpg" alt="输入图像">
                <h3>输入图像</h3>
            </div>
            <div class="image-box">
                <img id="output-image" src="placeholder.jpg" alt="输出图像">
                <h3>输出图像</h3>
            </div>
        </div>
        
        <div class="panel">
            <h2>参数设置</h2>
            <div class="form-row">
                <div>
                    <label for="encoder">编码器:</label>
                    <select id="encoder">
                        <option value="swinir">SwinIR</option>
                        <option value="edsr">EDSR</option>
                        <option value="rdn">RDN</option>
                    </select>
                </div>
                
                <div>
                    <label for="scale">缩放比例:</label>
                    <select id="scale">
                        <option value="2">×2</option>
                        <option value="3">×3</option>
                        <option value="4" selected>×4</option>
                    </select>
                </div>
                
                <div>
                    <label for="model">模型:</label>
                    <select id="model">
                        <option value="standard">标准</option>
                        <option value="lightweight">轻量级</option>
                        <option value="memory_efficient">内存高效</option>
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <button id="upload-btn">上传图像</button>
                <input type="file" id="file-input" style="display: none">
                <button id="process-btn">开始处理</button>
            </div>
        </div>
        
        <div class="panel">
            <h2>进阶选项</h2>
            <div class="checkbox-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="save-features"> 保存中间特征
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="show-attention"> 显示注意力图
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="batch-process"> 批量处理
                </label>
            </div>
        </div>
    </div>
    
    <script>
        // 简单的交互逻辑
        document.getElementById('upload-btn').addEventListener('click', function() {
            document.getElementById('file-input').click();
        });
        
        document.getElementById('file-input').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                let file = e.target.files[0];
                let reader = new FileReader();
                
                reader.onload = function(e) {
                    document.getElementById('input-image').src = e.target.result;
                }
                
                reader.readAsDataURL(file);
            }
        });
        
        document.getElementById('process-btn').addEventListener('click', function() {
            // 这里应该是调用后端API处理图像的逻辑
            // 在实际应用中，需要发送AJAX请求到服务器
            
            // 模拟处理
            setTimeout(function() {
                // 模拟接收处理后的图像
                document.getElementById('output-image').src = 'butterfly_sr.png';
                alert('图像处理完成！');
            }, 2000);
        });
    </script>
</body>
</html>
``` 