# 不同编码器的LMLTE模型性能对比

## 评估设置

- 超分辨率比例: 4x
- 评估时间: 2025-03-27 18:29:16

## 评估结果 (PSNR in dB)

| 数据集 | EDSR | SwinIR |
|--------|----------|----------|
| Set5 | 0.00 | 13.03 |
| Set14 | 0.00 | 15.91 |
| DIV2K_valid_HR | 0.00 | 17.48 |
| **平均** | **0.00** | **15.47** |

## 结论

在本次比较中，**SwinIR**表现最好，平均PSNR达到了**15.47 dB**。

### 编码器对比分析

各编码器的特点：

- **EDSR**：深度残差网络，通过大量残差块和跳跃连接提取特征，结构简单但效果强大。
- **SwinIR**：基于Transformer的架构，通过自注意力机制捕捉长距离依赖，全局建模能力更强。

## 详细输出

### EDSR 编码器

- [Set5](EDSR_Set5_output_20250327_175914.txt)
- [Set14](EDSR_Set14_output_20250327_175914.txt)
- [DIV2K_valid_HR](EDSR_DIV2K_valid_HR_output_20250327_175914.txt)

### SwinIR 编码器

- [Set5](SwinIR_Set5_output_20250327_175914.txt)
- [Set14](SwinIR_Set14_output_20250327_175914.txt)
- [DIV2K_valid_HR](SwinIR_DIV2K_valid_HR_output_20250327_175914.txt)

