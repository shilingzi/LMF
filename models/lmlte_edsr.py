import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
from models import register
from utils import make_coord
from models.arch_ciaosr.arch_csnln import CrossScaleAttention
from models.lmlte import LMLTE

@register('lmlte-edsr')
class LMLTE_EDSR(LMLTE):
    """
    LMLTE模型，使用EDSR作为编码器
    """
    def __init__(self, encoder_spec={'name': 'edsr-baseline', 'args': {'no_upsampling': True}}, 
                 imnet_spec=None, hypernet_spec=None,
                 imnet_q=None, imnet_k=None, imnet_v=None,
                 hidden_dim=128, local_ensemble=True, cell_decode=True,
                 mod_input=False, cmsr_spec=None):
        
        # 确保使用EDSR作为编码器
        if encoder_spec['name'] != 'edsr-baseline' and encoder_spec['name'] != 'edsr':
            print(f"Warning: 覆盖指定的编码器 {encoder_spec['name']} 为 EDSR")
            encoder_spec = {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}
        
        if 'args' not in encoder_spec:
            encoder_spec['args'] = {'no_upsampling': True}
        elif 'no_upsampling' not in encoder_spec['args']:
            encoder_spec['args']['no_upsampling'] = True
        
        # 调用父类的构造函数
        super().__init__(encoder_spec, imnet_spec, hypernet_spec,
                         imnet_q, imnet_k, imnet_v,
                         hidden_dim, local_ensemble, cell_decode,
                         mod_input, cmsr_spec)
        
        print(f"LMLTE_EDSR 模型初始化完成，使用 {encoder_spec['name']} 作为编码器")
        
        # 计算并显示模型的参数总数
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"EDSR编码器的总参数量: {'{:.1f}K'.format(total_params/ 1e3)}")
        
    def forward(self, inp, coord, cell=None):
        """
        前向传播
        
        :param inp: 输入图像
        :param coord: 坐标
        :param cell: 像素单元大小
        """
        # 使用父类的前向传播逻辑
        return super().forward(inp, coord, cell) 