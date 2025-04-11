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

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(inChannels, growRate, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN_Backbone(nn.Module):
    def __init__(self, G0=64, D=16, C=8, growth_chan=64):
        super(RDN_Backbone, self).__init__()
        self.D = D
        self.G0 = G0
        self.C = C
        self.G = growth_chan
        
        # Shallow feature extraction
        self.SFENet1 = nn.Conv2d(3, G0, kernel_size=3, padding=1, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kernel_size=3, padding=1, stride=1)
        
        # Residual Dense Blocks
        self.RDBs = nn.ModuleList()
        for i in range(D):
            self.RDBs.append(RDB(G0, self.G, C))
            
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(D * G0, G0, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kernel_size=3, padding=1, stride=1)
        ])
        
    def forward(self, x):
        # Shallow Feature Extraction
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        
        # Residual Dense Blocks
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
            
        # Global Feature Fusion
        x = self.GFF(torch.cat(RDBs_out, 1)) + f__1
        
        return x

class LRLTE(nn.Module):
    """Local Reciprocal LTE module for feature enhancement"""
    def __init__(self, channels):
        super(LRLTE, self).__init__()
        self.channels = channels
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Final projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Scaling factor
        self.scale = channels ** -0.5
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Generate Query, Key, Value
        q = self.query(x).view(b, c, -1).permute(0, 2, 1)  # B, HW, C
        k = self.key(x).view(b, c, -1)  # B, C, HW
        v = self.value(x).view(b, c, -1).permute(0, 2, 1)  # B, HW, C
        
        # Compute attention scores
        attn = torch.matmul(q, k) * self.scale  # B, HW, HW
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value
        out = torch.matmul(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        out = self.out_proj(out)
        
        # Residual connection
        return out + x

class UpBlock(nn.Module):
    def __init__(self, channels, scale):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.up(x)

class ImplicitCoordEncoder(nn.Module):
    def __init__(self, channels):
        super(ImplicitCoordEncoder, self).__init__()
        self.conv1 = nn.Conv2d(2, channels//2, kernel_size=1)
        self.conv2 = nn.Conv2d(channels//2, channels, kernel_size=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, coord):
        # coord is expected to be normalized to [-1, 1]
        x = self.conv1(coord)
        x = self.act(x)
        x = self.conv2(x)
        return x

@register('lmlte_rdn')
class LMLTE_RDN(nn.Module):
    def __init__(self, scale=2, G0=64, D=16, C=8, growth_chan=64, local_mode='LRLTE'):
        super(LMLTE_RDN, self).__init__()
        self.scale = scale
        self.local_mode = local_mode
        
        # RDN Backbone
        self.backbone = RDN_Backbone(G0, D, C, growth_chan)
        
        # Local Modulation
        if local_mode == 'LRLTE':
            self.local_modulator = LRLTE(G0)
        else:
            self.local_modulator = nn.Identity()
            
        # Coordinate encoder
        self.coord_encoder = ImplicitCoordEncoder(G0)
        
        # Upsampling
        self.up = UpBlock(G0, scale)
        
    def forward(self, x, coord=None, cell=None):
        """
        x: input image [B, 3, H, W]
        coord: coordinates [B, 2, H_out, W_out]
        cell: cell sizes [B, 2, H_out, W_out]
        """
        # Get features from backbone
        feat = self.backbone(x)
        
        # Apply local modulation
        feat = self.local_modulator(feat)
        
        # Process with coordinates if provided
        if coord is not None:
            coord_feat = self.coord_encoder(coord)
            
            # Resize feature to match coordinate size if needed
            if feat.shape[2:] != coord_feat.shape[2:]:
                feat = F.interpolate(feat, size=coord_feat.shape[2:], mode='bilinear', align_corners=False)
                
            # Combine features
            feat = feat + coord_feat
        
        # Upsampling
        if coord is None:
            # Standard upsampling
            out = self.up(feat)
        else:
            # Already at target resolution due to coordinates
            out = self.up(feat)
            
        return out

# For model registry
def make_lmlte_rdn(scale=2, G0=64, D=16, C=8, growth_chan=64, local_mode='LRLTE'):
    return LMLTE_RDN(scale, G0, D, C, growth_chan, local_mode)

@register('lmlte-rdn')
class LMLTE_RDN_V2(LMLTE):
    """
    LMLTE模型，使用RDN作为编码器
    """
    def __init__(self, encoder_spec={'name': 'rdn', 'args': {'no_upsampling': True}}, 
                 imnet_spec=None, hypernet_spec=None,
                 imnet_q=None, imnet_k=None, imnet_v=None,
                 hidden_dim=128, local_ensemble=True, cell_decode=True,
                 mod_input=False, cmsr_spec=None):
        
        # 确保使用RDN作为编码器
        if encoder_spec['name'] != 'rdn':
            print(f"Warning: 覆盖指定的编码器 {encoder_spec['name']} 为 RDN")
            encoder_spec = {'name': 'rdn', 'args': {'no_upsampling': True}}
        
        if 'args' not in encoder_spec:
            encoder_spec['args'] = {'no_upsampling': True}
        elif 'no_upsampling' not in encoder_spec['args']:
            encoder_spec['args']['no_upsampling'] = True
        
        # 调用父类的构造函数
        super().__init__(encoder_spec, imnet_spec, hypernet_spec,
                         imnet_q, imnet_k, imnet_v,
                         hidden_dim, local_ensemble, cell_decode,
                         mod_input, cmsr_spec)
        
        print(f"LMLTE_RDN_V2 模型初始化完成，使用 RDN 作为编码器")
        
        # 计算并显示模型的参数总数
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"RDN编码器的总参数量: {'{:.1f}K'.format(total_params/ 1e3)}")
        
    def forward(self, inp, coord, cell=None):
        """
        前向传播
        
        :param inp: 输入图像
        :param coord: 坐标
        :param cell: 像素单元大小
        """
        # 使用父类的前向传播逻辑
        return super().forward(inp, coord, cell) 