import torch
import torch.nn as nn
from .models import register

@register('metasr_simple')
class MetaSR(nn.Module):
    """MetaSR模型的简化实现"""
    
    def __init__(self, encoder_spec, imnet_spec=None):
        super().__init__()
        self.encoder = make_encoder(encoder_spec)
        
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            self.imnet = make_imnet(imnet_spec, in_dim=imnet_in_dim)
        else:
            self.imnet = None
    
    def gen_feat(self, inp):
        return self.encoder(inp)
    
    def query_rgb(self, feat, coord):
        feat = feat.permute(0, 2, 3, 1)
        return self.imnet(feat, coord)
    
    def forward(self, inp, coord):
        feat = self.gen_feat(inp)
        return self.query_rgb(feat, coord)

def make_encoder(encoder_spec):
    """创建编码器"""
    if encoder_spec is None:
        return nn.Identity()
    
    if isinstance(encoder_spec, dict):
        name = encoder_spec['name']
        # 实际项目中应该有更完善的实现
        return nn.Conv2d(3, 64, 3, padding=1)
    else:
        return encoder_spec

def make_imnet(imnet_spec, **kwargs):
    """创建隐式网络"""
    if isinstance(imnet_spec, dict):
        name = imnet_spec['name']
        # 实际项目中应该有更完善的实现
        return nn.Linear(64, 3)
    else:
        return imnet_spec 