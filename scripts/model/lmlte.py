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

@register('lmlte')
class LMLTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hypernet_spec=None,
                 imnet_q=None, imnet_k=None, imnet_v=None,
                 hidden_dim=128, local_ensemble=True, cell_decode=True,
                 mod_input=False, cmsr_spec=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.max_scale = 4  # Max training scale
        self.mod_input = mod_input  # Set to True if use compressed latent code
        self.non_local_attn = True
        self.multi_scale = [2]
        self.local_size=2
        self.softmax_scale = 1
        self.feat_unfold = True
        self.unfold_range = 3 
        self.unfold_dim = self.unfold_range * self.unfold_range

        self.encoder = models.make(encoder_spec)
        self.coef = nn.Linear(2*self.encoder.out_dim, hidden_dim , bias=False)
        self.freq = nn.Linear(2*self.encoder.out_dim, hidden_dim , bias=False)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        
        self.cs_attn = CrossScaleAttention(channel=self.encoder.out_dim, scale=self.multi_scale)
        self.non_local_attn_dim = self.encoder.out_dim * len(self.multi_scale)

        # Use latent MLPs to generate modulations for the render MLP
        if hypernet_spec is not None:
            hypernet_in_dim = hidden_dim
            self.mod_dim = 0
            self.mod_dim += imnet_spec['args']['hidden_dim'] if imnet_spec['args']['mod_scale'] else 0
            self.mod_dim += imnet_spec['args']['hidden_dim'] if imnet_spec['args']['mod_shift'] else 0
            self.mod_dim *= imnet_spec['args']['hidden_depth']

            hypernet_out_dim = self.mod_dim
            if self.mod_input:
                self.mod_coef_dim = self.mod_freq_dim = imnet_spec['args']['hidden_dim']
                hypernet_out_dim += self.mod_coef_dim + self.mod_freq_dim

            self.hypernet = models.make(hypernet_spec, args={'in_dim': hypernet_in_dim, 'out_dim': hypernet_out_dim})
        else:
            self.hypernet = None
            
        self.imnet_q = models.make(imnet_q, args={'in_dim': self.encoder.out_dim, 'out_dim': self.encoder.out_dim})
        self.imnet_k = models.make(imnet_k, args={'in_dim': self.encoder.out_dim+2, 'out_dim': self.encoder.out_dim})
        self.imnet_v = models.make(imnet_v, args={'in_dim': 2*self.encoder.out_dim, 'out_dim': 2*self.encoder.out_dim})


        # Render MLP
        if imnet_spec is not None:
            if self.mod_input:
                self.imphase = nn.Linear(2, imnet_spec['args']['hidden_dim'] // 2, bias=False)
            imnet_in_dim = imnet_spec['args']['hidden_dim'] if self.mod_input else hidden_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim, 'mod_up_merge': False})
        else:
            self.imnet = None

        # For time evaluation
        self.t_total = []
        self.feat_coord = None

        # Use CMSR in testing
        self.cmsr = cmsr_spec is not None
        if self.cmsr:
            self.mse_threshold = cmsr_spec["mse_threshold"]  # 0.00002
            self.s2m_tables = cmsr_spec["s2m_tables"]
            self.updating_cmsr = "updating_scale" in cmsr_spec

            if self.updating_cmsr:
                self.updating_scale = cmsr_spec["updating_scale"]
                self.scale2mean = self.s2m_tables[self.updating_scale]
                self.loss_fn = nn.MSELoss()
                print(f'Generating S2M Table at scale {self.updating_scale} created with MSE: {self.mse_threshold}')
            else:
                # Monitor the computational cost saved by CMSR
                self.cmsr_log = cmsr_spec["log"]
                if self.cmsr_log:
                    self.total_qn, self.total_q = 0, 0
                print(f'Using S2M Table created with MSE: {self.mse_threshold}')

    def gen_feats(self, inp, inp_coord=None):
        """
        Generate latent codes using the encoder.

        :param inp: Input image (B, h * w, 3)
        :param inp_coord: Input coordinates (B, h * w, 2)
        :return: Feature maps (B, C, h, w) and (B, 9*C, h, w)
        """

        feat = self.encoder(inp)
        # 计算并显示模型的参数总数
       # total_params = sum(p.numel() for p in self.encoder.parameters())
        #print(f"encoder模型的总参数量: {'{:.1f}K'.format(total_params/ 1e3)}")            
        self.feat=feat
        [bs, in_c, in_h, in_w] = feat.shape

        if inp_coord is not None:
            self.feat_coord = inp_coord.permute(0, 3, 1, 2)
        elif self.training or self.feat_coord is None or self.feat_coord.shape[-2] != feat.shape[-2]\
                or self.feat_coord.shape[-1] != feat.shape[-1]:
            self.feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # self.t1 = time.time()

        if self.non_local_attn:
            crop_h, crop_w = 48, 48
            if in_h * in_w > crop_h * crop_w:
                # Fixme: Generate cross attention by image patches to avoid OOM
                self.non_local_feat = torch.zeros(bs, self.non_local_attn_dim, in_h, in_w).to(feat.device)
                for i in range(in_h // crop_h):
                    for j in range(in_w // crop_w):
                        i1, i2 = i * crop_h, ((i + 1) * crop_h if i < in_h // crop_h - 1 else in_h)
                        j1, j2 = j * crop_w, ((j + 1) * crop_w if j < in_w // crop_w - 1 else in_w)

                        padding = 3 // 2
                        pad_i1, pad_i2 = (padding if i1 - padding >= 0 else 0), (padding if i2 + padding <= in_h else 0)
                        pad_j1, pad_j2 = (padding if j1 - padding >= 0 else 0), (padding if j2 + padding <= in_w else 0)

                        crop_feat = feat[:, :, i1 - pad_i1:i2 + pad_i2, j1 - pad_j1:j2 + pad_j2]
                        crop_non_local_feat = self.cs_attn(crop_feat)
                        self.non_local_feat[:, :, i1:i2, j1:j2] = crop_non_local_feat[:, :,
                                                                  pad_i1:crop_non_local_feat.shape[-2] - pad_i2,
                                                                  pad_j1:crop_non_local_feat.shape[-1] - pad_j2]
            else:
                self.non_local_feat = self.cs_attn(feat)
        # if self.feat_unfold:
        #     # 3x3 feature unfolding
        #     rich_feat = F.unfold(feat, self.unfold_range, padding=self.unfold_range // 2).view(
        #         bs, in_c * self.unfold_dim, in_h, in_w)
        # else:
        #     rich_feat = feat


        return feat
    
    def positional_encoding(self,input,L): # [B,...,N]
        
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32).to(input.device)
        freq  =  freq*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]

        return input_enc

    def gen_modulations(self, feat, cell=None):
        """
        Generate latent modulations using the latent MLP.

        :param coef: Coefficient (B, C, h, w)
        :param freq: Frequency (B, C, h, w)
        :param cell: Cell areas (B, H * W, 2)
        :return: Latent modulations (B, C', h, w)
        """
        
        ######################
       
        bs, c, h, w = feat.shape
      
        feat_coord = self.feat_coord

        # Field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # 2x2 feature ensemble
        if self.local_size == 1:
            v_lst = [(0, 0)]
        else:
            v_lst = [(i, j) for i in range(-1, 2, 4 - self.local_size) for j in range(-1, 2, 4 - self.local_size)]

        preds_k, preds_v = [], []
        for v in v_lst:
            vx, vy = v[0], v[1]

            key = feat.view(bs, c, h * w).permute(0, 2, 1)
            value = feat
            if self.non_local_attn:
                value = torch.cat([value, self.non_local_feat], dim=1)
                value = value.view(bs, 2*c, h * w).permute(0, 2, 1)

            coord_q = feat_coord.view(bs, feat_coord.shape[1], h * w).permute(0, 2, 1)
            coord_k = coord_q.clone()
            coord_k[:, :, 0] += vx * rx / feat.shape[-2]  # + eps_shift
            coord_k[:, :, 1] += vy * ry / feat.shape[-1]  # + eps_shift

            bs, q = coord_q.shape[:2]
            Q, K = coord_q, coord_k
            rel = Q - K
            rel_coord = rel
            rel[:, :, 0] *= feat.shape[-2]  # without mul
            rel[:, :, 1] *= feat.shape[-1]
            inp = rel
            
            rel_encode = self.positional_encoding(rel_coord,L=8)
            

            scale_ = cell[:, :feat.shape[-2] * feat.shape[-1], :].clone()
            scale_[:, :, 0] *= feat.shape[-2]
            scale_[:, :, 1] *= feat.shape[-1]
            
        
            coef = self.coef(value)
            freq = self.freq(value) #[16, 2304, 128]
            freq = torch.stack(torch.split(freq, 2, dim=-1), dim=-1) #[16, 2304, 2, 64]
            #freq = torch.mul(freq, inp.unsqueeze(-1))
            freq = torch.sum(freq, dim=-2) #[16, 2304, 64]
           
            
           
            if self.cell_decode:
                # Use relative height, width info
                rel_cell = cell.clone()[:, :h * w, :]
                rel_cell[:, :, 0] *= h
                rel_cell[:, :, 1] *= w
                freq += self.phase(rel_cell.view((bs * h * w, -1))).view(bs, h * w, -1)
            freq = torch.cat((torch.cos(np.pi * freq), torch.sin(np.pi * freq)), dim=-1)

            initial_mod = torch.mul(coef, freq)
            
            key_s = torch.stack(torch.split(key, 2, dim=-1), dim=-1) #[16, 2304, 2, 32]
            key_s = torch.mul(key_s, inp.unsqueeze(-1))
            key_s = torch.sum(key_s, dim=-2) #[16, 2304, 32]

            weight_k = self.imnet_k(torch.cat([key_s, rel_encode, scale_], dim=-1).view(bs * q, -1)).view(bs, q, -1)
            weight_v = self.imnet_v(initial_mod.view(bs * q, -1)).view(bs, q, -1)

            preds_k.append((key * weight_k).view(bs, q, -1))
            preds_v.append((value * weight_v).view(bs, q, -1))

        preds_k = torch.stack(preds_k, dim=-1)
        preds_v = torch.stack(preds_v, dim=-2)

        query = feat.view(bs, c, h * w).permute(0, 2, 1).contiguous().view(bs * h * w, -1)
        query = self.imnet_q(query).view(bs, h * w, -1).unsqueeze(2)

        # Query modulations
        attn = (query @ preds_k)
        inp_q = ((attn / self.softmax_scale).softmax(dim=-1) @ preds_v)
        mod = self.hypernet(inp_q.view(bs * q, -1)).view(bs, h, w, -1).permute(0, 3, 1, 2)
       
        return mod

    def update_scale2mean(self, coef, freq, mod, coord, cell=None):
        """
        Update the Scale2mod table for CMSR testing.

        :param coef: Coefficient (B, C, h, w)
        :param freq: Frequency (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :return:
        """

        bs = coord.shape[0]
        # Query RGBs with target scale
        max_pred = self.query_rgb(coef, freq, mod, coord, cell)
        max_pred = max_pred.view(bs * coord.shape[1], -1)

        # Bilinear upsample mod mean to target scale
        mod_mean = torch.mean(torch.abs(mod[:, self.mod_dim // 2:self.mod_dim, :, :]), 1, keepdim=True)
        mod_mean = F.grid_sample(
            mod_mean, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        min_, max_ = 0, 0.5
        samples = [min_ + (max_ - min_) * i / 100 for i in range(101)]
        max_scale = math.sqrt(coord.shape[1] / coef.shape[-2] / coef.shape[-1])
        for scale in self.scale2mean.keys():
            if scale >= max_scale:
                break

            # Query rgbs with current scale
            qh, qw = int(coef.shape[-2] * scale), int(coef.shape[-1] * scale)
            q_coord = make_coord([qh, qw], flatten=False).to(coord.device).view(bs, qh * qw, -1)
            q_cell = torch.ones_like(q_coord)
            q_cell[:, :, 0] *= 2 / qh
            q_cell[:, :, 1] *= 2 / qw
            q_pred = self.query_rgb(coef, freq, mod, q_coord, q_cell)

            # Bilinear upsample rgbs to target scale
            pred = F.grid_sample(
                q_pred.view(bs, qh, qw, -1).permute(0, 3, 1, 2), coord.flip(-1).unsqueeze(1), mode='bilinear',
                padding_mode='border', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            pred = pred.view(bs * coord.shape[1], -1)

            max_sample = self.scale2mean[scale]
            for mid in [i for i in samples]:
                mask_indice = torch.where(torch.abs(mod_mean - mid).flatten() <= 0.001)[0]
                loss = self.loss_fn(pred[mask_indice, :], max_pred[mask_indice, :])

                if loss == loss:
                    if loss <= float(self.mse_threshold):
                        # Fully rendered at current scale
                        samples.remove(mid)
                        max_sample = mid
                    else:
                        break

            # if max_sample < self.scale2mean[scale]:
            #     print(self.scale2mean)
            self.scale2mean[scale] = max_sample if max_sample < self.scale2mean[scale] else self.scale2mean[scale]
            for s in self.scale2mean.keys():
                if s < scale and self.scale2mean[s] > self.scale2mean[scale]:
                    self.scale2mean[s] = self.scale2mean[scale]

        if samples:
            self.scale2mean[max_scale] = samples[-1]
            for s in self.scale2mean.keys():
                if s < max_scale and self.scale2mean[s] > self.scale2mean[max_scale]:
                    self.scale2mean[s] = self.scale2mean[max_scale]

        return self.scale2mean

    def query_rgb_cmsr(self, coef, freq, mod, coord, cell=None):
        """
        Query RGB values of each coordinate using latent modulations and latent codes. (CMSR included)

        :param coef: Coefficient (B, C, h, w)
        :param freq: Frequency (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :return: Predicted RGBs (B, H * W, 3)
        """

        bs, qn = coord.shape[:2]
        if self.cmsr_log:
            self.total_qn += qn

        mod_mean = torch.mean(torch.abs(mod[:, self.mod_dim // 2:self.mod_dim, :, :]), 1, keepdim=True)

        # Load the Scale2mod table
        scale = math.sqrt(qn / coef.shape[-2] / coef.shape[-1])
        for k, v in self.s2m_tables.items():
            scale2mean = self.s2m_tables[k]
            if k >= scale:
                break
        decode_scales, mask_thresholds = [], [0]
        for s, t in scale2mean.items():
            if s >= scale:
                break
            decode_scales.append(s)
            mask_thresholds.append(t)
        if mask_thresholds[-1] < 1:
            decode_scales.append(scale)
            mask_thresholds.append(1)
        mask_level = len(mask_thresholds) - 1

        i_start, i_end = 0, mask_level - 1
        q_coords, masked_coords, masked_cells, mask_indices = [], [], [], []
        for i in range(mask_level):
            decode_scale = decode_scales[i]
            # Skip decoding if decoding scale < 1
            if decode_scale < 1:
                i_start += 1
                continue

            qh, qw = int(coef.shape[-2] * decode_scale), int(coef.shape[-1] * decode_scale)
            q_coord = F.interpolate(self.feat_coord, size=(qh, qw), mode='bilinear',
                                    align_corners=False, antialias=False).permute(0, 2, 3, 1).view(bs, qh * qw, -1)
            #q_coord = make_coord([qh, qw], flatten=False).cuda().view(bs, qh * qw, -1)

            # Only query coordinates where mod means indicate that they can be decoded to desired accuracy at current scale
            if i == i_end or i == i_start:
                q_mod_mean = F.grid_sample(
                    mod_mean, q_coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                if i == i_end:
                    # Query pixels where mod mean >= min threshold
                    q_mask_indice = torch.where(q_mod_mean.flatten() >= mask_thresholds[i])[0]
                else:
                    # Query pixels where mod mean <= max threshold
                    q_mask_indice = torch.where(q_mod_mean.flatten() <= mask_thresholds[i + 1])[0]
            else:
                # Query pixels where min threshold <= mod mean <= max threshold
                min_, max_ = mask_thresholds[i], mask_thresholds[i + 1]
                mid = (max_ + min_) / 2
                r = (max_ - min_) / 2
                q_mod_mean = F.grid_sample(
                    torch.abs(mod_mean - mid), q_coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_mask_indice = torch.where(q_mod_mean.flatten() <= r)[0]

            mask_indices.append(q_mask_indice)
            if self.cmsr_log:
                self.total_q += len(q_mask_indice) / ((scale / decode_scale) ** 2)
            q_coords.append(q_coord)

            if len(q_mask_indice) <= 0:
                continue

            masked_coords.append(q_coord.view(bs * qh * qw, -1)[q_mask_indice, :].view(bs, len(q_mask_indice) // bs, -1))
            masked_cell = torch.ones_like(masked_coords[-1])
            masked_cell[:, :, 0] *= 2 / qh
            masked_cell[:, :, 1] *= 2 / qw
            masked_cell = masked_cell * max(decode_scale / self.max_scale, 1)
            masked_cells.append(masked_cell)

        # CMSR debug log
        if self.cmsr_log:
            print('valid mask: ', self.total_q / self.total_qn)
        pred = self.batched_query_rgb(coef, freq, mod, torch.cat(masked_coords, dim=1), torch.cat(masked_cells, dim=1), self.query_bsize)

        # Merge rgb predictions at different scales
        ret = self.inp
        skip_indice_i = 0
        for i in range(i_start, mask_level):
            decode_scale = decode_scales[i]
            qh, qw = int(coef.shape[-2] * decode_scale), int(coef.shape[-1] * decode_scale)

            q_mask_indice = mask_indices[i - i_start]
            q_coord = q_coords[i - i_start]
            if len(q_mask_indice) <= 0:
                continue

            # Bilinear upsample predictions at last scale
            ret = F.grid_sample(
                ret, q_coord.flip(-1).unsqueeze(1), mode='bilinear',
                padding_mode='border', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1).contiguous().view(bs * qh * qw, -1)

            # Merge predictions at current scale
            ret[q_mask_indice, :] = pred[:, skip_indice_i:skip_indice_i + len(q_mask_indice) // bs, :].view(len(q_mask_indice), -1)
            skip_indice_i += len(q_mask_indice)

            if i < mask_level - 1:
                ret = ret.view(bs, qh, qw, -1).permute(0, 3, 1, 2)
            else:
                if decode_scales[-1] < scale and qh * qw != qn:
                    ret = F.grid_sample(
                        ret.view(bs, qh, qw, -1).permute(0, 3, 1, 2), coord.flip(-1).unsqueeze(1), mode='bilinear',
                        padding_mode='border', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                ret = ret.view(bs, qn, -1)

        return ret

    def query_rgb(self, coef, freq, mod, coord, cell=None, batched=False):
        """
        Query RGB values of each coordinate using latent modulations and latent codes. (without CMSR)

        :param coef: Coefficient (B, C, h, w)
        :param freq: Frequency (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :param batched: Set to True if used by batched_query_rgb.
        :return: Predicted RGBs (B, H * W, 3)
        """

        if self.imnet is None:
            return F.grid_sample(
            self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
            padding_mode='border', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).contiguous()

        bs, q = coord.shape[:2]
        h, w = coef.shape[-2:]

        local_ensemble = self.local_ensemble
        if local_ensemble:
            vx_lst = [-1, 1]  # left, right
            vy_lst = [-1, 1]  # top, bottom
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # Field radius (global: [-1, 1])
        rx = 2 / h / 2
        ry = 2 / w / 2

        if not self.training:
            coords = []
            for vx in vx_lst:
                for vy in vy_lst:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coords.append(coord_)
            coords = torch.cat(coords, dim=1)
            coords.clamp_(-1 + 1e-6, 1 - 1e-6)

            q_coords = F.grid_sample(
                self.feat_coord, coords.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

        idx = 0
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                if not self.training:
                    coord_ = coords[:, idx * coord.shape[1]:(idx + 1) * coord.shape[1], :]
                    rel_coord = coord - q_coords[:, idx * coord.shape[1]:(idx + 1) * coord.shape[1], :]
                    rel_coord[:, :, 0] *= h
                    rel_coord[:, :, 1] *= w
                    idx += 1
                else:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_coord = F.grid_sample(
                        self.feat_coord, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= h
                    rel_coord[:, :, 1] *= w

                # Prepare frequency
                inp = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                inp = torch.stack(torch.split(inp, 2, dim=-1), dim=-1)
                inp = torch.mul(inp, rel_coord.unsqueeze(-1))
                inp = torch.sum(inp, dim=-2)
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= h
                    rel_cell[:, :, 1] *= w
                    if self.mod_input:
                        inp += self.imphase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                    else:
                        inp += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                inp = torch.cat((torch.cos(np.pi * inp), torch.sin(np.pi * inp)), dim=-1)

                # Coefficient x frequency
                inp = torch.mul(F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                                .permute(0, 2, 1), inp).contiguous().view(bs * q, -1)

                # Use latent modulations to boost the render mlp
                if self.training:
                    q_mod = F.grid_sample(
                        mod, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1).contiguous()

                    pred = self.imnet(inp, mod=q_mod.view(bs * q, -1)).view(bs, q, -1)
                    preds.append(pred)
                else:
                    pred0 = self.imnet(inp, only_layer0=True)
                    preds.append(pred0)
                    # coords.append(coord_)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        if not self.training:
            # Upsample modulations of each layer seperately, avoiding OOM
            preds = self.imnet(torch.cat(preds, dim=0), mod=mod, coord=coords, # torch.cat(coords, dim=1),
                               skip_layer0=True).view(len(vx_lst) * len(vy_lst), bs, q, -1)

        tot_area = torch.stack(areas).sum(dim=0)
        if local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        # LR skip
        if not batched:
            ret += F.grid_sample(
                self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
                padding_mode='border', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1).contiguous()

        return ret

    def preprocess_coord_cell(self, feat, coord, cell, local_ensemble=True):
        """
        Prepare the coordinates and cells.

        :param feat: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :param local_ensemble:
        :return:
        """

        if local_ensemble:
            vx_lst = [-1, 1]  # left, right
            vy_lst = [-1, 1]  # top, bottom
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        h, w = feat.shape[-2:]

        # Field radius (global: [-1, 1])
        rx = 2 / h / 2
        ry = 2 / w / 2

        coords, rel_coords, rel_cells, areas = [], [], [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                coords.append(coord_)

                q_coords = F.grid_sample(
                    self.feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coords
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
                rel_coords.append(rel_coord)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= h
                    rel_cell[:, :, 1] *= w
                    rel_cells.append(rel_cell)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        coords = torch.cat(coords, dim=1)
        rel_coords = torch.cat(rel_coords, dim=1)
        if self.cell_decode:
            rel_cells = torch.cat(rel_cells, dim=1)
        return coords, rel_coords, rel_cells, areas

    def query_rgb_fast(self, coef, freq, mod, coord, cell=None):
        """
        Query RGB values of input coordinates using latent modulations and latent codes.

        :param coef: Coefficient (B, C, h, w)
        :param freq: Frequency (B, C, h, w)
        :param mod: modulations (B, C', h, w)
        :param coord: coordinates (B, H * W, 2)
        :param cell: cell areas (B, H * W, 2)
        :return: predicted RGBs (B, H * W, 3)
        """

        if self.imnet is None:
            return F.grid_sample(
            self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
            padding_mode='border', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).contiguous()

        bs, q = coord.shape[:2]
        ls = 4 if self.local_ensemble else 1

        coords, rel_coords, rel_cells, areas = self.preprocess_coord_cell(
            mod, coord, cell, local_ensemble=self.local_ensemble)
        le_q, nle_q = q, 0

        # Prepare frequency
        inp = F.grid_sample(
            freq, coords.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        inp = torch.stack(torch.split(inp, 2, dim=-1), dim=-1)
        inp = torch.mul(inp, rel_coords.unsqueeze(-1))
        inp = torch.sum(inp, dim=-2)
        if self.cell_decode:
            if self.mod_input:
                inp += self.imphase(rel_cells.view((bs * (ls * le_q + nle_q), -1))).view(bs, ls * le_q + nle_q, -1)
            else:
                inp += self.phase(rel_cells.view((bs * (ls * le_q + nle_q), -1))).view(bs, ls * le_q + nle_q, -1)
        inp = torch.cat((torch.cos(np.pi * inp), torch.sin(np.pi * inp)), dim=-1)

        # Coefficient x frequency
        inp = torch.mul(F.grid_sample(coef, coords.flip(-1).unsqueeze(1),
                                      mode='nearest', align_corners=False)[:, :, 0, :]
                        .permute(0, 2, 1), inp).contiguous().view(bs * (ls * le_q + nle_q), -1)

        # Upsample modulations of each layer seperately, avoiding OOM
        preds = self.imnet(inp, mod=mod, coord=coords).view(bs, ls * le_q + nle_q, -1)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        le_ret = 0
        for pred, area in zip(preds[:, :ls * le_q, :].view(bs, ls, le_q, -1).permute(1, 0, 2, 3), areas):
            le_ret = le_ret + pred * (area / tot_area).unsqueeze(-1)

        # LR skip
        bil = F.grid_sample(
            self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
            padding_mode='border', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).contiguous()
        ret = le_ret
        ret += bil
        return ret

    def batched_query_rgb(self, coef, freq, mod, coord, cell, bsize):
        """
        Query RGB values of each coordinate batch using latent modulations and latent codes.

        :param coef: Coefficient (B, C, h, w)
        :param freq: Frequency (B, C, h, w)
        :param mod: modulations (B, C', h, w)
        :param coord: coordinates (B, H * W, 2)
        :param cell: cell areas (B, H * W, 2)
        :param bsize: Number of pixels in each query
        :return: predicted RGBs (B, H * W, 3)
        """

        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb_fast(coef, freq, mod, coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, inp, coord=None, cell=None, inp_coord=None, bsize=None):
        """
        Forward function.

        :param inp: Input image (B, h * w, 3)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :param inp_coord: Input coordinates (B, h * w, 2)
        :param bsize: Number of pixels in each query
        :return: Predicted image (B, H * W, 3)
        """

        self.inp = inp
        if coord is None and cell is None:
            # Evaluate the efficiency of encoder only
            feat = self.encoder(inp)
            return None

        # Adjust the number of query pixels for different GPU memory limits.
        # Using lmf, we can query a 4k image simultaneously with 12GB GPU memory.
        self.query_bsize = bsize if bsize is not None else int(2160 * 3840 * 0.5)
        self.query_bsize = math.ceil(coord.shape[1] / math.ceil(coord.shape[1] / self.query_bsize))

        feat = self.gen_feats(inp, inp_coord)
        mod = self.gen_modulations(feat, cell)
        if self.mod_input:
            self.coeff = mod[:, self.mod_dim:self.mod_dim + self.mod_coef_dim, :, :]
            self.freqq = mod[:, self.mod_dim + self.mod_coef_dim:, :, :]

        if self.training:
            out = self.query_rgb(self.coeff, self.freqq, mod, coord, cell)
        else:
            if self.cmsr and self.updating_cmsr:
                # Update the Scale2mod Table for CMSR
                self.update_scale2mean(self.coeff, self.freqq, mod, coord, cell)
                return None

            out_of_distribution = coord.shape[1] > (self.max_scale ** 2) * inp.shape[-2] * inp.shape[-1]
            if self.cmsr and out_of_distribution:
                # Only use CMSR for out-of-training scales
                out = self.query_rgb_cmsr(self.coeff, self.freqq, mod, coord, cell)
            else:
                out = self.batched_query_rgb(self.coeff, self.freqq, mod, coord, cell, self.query_bsize)

            # self.t_total.append(time.time() - self.t1)
            #if len(self.t_total) >= 100:
            #    print(sum(self.t_total[1:]) / (len(self.t_total) - 1))

        return out




