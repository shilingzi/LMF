import argparse
import os
from PIL import Image

import yaml
import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--fast', default=True)  # Set fast to True for LMF, False for original LIIF/LTE/CiaoSR
    parser.add_argument('--cmsr', default=False)
    parser.add_argument('--cmsr_mse', default=0.00002)
    parser.add_argument('--cmsr_path')
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU模式')
    args = parser.parse_args()

    # 根据参数选择设备
    if args.force_cpu:
        DEVICE = 'cpu'
    else:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    print(f"使用设备: {DEVICE}")

    # Maximum scale factor during training
    scale_max = 4

    if args.cmsr:
        try:
            # Test with CMSR
            with open(args.cmsr_path, 'r') as f:
                s2m_tables = yaml.load(f, Loader=yaml.FullLoader)
            cmsr_spec = {
                "mse_threshold": float(args.cmsr_mse),
                "path": args.cmsr_path,
                "s2m_tables": s2m_tables,
                "log": False,
            }
        except FileNotFoundError:
            cmsr_spec = None
    else:
        cmsr_spec = None

    # 显式指定加载到CPU，然后再迁移到适当的设备
    model_spec = torch.load(args.model, map_location='cpu')['model']
    model_spec["args"]["cmsr_spec"] = cmsr_spec
    model = models.make(model_spec).to(DEVICE)
    model_sd = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(model_sd, strict=False)
    model.eval()
    
    print(f"正在处理图像: {args.input}")
    print(f"放大倍数: {args.scale}")
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB')).to(DEVICE)

    h = int(img.shape[-2] * int(args.scale))
    w = int(img.shape[-1] * int(args.scale))
    scale = h / img.shape[-2]
    coord = make_coord((h, w)).to(DEVICE)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    cell_factor = max(scale/scale_max, 1)
    if args.fast:
        with torch.no_grad():
            pred = model(((img - 0.5) / 0.5).unsqueeze(0),
                         coord.unsqueeze(0), cell_factor * cell.unsqueeze(0))[0]
    else:
        pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(0),
                               coord.unsqueeze(0), cell_factor * cell.unsqueeze(0), bsize=30000)[0]

    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
    print(f"处理完成，结果已保存至: {args.output}")
