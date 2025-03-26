import argparse
import os
import math
from functools import partial

import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from utils import make_coord

def batched_predict(model, inp, coord, cell, bsize):
    """
    Perform batched predictions using a model on provided inputs.

    :param model: The model used for generating predictions.
    :param inp: Input data to the model. Typically, this is a batch of images or features.
    :param coord: Coordinates associated with the input data, used for models that require spatial context.
    :param cell: Scaling factors or additional data related to each coordinate, supporting the model's prediction.
    :param bsize: Batch size used to split the data into manageable chunks during prediction.
    :return: The concatenated tensor of predictions for all batches.
    """

    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def clip_test(img_lq, model):
        
        
        sf = 6
        b, c, h, w = img_lq.size()
      
        tile = 192
        tile_overlap =32
        
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                
                target_size = (round(in_patch.shape[-2]*sf), 
                               round(in_patch.shape[-1]*sf))

                hr_coord = make_coord(target_size[-2:]).unsqueeze(0).expand(b, -1, 2).to(img_lq)  #.cuda() TODO
                cell = torch.ones_like(hr_coord).to(img_lq)  #.cuda()   TODO
                cell[:, :, 0] *= 2 / target_size[-2]
                cell[:, :, 1] *= 2 / target_size[-1]

                out_patch = model(in_patch, hr_coord, cell) 
                
                ih, iw = in_patch.shape[-2:]
                shape = [in_patch.shape[0], round(ih * sf), round(iw * sf), 3]
                out_patch = out_patch.view(*shape).permute(0, 3, 1, 2).contiguous()

                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)
        output = output.view(b, 3, -1).permute(0,2,1).contiguous()
      
        return output
def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
              verbose=False, save_folder=None):
    """
    Evaluate the Peak Signal-to-Noise Ratio (PSNR) of a model over a dataset loaded through a specified loader.

    :param loader: The DataLoader providing the input data.
    :param model: The model to be evaluated.
    :param data_norm: Normalization parameters for input and ground truth data.
    :param eval_type: Type of evaluation to perform, e.g., specific scales or datasets.
    :param eval_bsize: Batch size for evaluation, can be None for full-batch processing.
    :param window_size: The size of windowing for input padding, useful in SwinIR.
    :param scale_max: Maximum scale factor used in model training.
    :param fast: If True, evaluates using none-batched method.
    :param verbose: If True, provides detailed progress output.
    :param save_folder: Folder path where the resulting images are saved.
    :return: The average PSNR over all evaluated data.
    """

    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        scale = scale_max
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    index = 1
    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['coord']
            cell = batch['cell']

        # Cell clip for extrapolation
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell*max(scale/scale_max, 1))
        else:
            if fast:
                with torch.no_grad():
                    #pred = clip_test(inp,model)
                    pred = model(inp, coord, cell*max(scale/scale_max, 1))
            else:
                pred = batched_predict(model, inp, coord, cell*max(scale/scale_max, 1), eval_bsize)
            
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        # save sr image
        save_folder = save_folder if save_folder is not None else eval_type
        if save_folder is not None:
            ih, iw = batch['inp'].shape[-2:]
            save_img = pred.view(round((ih + h_pad) * scale),
                                 round((iw + w_pad) * scale), 3).permute(2, 0, 1).cpu()
            save_path = "./outputs/" + save_folder
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            index_str = str(index) if index >= 100 else ('0' + str(index) if index >= 10 else '00' + str(index))
            transforms.ToPILImage()(save_img).save(save_path + "/" + save_folder + "_" + index_str + ".png")
            index += 1

        if eval_type is not None:  # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
            
        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
            
    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=True)  # Set fast to True for LMF, False for original LIIF/LTE/CiaoSR
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cmsr', default=False)
    parser.add_argument('--cmsr_mse', default=0.00002)
    parser.add_argument('--cmsr_path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8, pin_memory=True)

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

    model_spec = torch.load(args.model)['model']
    model_spec["args"]["cmsr_spec"] = cmsr_spec
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader,
                    model,
                    data_norm=config.get('data_norm'),
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'),
                    window_size=int(args.window),
                    scale_max=int(args.scale_max),
                    fast=args.fast,
                    verbose=True)
    print('result: {:.4f}'.format(res))
