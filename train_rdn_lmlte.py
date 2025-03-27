import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml
import time
import datetime
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import models
import models.lmlte_rdn  # 导入RDN-LMLTE模型
import utils
from datasets import make_dataset, make_data_loader

from models.lmlte_rdn import LMLTE_RDN  # 确保导入LMLTE_RDN模型

def make_data_loaders(config):
    train_dataset = make_dataset(config.get('train_dataset'))
    val_dataset = make_dataset(config.get('val_dataset'))
    train_loader = make_data_loader(train_dataset, 
                                    config.get('batch_size'),
                                    True,
                                    num_workers=8,
                                    pin_memory=True)
    val_loader = make_data_loader(val_dataset, 
                                 config.get('batch_size'),
                                 False,
                                 num_workers=4,
                                 pin_memory=True)
    return train_loader, val_loader

def prepare_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建模型
    model = models.make(config['model']).to(device)
    
    # 输出模型参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数量: {params}')
    
    # 构建优化器
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    
    # 构建学习率调度器
    lr_scheduler = utils.make_lr_scheduler(optimizer, config.get('multi_step_lr'))
    
    # 设置损失函数
    loss_fn = nn.L1Loss()
    
    return device, model, optimizer, lr_scheduler, loss_fn

def train(train_loader, model, optimizer, loss_fn, device, epoch, config):
    model.train()
    loss_all = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        inp = batch['inp']
        gt = batch['gt']
        
        # 数据归一化
        data_norm = config.get('data_norm')
        if data_norm:
            inp = utils.normalize_data(inp, data_norm.get('inp'))
            gt = utils.normalize_data(gt, data_norm.get('gt'))
                
        # 前向传播
        pred = model(inp, batch['coord'], batch['cell'])
        
        # 计算损失
        loss = loss_fn(pred, gt)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_all += loss.item()
        
        # 打印训练进度
        if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == len(train_loader):
            print(f'Train Epoch: {epoch} [{batch_idx+1}/{len(train_loader)}]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Time: {time.time() - start_time:.2f}s')
            start_time = time.time()
    
    return loss_all / len(train_loader)

def validate(val_loader, model, loss_fn, device, config):
    model.eval()
    loss_all = 0
    psnr_all = 0
    count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            inp = batch['inp']
            gt = batch['gt']
            
            # 数据归一化
            data_norm = config.get('data_norm')
            if data_norm:
                inp = utils.normalize_data(inp, data_norm.get('inp'))
                gt = utils.normalize_data(gt, data_norm.get('gt'))
            
            # 前向传播
            pred = model(inp, batch['coord'], batch['cell'])
            
            # 计算损失和PSNR
            loss = loss_fn(pred, gt)
            loss_all += loss.item()
            
            # 计算PSNR
            if data_norm:
                pred = utils.denormalize_data(pred, data_norm.get('gt'))
                gt = utils.denormalize_data(gt, data_norm.get('gt'))
                
            psnr = utils.calc_psnr(pred, gt)
            psnr_all += psnr
            count += 1
            
    return loss_all / len(val_loader), psnr_all / count

def main(config, args):
    # 准备保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置文件
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    # 准备数据加载器
    train_loader, val_loader = make_data_loaders(config['train_dataset']), None
    if config.get('val_dataset'):
        _, val_loader = make_data_loaders(config)
    
    # 准备训练
    device, model, optimizer, lr_scheduler, loss_fn = prepare_training(config)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    
    # 加载检查点（如果有）
    resume = config.get('resume')
    if resume is not None:
        print(f'Loading checkpoint: {resume}')
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Loaded checkpoint at epoch {checkpoint["epoch"]}')
    else:
        start_epoch = 1
    
    # 训练循环
    max_epoch = config.get('epoch_max', 1000)
    save_epoch = config.get('epoch_save', 100)
    val_epoch = config.get('epoch_val', 1)
    best_psnr = -1
    
    timer = utils.Timer()
    
    for epoch in range(start_epoch, max_epoch + 1):
        t_epoch_start = timer.t()
        
        # 训练一个epoch
        train_loss = train(train_loader, model, optimizer, loss_fn, device, epoch, config)
        
        # 打印训练信息
        print(f'Epoch: {epoch}/{max_epoch}\t Train Loss: {train_loss:.6f}')
        writer.add_scalar('train_loss', train_loss, epoch)
        
        # 更新学习率
        if lr_scheduler is not None:
            lr_scheduler.step()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # 验证
        if val_loader is not None and (epoch % val_epoch == 0):
            val_loss, val_psnr = validate(val_loader, model, loss_fn, device, config)
            print(f'Validation Loss: {val_loss:.6f}\t PSNR: {val_psnr:.2f}')
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_psnr', val_psnr, epoch)
            
            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'psnr': val_psnr,
                }
                if lr_scheduler is not None:
                    state['lr_scheduler'] = lr_scheduler.state_dict()
                torch.save(state, os.path.join(save_dir, 'epoch-best.pth'))
                print(f'Best PSNR: {best_psnr:.2f}')
        
        # 保存模型
        if epoch % save_epoch == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if lr_scheduler is not None:
                state['lr_scheduler'] = lr_scheduler.state_dict()
            torch.save(state, os.path.join(save_dir, f'epoch-{epoch}.pth'))
        
        # 保存最新模型
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if lr_scheduler is not None:
            state['lr_scheduler'] = lr_scheduler.state_dict()
        torch.save(state, os.path.join(save_dir, 'epoch-last.pth'))
        
        # 计算时间
        t = timer.t()
        prog = (epoch - start_epoch + 1) / (max_epoch - start_epoch + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        print(f'已用时间: {t_elapsed} ({epoch}/{max_epoch}) 预计总时间: {t_all}')
    
    writer.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train-lmf/train_rdn-lmlte_new.yaml')
    parser.add_argument('--save_dir', default='save/rdn_lm-lmlte_new')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('配置文件加载成功: {}'.format(args.config))
    print(f'使用设备: {args.gpu}')
    
    # 开始训练
    main(config, args) 