"""
COCO Semantic Segmentation Training Script
支持 DDP 多卡训练、混合精度、TensorBoard 可视化
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (
    COCOSegmentationDataset,
    collate_fn,
    get_train_transforms,
    get_val_transforms,
    NUM_CLASSES,
)
from models import create_model, create_loss
from utils import SegmentationMetrics, visualize_prediction


def setup_distributed(rank, world_size):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_polynomial_lr_scheduler(optimizer, total_iters, warmup_iters=0, power=0.9, min_lr=1e-6):
    """
    Polynomial learning rate decay with optional warmup
    """
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return current_iter / max(warmup_iters, 1)
        else:
            progress = (current_iter - warmup_iters) / max(total_iters - warmup_iters, 1)
            return max(min_lr / optimizer.defaults['lr'], (1 - progress) ** power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    config,
    writer,
    rank=0,
):
    """训练一个 epoch"""
    model.train()
    metrics = SegmentationMetrics(NUM_CLASSES)

    accumulation_steps = config['training'].get('accumulation_steps', 1)
    log_interval = config['logging'].get('log_every_n_steps', 50)

    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader

    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        # 混合精度前向传播
        with autocast(enabled=config['training'].get('use_amp', True)):
            outputs = model(images)
            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(outputs, masks)
                loss_dict = {'ce_loss': loss.item()}
            else:
                loss, loss_dict = criterion(outputs, masks)

            loss = loss / accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # 统计
        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        # 计算指标
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            metrics.update(preds, masks)

        # 日志
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            global_step = epoch * len(train_loader) + batch_idx

            writer.add_scalar('train/loss', total_loss / num_batches, global_step)
            writer.add_scalar('train/lr', current_lr, global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{k}', v, global_step)

            if isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{total_loss / num_batches:.4f}',
                    'lr': f'{current_lr:.2e}',
                })

    # Epoch 统计
    train_metrics = metrics.compute()
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, train_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, config, rank=0):
    """验证"""
    model.eval()
    metrics = SegmentationMetrics(NUM_CLASSES)

    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(val_loader, desc='Validation')
    else:
        pbar = val_loader

    for images, masks in pbar:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        with autocast(enabled=config['training'].get('use_amp', True)):
            outputs = model(images)
            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(outputs, masks)
            else:
                loss, _ = criterion(outputs, masks)

        total_loss += loss.item()
        num_batches += 1

        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)

    val_metrics = metrics.compute()
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, val_metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, save_path, is_best=False):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
    }

    torch.save(state, save_path)

    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(state, best_path)


def main(rank, world_size, config):
    """主训练函数"""
    is_distributed = world_size > 1

    if is_distributed:
        setup_distributed(rank, world_size)

    # 设置设备
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # 创建保存目录
    if rank == 0:
        save_dir = Path(config['checkpoint']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # 数据集
    data_root = config['data']['root']
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)

    train_dataset = COCOSegmentationDataset(
        root=data_root,
        ann_file=config['data']['train_ann'],
        image_dir=config['data']['train_images'],
        transform=train_transforms,
    )

    val_dataset = COCOSegmentationDataset(
        root=data_root,
        ann_file=config['data']['val_ann'],
        image_dir=config['data']['val_images'],
        transform=val_transforms,
    )

    # 数据加载器
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 模型
    model = create_model(config)
    model = model.to(device)

    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 损失函数和优化器
    criterion = create_loss(config)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # 学习率调度
    total_iters = config['training']['epochs'] * len(train_loader)
    warmup_iters = config['training'].get('warmup_epochs', 5) * len(train_loader)
    scheduler = get_polynomial_lr_scheduler(
        optimizer,
        total_iters,
        warmup_iters,
        min_lr=config['training'].get('min_lr', 1e-6),
    )

    # 混合精度
    scaler = GradScaler(enabled=config['training'].get('use_amp', True))

    # 训练循环
    best_miou = 0.0
    epochs = config['training']['epochs']

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Starting training with {world_size} GPU(s)")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Total epochs: {epochs}")
        print(f"{'='*60}\n")

    for epoch in range(epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, epoch, config, writer, rank
        )

        # 验证
        val_loss, val_metrics = validate(model, val_loader, criterion, config, rank)

        # 日志
        if rank == 0:
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/train_mIoU', train_metrics['mIoU'], epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)
            writer.add_scalar('epoch/val_mIoU', val_metrics['mIoU'], epoch)
            writer.add_scalar('epoch/val_pixel_acc', val_metrics['pixel_acc'], epoch)

            print(f"\nEpoch {epoch}/{epochs-1}")
            print(f"  Train Loss: {train_loss:.4f}, mIoU: {train_metrics['mIoU']*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, mIoU: {val_metrics['mIoU']*100:.2f}%, "
                  f"Pixel Acc: {val_metrics['pixel_acc']*100:.2f}%")

            # 保存检查点
            is_best = val_metrics['mIoU'] > best_miou
            if is_best:
                best_miou = val_metrics['mIoU']
                print(f"  New best mIoU: {best_miou*100:.2f}%")

            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics, str(save_dir / f'checkpoint_epoch_{epoch}.pth'),
                is_best=is_best
            )

    if rank == 0:
        writer.close()
        print(f"\nTraining completed! Best mIoU: {best_miou*100:.2f}%")

    if is_distributed:
        cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description='COCO Semantic Segmentation Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 检测分布式训练
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    else:
        local_rank = args.local_rank
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # 单卡或多卡训练
    if world_size > 1:
        main(local_rank, world_size, config)
    else:
        main(0, 1, config)
