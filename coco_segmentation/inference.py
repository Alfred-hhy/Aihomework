"""
Inference script for COCO Semantic Segmentation
支持 TTA (Test Time Augmentation) 和输出单通道灰度 mask
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import TestDataset, test_collate_fn, NUM_CLASSES
from models import create_model


def load_checkpoint(model, checkpoint_path):
    """加载模型权重"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 处理 DDP 保存的权重 (移除 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    print("Checkpoint loaded successfully!")


def get_tta_transforms(input_size, scales=[1.0], flip=False):
    """
    获取 TTA 变换
    """
    transforms_list = []

    for scale in scales:
        h = int(input_size[0] * scale)
        w = int(input_size[1] * scale)

        # 正常
        transforms_list.append({
            'transform': A.Compose([
                A.Resize(height=h, width=w),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            'flip': False,
            'scale': scale,
        })

        # 水平翻转
        if flip:
            transforms_list.append({
                'transform': A.Compose([
                    A.Resize(height=h, width=w),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]),
                'flip': True,
                'scale': scale,
            })

    return transforms_list


@torch.no_grad()
def predict_single_image(model, image, config, device):
    """
    对单张图片进行预测 (支持 TTA)

    Args:
        model: 模型
        image: [H, W, 3] numpy array (RGB)
        config: 配置
        device: 设备

    Returns:
        pred: [H, W] numpy array, 预测类别
    """
    inf_config = config['inference']
    input_size = inf_config['input_size']
    use_tta = inf_config.get('use_tta', False)
    tta_scales = inf_config.get('tta_scales', [1.0])
    tta_flip = inf_config.get('tta_flip', False)

    original_size = image.shape[:2]  # (H, W)

    if use_tta:
        tta_transforms = get_tta_transforms(input_size, tta_scales, tta_flip)
    else:
        tta_transforms = get_tta_transforms(input_size, [1.0], False)

    all_probs = []

    for tta in tta_transforms:
        # 应用变换
        transformed = tta['transform'](image=image)
        img_tensor = transformed['image'].unsqueeze(0).to(device)

        # 推理
        with autocast():
            logits = model(img_tensor)  # [1, C, H, W]

        # 上采样到原始尺寸
        logits = F.interpolate(
            logits,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )

        # 如果是翻转的，需要翻转回来
        if tta['flip']:
            logits = torch.flip(logits, dims=[3])

        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)

    # 平均所有 TTA 结果
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    pred = avg_probs.argmax(dim=1).squeeze(0)

    return pred.cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict_batch(model, images, original_sizes, config, device):
    """
    批量预测 (不使用 TTA，用于快速推理)

    Args:
        model: 模型
        images: [B, 3, H, W] tensor
        original_sizes: list of (H, W) tuples
        config: 配置
        device: 设备

    Returns:
        preds: list of [H, W] numpy arrays
    """
    images = images.to(device)

    with autocast():
        logits = model(images)  # [B, C, H, W]

    preds = []
    for i in range(logits.shape[0]):
        # 上采样到原始尺寸
        logit = logits[i:i+1]
        h, w = original_sizes[i]
        logit = F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=False)
        pred = logit.argmax(dim=1).squeeze(0)
        preds.append(pred.cpu().numpy().astype(np.uint8))

    return preds


def run_inference(config, checkpoint_path, output_dir, use_tta=None):
    """
    运行推理

    Args:
        config: 配置
        checkpoint_path: 模型权重路径
        output_dir: 输出目录
        use_tta: 是否使用 TTA (None 表示使用配置文件的设置)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = create_model(config)
    load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    # 测试数据集
    data_root = config['data']['root']
    test_dir = config['data']['test_images']

    # 是否使用 TTA
    if use_tta is not None:
        config['inference']['use_tta'] = use_tta

    if config['inference'].get('use_tta', False):
        print("Using TTA (Test Time Augmentation)")
        # TTA 模式：逐张图片处理
        test_images_dir = os.path.join(data_root, test_dir)
        image_files = sorted([
            f for f in os.listdir(test_images_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

        print(f"Found {len(image_files)} test images")

        for img_name in tqdm(image_files, desc='Inference with TTA'):
            img_path = os.path.join(test_images_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pred = predict_single_image(model, image, config, device)

            # 保存为灰度图
            save_name = os.path.splitext(img_name)[0] + '.png'
            save_path = output_dir / save_name
            cv2.imwrite(str(save_path), pred)

    else:
        print("Using batch inference (no TTA)")
        # 批量模式
        from data import get_test_transforms

        test_transform = get_test_transforms(config)
        test_dataset = TestDataset(data_root, test_dir, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            collate_fn=test_collate_fn,
            pin_memory=True,
        )

        print(f"Found {len(test_dataset)} test images")

        for images, names, sizes in tqdm(test_loader, desc='Inference'):
            preds = predict_batch(model, images, sizes, config, device)

            for pred, name in zip(preds, names):
                save_name = os.path.splitext(name)[0] + '.png'
                save_path = output_dir / save_name
                cv2.imwrite(str(save_path), pred)

    print(f"\nPredictions saved to {output_dir}")
    print(f"Total images processed: {len(list(output_dir.glob('*.png')))}")


def main():
    parser = argparse.ArgumentParser(description='COCO Semantic Segmentation Inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Output directory for predictions')
    parser.add_argument('--tta', action='store_true',
                        help='Use TTA (overrides config)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable TTA (overrides config)')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # TTA 设置
    use_tta = None
    if args.tta:
        use_tta = True
    elif args.no_tta:
        use_tta = False

    run_inference(config, args.checkpoint, args.output, use_tta)


if __name__ == '__main__':
    main()
