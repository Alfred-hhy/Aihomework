"""
Data augmentation transforms using albumentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(config):
    """
    获取训练数据增强变换
    """
    aug_config = config['augmentation']['train']
    input_size = aug_config['input_size']

    transforms_list = [
        # 先缩放，然后padding到512x512，最后随机裁剪
        A.LongestMaxSize(max_size=int(input_size[0] * 1.5)),
        A.PadIfNeeded(min_height=input_size[0], min_width=input_size[1], border_mode=0, value=0),
        A.RandomCrop(height=input_size[0], width=input_size[1], p=1.0),
    ]

    # 水平翻转
    if aug_config.get('horizontal_flip', True):
        transforms_list.append(A.HorizontalFlip(p=0.5))

    # 颜色增强
    if aug_config.get('color_jitter', True):
        transforms_list.extend([
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3,
            ),
        ])

    # 高斯模糊
    if aug_config.get('gaussian_blur', True):
        transforms_list.append(
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.3)
        )

    # 归一化和转换为 tensor
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_val_transforms(config):
    """
    获取验证/测试数据增强变换
    """
    aug_config = config['augmentation']['val']
    input_size = aug_config['input_size']

    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_test_transforms(config):
    """
    获取测试数据变换
    """
    inf_config = config['inference']
    input_size = inf_config['input_size']

    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(config):
    """
    获取 TTA (Test Time Augmentation) 变换列表
    返回多组变换用于推理时集成
    """
    inf_config = config['inference']
    input_size = inf_config['input_size']
    scales = inf_config.get('tta_scales', [1.0])
    use_flip = inf_config.get('tta_flip', False)

    transforms_list = []

    for scale in scales:
        h = int(input_size[0] * scale)
        w = int(input_size[1] * scale)

        # 正常
        transforms_list.append({
            'transform': A.Compose([
                A.Resize(height=h, width=w),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]),
            'flip': False,
            'scale': scale,
        })

        # 水平翻转
        if use_flip:
            transforms_list.append({
                'transform': A.Compose([
                    A.Resize(height=h, width=w),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ]),
                'flip': True,
                'scale': scale,
            })

    return transforms_list
