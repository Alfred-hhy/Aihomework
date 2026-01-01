"""
SegFormer model wrapper for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormerWrapper(nn.Module):
    """
    SegFormer 模型封装
    支持从 HuggingFace 加载预训练权重
    """

    def __init__(
        self,
        num_classes: int = 81,
        backbone: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        pretrained: bool = True,
    ):
        """
        Args:
            num_classes: 类别数量 (包括背景)
            backbone: HuggingFace 模型名称或路径
            pretrained: 是否加载预训练权重
        """
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            print(f"Loading pretrained SegFormer from {backbone}...")
            # 从预训练模型加载，但修改分类头的类别数
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # 从配置创建模型
            config = SegformerConfig.from_pretrained(backbone)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

        print(f"SegFormer initialized with {num_classes} classes")

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            logits: 分割输出 [B, num_classes, H, W] (上采样到输入尺寸)
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]

        # 上采样到输入尺寸
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return logits

    def get_param_groups(self, lr: float, weight_decay: float):
        """
        获取参数组 (用于不同学习率)
        - backbone 使用较小的学习率
        - 分类头使用正常学习率
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if 'decode_head' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr, 'weight_decay': weight_decay},
        ]


def create_model(config):
    """
    根据配置创建模型
    """
    model_config = config['model']

    if model_config['name'] == 'segformer':
        model = SegFormerWrapper(
            num_classes=model_config['num_classes'],
            backbone=model_config['backbone'],
            pretrained=model_config['pretrained'],
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")

    return model
