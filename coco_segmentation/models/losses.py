"""
Loss functions for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] 模型输出
            targets: [B, H, W] 真实标签
        """
        num_classes = logits.shape[1]

        # 创建有效像素 mask
        valid_mask = targets != self.ignore_index
        targets_valid = targets.clone()
        targets_valid[~valid_mask] = 0

        # Softmax
        probs = F.softmax(logits, dim=1)

        # One-hot encoding
        targets_one_hot = F.one_hot(targets_valid, num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Apply mask
        valid_mask = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Dice coefficient per class
        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs + targets_one_hot).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over classes (ignore background optionally)
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction='none',
            ignore_index=self.ignore_index,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int = 255,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = self.ce_weight * ce + self.dice_weight * dice

        return total_loss, {'ce_loss': ce.item(), 'dice_loss': dice.item()}


def create_loss(config):
    """
    根据配置创建损失函数
    """
    loss_type = config['training'].get('loss_type', 'ce_dice')
    dice_weight = config['training'].get('dice_weight', 0.5)

    if loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=255)
    elif loss_type == 'dice':
        return DiceLoss(ignore_index=255)
    elif loss_type == 'ce_dice':
        return CombinedLoss(
            ce_weight=1.0,
            dice_weight=dice_weight,
            ignore_index=255,
        )
    elif loss_type == 'focal':
        return FocalLoss(ignore_index=255)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
