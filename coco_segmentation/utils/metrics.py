"""
Metrics for semantic segmentation
"""

import torch
import numpy as np


class SegmentationMetrics:
    """
    语义分割评估指标
    计算 mIoU, pixel accuracy 等
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """重置统计"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, targets):
        """
        更新混淆矩阵
        Args:
            preds: [B, H, W] 或 [H, W] 预测结果
            targets: [B, H, W] 或 [H, W] 真实标签
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        preds = preds.flatten()
        targets = targets.flatten()

        # 移除忽略的像素
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        # 只统计有效范围内的预测
        valid_pred = (preds >= 0) & (preds < self.num_classes)
        valid_target = (targets >= 0) & (targets < self.num_classes)
        valid = valid_pred & valid_target

        preds = preds[valid]
        targets = targets[valid]

        # 更新混淆矩阵
        indices = self.num_classes * targets + preds
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        计算各项指标
        Returns:
            dict: 包含 mIoU, pixel_acc, class_iou 等
        """
        # IoU per class
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )

        # 避免除零
        union = np.maximum(union, 1)
        iou = intersection / union

        # 有效类别 (至少有一个像素)
        valid_classes = (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0)) > 0

        # mIoU (只计算有效类别)
        if valid_classes.sum() > 0:
            miou = iou[valid_classes].mean()
        else:
            miou = 0.0

        # Pixel accuracy
        correct = intersection.sum()
        total = self.confusion_matrix.sum()
        pixel_acc = correct / max(total, 1)

        # Mean accuracy per class
        class_acc = intersection / np.maximum(self.confusion_matrix.sum(axis=1), 1)
        mean_acc = class_acc[valid_classes].mean() if valid_classes.sum() > 0 else 0.0

        return {
            'mIoU': miou,
            'pixel_acc': pixel_acc,
            'mean_acc': mean_acc,
            'class_iou': iou,
            'class_acc': class_acc,
        }


def compute_miou(preds, targets, num_classes, ignore_index=255):
    """
    快速计算单个 batch 的 mIoU
    """
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(preds, targets)
    return metrics.compute()['mIoU']
