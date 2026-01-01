"""
Visualization utilities for semantic segmentation
"""

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt


# 颜色调色板 (PASCAL VOC 风格)
def create_colormap(num_classes=81):
    """
    创建类别颜色映射
    """
    def bit_get(val, idx):
        return (val >> idx) & 1

    colormap = np.zeros((num_classes, 3), dtype=np.uint8)

    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bit_get(c, 0) << (7 - j))
            g = g | (bit_get(c, 1) << (7 - j))
            b = b | (bit_get(c, 2) << (7 - j))
            c = c >> 3
        colormap[i] = [r, g, b]

    return colormap


COLORMAP = create_colormap(81)


def mask_to_color(mask, colormap=None):
    """
    将分割 mask 转换为彩色图像
    Args:
        mask: [H, W] numpy array, 类别标签
        colormap: 颜色映射表
    Returns:
        [H, W, 3] numpy array, RGB 彩色图像
    """
    if colormap is None:
        colormap = COLORMAP

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label in np.unique(mask):
        if label < len(colormap):
            color_mask[mask == label] = colormap[label]

    return color_mask


def visualize_prediction(image, mask, pred, save_path=None):
    """
    可视化预测结果
    Args:
        image: [H, W, 3] 原始图像 (RGB)
        mask: [H, W] 真实标签
        pred: [H, W] 预测结果
        save_path: 保存路径
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image.transpose(1, 2, 0) * std + mean) * 255
        image = image.astype(np.uint8)

    mask_color = mask_to_color(mask)
    pred_color = mask_to_color(pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def overlay_mask(image, mask, alpha=0.5):
    """
    将 mask 叠加到原图上
    Args:
        image: [H, W, 3] RGB 图像
        mask: [H, W] 分割 mask
        alpha: 透明度
    Returns:
        [H, W, 3] 叠加后的图像
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image.transpose(1, 2, 0) * std + mean) * 255
        image = image.astype(np.uint8)

    mask_color = mask_to_color(mask)

    # 只叠加非背景区域
    non_bg = mask > 0
    overlay = image.copy()
    overlay[non_bg] = (
        alpha * mask_color[non_bg] + (1 - alpha) * image[non_bg]
    ).astype(np.uint8)

    return overlay


def save_mask_as_gray(mask, save_path):
    """
    保存 mask 为灰度图 (单通道)
    Args:
        mask: [H, W] numpy array 或 torch.Tensor
        save_path: 保存路径
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.uint8)
    cv2.imwrite(save_path, mask)
