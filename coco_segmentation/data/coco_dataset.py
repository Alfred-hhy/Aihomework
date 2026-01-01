"""
COCO Semantic Segmentation Dataset
将 COCO instance annotations 转换为语义分割 mask
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import cv2


# COCO category IDs (不连续，共80个)
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

# COCO category_id -> 连续标签 (0=background, 1-80=categories)
COCO_CAT_TO_LABEL = {cat_id: i + 1 for i, cat_id in enumerate(COCO_CATEGORY_IDS)}
COCO_LABEL_TO_CAT = {v: k for k, v in COCO_CAT_TO_LABEL.items()}

# 类别名称 (用于可视化)
COCO_CATEGORY_NAMES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

NUM_CLASSES = 81  # 80 categories + 1 background


class COCOSegmentationDataset(Dataset):
    """
    COCO 语义分割数据集
    将 instance segmentation annotations 转换为 semantic segmentation masks
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        image_dir: str,
        transform=None,
        ignore_index: int = 255,
    ):
        """
        Args:
            root: 数据集根目录
            ann_file: 标注文件路径 (相对于root)
            image_dir: 图片目录名 (相对于root)
            transform: 数据增强变换
            ignore_index: 忽略的标签值
        """
        self.root = root
        self.transform = transform
        self.ignore_index = ignore_index

        self.image_dir = os.path.join(root, image_dir)
        ann_path = os.path.join(root, ann_file)

        # 加载 COCO 标注
        print(f"Loading COCO annotations from {ann_path}...")
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        print(f"Loaded {len(self.ids)} images")

    def __len__(self):
        return len(self.ids)

    def _generate_mask(self, img_id, height, width):
        """
        从 instance annotations 生成语义分割 mask
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 初始化 mask 为背景 (0)
        mask = np.zeros((height, width), dtype=np.uint8)

        # 按面积从大到小排序，小物体覆盖大物体
        anns = sorted(anns, key=lambda x: x['area'], reverse=True)

        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in COCO_CAT_TO_LABEL:
                continue

            label = COCO_CAT_TO_LABEL[cat_id]

            # 处理分割标注
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # 多边形格式
                    for seg in ann['segmentation']:
                        if len(seg) >= 6:  # 至少3个点
                            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], label)
                elif isinstance(ann['segmentation'], dict):
                    # RLE 格式
                    if isinstance(ann['segmentation']['counts'], list):
                        rle = coco_mask.frPyObjects(
                            ann['segmentation'], height, width
                        )
                    else:
                        rle = ann['segmentation']
                    m = coco_mask.decode(rle)
                    mask[m > 0] = label

        return mask

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # 加载图片
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成 mask
        height, width = image.shape[:2]
        mask = self._generate_mask(img_id, height, width)

        # 应用数据增强
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask


class TestDataset(Dataset):
    """
    测试数据集 (无标签)
    """

    def __init__(self, root: str, image_dir: str, transform=None):
        self.root = root
        self.image_dir = os.path.join(root, image_dir)
        self.transform = transform

        # 获取所有图片文件
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"Found {len(self.images)} test images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (H, W)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, img_name, original_size


def collate_fn(batch):
    """
    自定义 collate 函数，处理不同大小的图片
    """
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    # 处理 mask 可能是 tensor 或 numpy array 的情况
    mask_list = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            mask_list.append(m.long())
        else:
            mask_list.append(torch.as_tensor(m, dtype=torch.long))
    masks = torch.stack(mask_list, dim=0)
    return images, masks


def test_collate_fn(batch):
    """
    测试集的 collate 函数
    """
    images, names, sizes = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, names, sizes
