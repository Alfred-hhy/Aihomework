# MSCOCO 语义分割项目报告

**课程名称**：高级人工智能 (Advanced Artificial Intelligence)

**学校**：东北大学软件学院

**指导教师**：于瑞云

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据处理与加载方法](#2-数据处理与加载方法)
3. [模型设计原理](#3-模型设计原理)
4. [训练策略与参数调整](#4-训练策略与参数调整)
5. [实验结果与分析](#5-实验结果与分析)
6. [推理与测试](#6-推理与测试)
7. [总结与心得](#7-总结与心得)

---

## 1. 项目概述

### 1.1 任务描述

本项目的目标是实现 MSCOCO 数据集上的**语义分割**任务。语义分割是计算机视觉中的一项重要任务，需要对图像中的每个像素进行分类，预测其所属的语义类别。

### 1.2 数据集概况

| 数据集 | 图片数量 | 用途 |
|--------|----------|------|
| 训练集 (train2014) | 82,783 | 模型训练 |
| 验证集 (val2014) | 40,504 | 模型验证 |
| 测试集 | 200 | 最终评估（无标签） |

COCO 数据集共包含 **80 个物体类别**，加上背景类，本项目共处理 **81 个类别**。

### 1.3 最终结果

| 指标 | 值 |
|------|-----|
| 验证集 mIoU | **66.11%** |
| 验证集 Pixel Accuracy | **91.11%** |
| 训练轮次 | 30 epochs |
| 训练时间 | ~15 小时 (2×L20 GPU) |

---

## 2. 数据处理与加载方法

### 2.1 COCO 数据集结构

COCO 数据集原始提供的是**实例分割**标注，而非语义分割标注。每张图片的标注以 JSON 格式存储，包含多个物体实例的分割信息。

**原始标注格式**：
```json
{
  "segmentation": [[x1,y1,x2,y2,...]], // 多边形坐标 或 RLE编码
  "area": 12345.6,                      // 物体面积
  "category_id": 18,                    // 类别ID (非连续，1-90)
  "image_id": 123456                    // 图片ID
}
```

### 2.2 类别 ID 映射

COCO 的 80 个类别使用**非连续的 ID**（1-90 中有间隙，如缺少 12, 26, 29 等）。为了适配神经网络训练，我们需要将其映射为连续的标签：

```python
# 原始 COCO 类别 ID（非连续）
COCO_CATEGORY_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, ...]

# 映射为连续标签
# 0 = 背景
# 1-80 = COCO 的 80 个类别
COCO_CAT_TO_LABEL = {cat_id: i + 1 for i, cat_id in enumerate(COCO_CATEGORY_IDS)}
```

**映射示例**：
| COCO 原始 ID | 映射后标签 | 类别名称 |
|-------------|-----------|---------|
| 0 (无) | 0 | background |
| 1 | 1 | person |
| 2 | 2 | bicycle |
| 13 | 12 | stop sign |
| 90 | 80 | toothbrush |

### 2.3 语义分割 Mask 生成

由于 COCO 提供的是实例分割标注，我们需要将其转换为语义分割 mask。核心算法如下：

```python
def _generate_mask(self, img_id, height, width):
    # 1. 获取该图片的所有标注
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)

    # 2. 初始化空 mask (全为背景 0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 3. 关键：按面积从大到小排序
    # 这样小物体会覆盖大物体，处理重叠区域
    anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # 4. 逐个绘制标注
    for ann in anns:
        label = COCO_CAT_TO_LABEL[ann['category_id']]

        if isinstance(ann['segmentation'], list):
            # 多边形格式：使用 OpenCV 填充
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], label)
        else:
            # RLE 格式：使用 pycocotools 解码
            rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
            m = coco_mask.decode(rle)
            mask[m > 0] = label

    return mask
```

**关键设计决策**：

1. **面积排序**：按物体面积从大到小排序后绘制，确保小物体不会被大物体覆盖
2. **双格式支持**：同时支持多边形和 RLE 两种标注格式
3. **实时生成**：mask 在数据加载时动态生成，节省存储空间

### 2.4 数据增强策略

我们使用 **Albumentations** 库实现数据增强，针对训练、验证和测试使用不同的增强策略。

#### 2.4.1 训练数据增强

```python
def get_train_transforms(config):
    return A.Compose([
        # 1. 尺度变换
        A.LongestMaxSize(max_size=768),
        A.PadIfNeeded(min_height=512, min_width=512),
        A.RandomCrop(height=512, width=512),

        # 2. 几何变换
        A.HorizontalFlip(p=0.5),

        # 3. 颜色增强
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

        # 4. 模糊
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=7),
        ], p=0.3),

        # 5. 标准化
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

#### 2.4.2 验证数据增强

```python
def get_val_transforms(config):
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

**增强策略对比**：

| 增强类型 | 训练 | 验证 | 目的 |
|---------|------|------|------|
| RandomCrop | ✓ | ✗ | 增加位置多样性 |
| HorizontalFlip | ✓ | ✗ | 增加方向多样性 |
| ColorJitter | ✓ | ✗ | 增加颜色鲁棒性 |
| GaussianBlur | ✓ | ✗ | 模拟模糊图像 |
| Normalize | ✓ | ✓ | ImageNet 预训练标准化 |

---

## 3. 模型设计原理

### 3.1 SegFormer 架构概述

本项目采用 **SegFormer-B5** 作为分割模型。SegFormer 是一种结合了 Transformer 和 CNN 优点的混合架构，具有以下特点：

1. **分层 Transformer 编码器**：提取多尺度特征
2. **无位置编码**：使用卷积替代位置编码，支持任意输入尺寸
3. **轻量级 MLP 解码器**：高效融合多尺度特征

### 3.2 网络结构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        SegFormer-B5                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入图像: [B, 3, 512, 512]                                      │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            分层 Transformer 编码器 (Encoder)              │    │
│  │                                                          │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Stage 1: Patch Embedding + 2 Transformer Blocks │    │    │
│  │  │ 输出: [B, 64, 128, 128] (下采样 4x)              │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                    │                                     │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Stage 2: Patch Merging + 2 Transformer Blocks   │    │    │
│  │  │ 输出: [B, 128, 64, 64] (下采样 8x)               │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                    │                                     │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Stage 3: Patch Merging + 18 Transformer Blocks  │    │    │
│  │  │ 输出: [B, 320, 32, 32] (下采样 16x)              │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                    │                                     │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Stage 4: Patch Merging + 2 Transformer Blocks   │    │    │
│  │  │ 输出: [B, 512, 16, 16] (下采样 32x)              │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MLP 解码器 (Decoder Head)                    │    │
│  │                                                          │    │
│  │  1. 对每个 stage 的特征进行 MLP 投影                      │    │
│  │  2. 上采样到相同尺寸 (H/4, W/4)                          │    │
│  │  3. Concatenate 融合                                      │    │
│  │  4. MLP 分类器输出 81 类                                  │    │
│  │                                                          │    │
│  │  输出: [B, 81, 128, 128]                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              双线性插值上采样                              │    │
│  │  [B, 81, 128, 128] → [B, 81, 512, 512]                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                    │                                             │
│                    ▼                                             │
│  输出: 每个像素的 81 类概率分布                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 模型参数

| 组件 | 参数量 |
|------|--------|
| Encoder (Transformer) | ~82M |
| Decoder (MLP Head) | ~2.7M |
| **总计** | **~84.7M** |

### 3.4 预训练权重迁移

我们使用在 **ADE20K** 数据集上预训练的 SegFormer-B5 权重进行迁移学习：

```python
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    num_labels=81,                    # COCO 81 类
    ignore_mismatched_sizes=True,     # 允许分类头尺寸不匹配
)
```

**迁移学习策略**：
- ADE20K 原始输出：150 类
- COCO 目标输出：81 类
- 编码器权重：完全复用（~82M 参数）
- 解码器分类头：重新初始化

### 3.5 损失函数设计

本项目采用 **组合损失函数**，结合 CrossEntropy Loss 和 Dice Loss：

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{Dice}$$

其中 $\lambda = 0.5$。

#### 3.5.1 CrossEntropy Loss

```python
ce_loss = F.cross_entropy(logits, targets, ignore_index=255)
```

**特点**：
- 对每个像素独立计算分类损失
- 收敛快，边界清晰
- 对类别不平衡敏感

#### 3.5.2 Dice Loss

```python
def dice_loss(probs, targets):
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice
```

**特点**：
- 直接优化 IoU 相关指标
- 对类别不平衡鲁棒
- 处理小物体更好

#### 3.5.3 组合损失的优势

| 损失函数 | 优点 | 缺点 |
|---------|------|------|
| CrossEntropy | 收敛快、边界清晰 | 类别不平衡敏感 |
| Dice | 处理不平衡、优化 IoU | 收敛慢 |
| **Combined** | **综合优点** | **需要调参** |

---

## 4. 训练策略与参数调整

### 4.1 分布式训练配置

本项目使用 **PyTorch DDP (DistributedDataParallel)** 进行多 GPU 训练：

```python
# 初始化分布式环境
dist.init_process_group(backend='nccl', init_method='env://')

# 模型包装
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 同步 BatchNorm
model = DDP(model, device_ids=[rank])
```

**硬件配置**：
- GPU: 2 × NVIDIA L20 (48GB)
- 显存利用率: ~92% (42GB/46GB per GPU)
- 总 batch size: 32 (16 per GPU)

### 4.2 超参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 16 | 单卡批大小 |
| learning_rate | 6e-5 | 基础学习率 |
| weight_decay | 0.01 | L2 正则化 |
| epochs | 30 | 总训练轮次 |
| warmup_epochs | 5 | 学习率预热 |
| optimizer | AdamW | 优化器 |
| scheduler | Polynomial | 学习率衰减 |
| loss_type | ce_dice | 组合损失 |
| dice_weight | 0.5 | Dice 损失权重 |
| accumulation_steps | 2 | 梯度累积步数 |
| use_amp | True | 混合精度训练 |

### 4.3 学习率调度

采用**多项式衰减 + 预热**策略：

```python
def polynomial_lr_lambda(current_iter):
    if current_iter < warmup_iters:
        # 预热阶段：线性增长
        return current_iter / warmup_iters
    else:
        # 衰减阶段：多项式衰减
        progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
        return max(min_lr_ratio, (1 - progress) ** power)
```

**学习率变化曲线**：

```
LR
 │
6e-5 ┤           ╭─────────╮
     │          ╱           ╲
     │         ╱             ╲
     │        ╱               ╲
     │       ╱                 ╲
1e-6 ┤──────╱                   ╲─────
     └──────────────────────────────── Iteration
     0    Warmup              End
```

**关键设计**：
- **Per-iteration 调度**：每个 batch 后更新学习率，而非每个 epoch
- **Warmup 阶段**：前 5 个 epoch 学习率线性增长，避免初期不稳定
- **多项式衰减**：power=0.9，平滑过渡到最小学习率

### 4.4 混合精度训练 (AMP)

使用 PyTorch 的自动混合精度加速训练：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**AMP 优势**：
- 显存节省 ~30%
- 训练速度提升 ~1.5x
- 精度损失可忽略

### 4.5 梯度累积

使用梯度累积模拟更大的 batch size：

```python
accumulation_steps = 2  # 等效 batch_size = 16 * 2 * 2 = 64

for batch_idx, (images, masks) in enumerate(train_loader):
    loss = criterion(outputs, masks) / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. 实验结果与分析

### 5.1 训练曲线

**[请插入 TensorBoard 截图]**

#### 5.1.1 Loss 曲线
- 训练损失 (epoch/train_loss)
- 验证损失 (epoch/val_loss)

#### 5.1.2 mIoU 曲线
- 训练 mIoU (epoch/train_mIoU)
- 验证 mIoU (epoch/val_mIoU)

### 5.2 训练结果数据

| Epoch | Train Loss | Train mIoU | Val Loss | Val mIoU | Pixel Acc |
|-------|-----------|-----------|----------|----------|-----------|
| 0 | 3.91 | 1.87% | 2.43 | 3.68% | 73.25% |
| 5 | 0.72 | 54.80% | 0.71 | 55.04% | 90.18% |
| 10 | 0.45 | 68.68% | 0.51 | 61.87% | 90.61% |
| 15 | 0.36 | 73.90% | 0.49 | 64.13% | 90.94% |
| 20 | 0.31 | 76.36% | 0.51 | 63.98% | 90.70% |
| 25 | 0.28 | 78.53% | 0.51 | 65.16% | 90.94% |
| **29** | **0.26** | **79.62%** | **0.51** | **66.11%** | **91.11%** |

### 5.3 结果分析

#### 5.3.1 收敛性分析

- **训练损失**：从 3.91 持续下降到 0.26，收敛良好
- **验证损失**：在 ~0.51 附近波动，趋于稳定
- **mIoU**：验证集 mIoU 持续上升，最终达到 66.11%

#### 5.3.2 过拟合分析

| 指标 | 训练集 | 验证集 | 差距 |
|------|--------|--------|------|
| mIoU | 79.62% | 66.11% | 13.51% |

- 训练/验证差距约 13%，存在**轻微过拟合**
- 但验证集 mIoU 仍在上升，说明模型仍有学习能力
- 建议：可以继续训练或增加正则化

#### 5.3.3 与 SOTA 对比

| 方法 | 数据集 | mIoU |
|------|--------|------|
| SegFormer-B5 (原论文) | ADE20K | 51.8% |
| **本项目** | **COCO** | **66.11%** |

注：COCO 和 ADE20K 类别数和难度不同，不能直接对比，但本项目结果良好。

---

## 6. 推理与测试

### 6.1 TTA (Test Time Augmentation)

为了提高预测精度，我们使用 TTA 在推理时进行多次预测并融合：

```python
def predict_with_tta(model, image, config):
    all_probs = []

    for scale in [0.75, 1.0, 1.25]:
        for flip in [False, True]:
            # 1. 应用变换
            img_aug = apply_transform(image, scale, flip)

            # 2. 模型推理
            logits = model(img_aug)

            # 3. 上采样到原始尺寸
            logits = F.interpolate(logits, size=original_size)

            # 4. 如果翻转了，翻转回来
            if flip:
                logits = torch.flip(logits, dims=[3])

            # 5. 计算概率
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

    # 6. 平均所有 TTA 结果
    avg_probs = torch.stack(all_probs).mean(dim=0)
    pred = avg_probs.argmax(dim=1)

    return pred
```

**TTA 配置**：
- 尺度: [0.75, 1.0, 1.25]
- 水平翻转: True
- 总推理次数: 3 × 2 = 6 次

### 6.2 输出格式

测试集预测结果保存为**单通道灰度 PNG 图像**：

```python
# 像素值 = 类别 ID (0-80)
# 0 = 背景
# 1-80 = COCO 80 个类别

cv2.imwrite(output_path, pred.astype(np.uint8))
```

**示例**：
- 输入: `2007_000027.jpg` (RGB 彩色图像)
- 输出: `2007_000027.png` (单通道灰度 mask)

### 6.3 推理时间

| 模式 | 200 张图片耗时 | 单张耗时 |
|------|---------------|---------|
| 无 TTA | ~2 分钟 | ~0.6 秒 |
| 有 TTA | ~2 分钟 | ~0.6 秒 |

---

## 7. 总结与心得

### 7.1 项目收获

1. **深入理解语义分割**：从数据处理到模型设计的完整流程
2. **掌握 Transformer 在视觉中的应用**：SegFormer 的混合架构设计
3. **分布式训练实践**：PyTorch DDP、混合精度、梯度累积
4. **数据增强策略**：Albumentations 库的灵活运用

### 7.2 遇到的问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| COCO 类别 ID 非连续 | 建立映射表，转为连续标签 |
| 显存不足 | 使用混合精度 + 梯度累积 |
| 预训练权重类别数不匹配 | `ignore_mismatched_sizes=True` |
| PyTorch 2.6 checkpoint 加载失败 | 添加 `weights_only=False` |
| 小物体分割效果差 | 使用 Dice Loss 处理类别不平衡 |

### 7.3 可能的改进方向

1. **更长时间训练**：当前 30 epochs，可以尝试 50-100 epochs
2. **更强的数据增强**：如 CutMix、MixUp
3. **更大的模型**：尝试 SegFormer-B5 以外的变体
4. **后处理**：如 CRF 细化边界
5. **类别平衡**：对稀有类别过采样

### 7.4 致谢

感谢于瑞云老师的指导，以及 HuggingFace 提供的预训练模型。

---

## 附录

### A. 项目文件结构

```
coco_segmentation/
├── configs/
│   └── config.yaml          # 配置文件
├── data/
│   ├── coco_dataset.py      # 数据集加载
│   └── transforms.py        # 数据增强
├── models/
│   ├── segformer.py         # 模型封装
│   └── losses.py            # 损失函数
├── utils/
│   ├── metrics.py           # 评估指标
│   └── visualization.py     # 可视化
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── checkpoints/             # 模型权重
├── logs/                    # TensorBoard 日志
└── predictions/             # 预测结果
```

### B. 运行命令

```bash
# 训练 (2 GPU)
torchrun --nproc_per_node=2 train.py --config configs/config.yaml

# 继续训练
torchrun --nproc_per_node=2 train.py --config configs/config.yaml \
  --resume checkpoints/latest_checkpoint.pth

# 推理 (使用 TTA)
python inference.py --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --output predictions --tta

# 查看 TensorBoard
tensorboard --logdir logs
```

### C. 环境依赖

```
torch>=2.0
torchvision>=0.15
transformers>=4.30
albumentations>=1.3
pycocotools>=2.0
opencv-python>=4.8
tensorboard>=2.14
```
