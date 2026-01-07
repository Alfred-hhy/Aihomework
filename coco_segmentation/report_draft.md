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
| 最佳 Epoch | **29** |
| 训练轮次 | 45 epochs |
| 训练时间 | ~22.5 小时 (2×L20 GPU) |

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
| epochs | 45 | 总训练轮次 |
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

以下是关键轮次的训练指标（完整45轮数据见附录D）：

| Epoch | Train Loss | Train mIoU | Val Loss | Val mIoU | Pixel Acc |
|-------|-----------|-----------|----------|----------|-----------|
| 1 | 1.8597 | 8.20% | 1.2984 | 13.57% | 83.12% |
| 5 | 0.7207 | 54.80% | 0.7101 | 55.04% | 90.18% |
| 9 | 0.4880 | 66.30% | 0.5375 | 61.73% | 90.54% |
| 10 | 0.4471 | 68.68% | 0.5089 | 61.87% | 90.61% |
| 15 | 0.3578 | 73.90% | 0.4899 | 64.13% | 90.94% |
| 20 | 0.3066 | 76.36% | 0.5135 | 63.98% | 90.70% |
| 25 | 0.2769 | 78.53% | 0.5127 | 65.16% | 90.94% |
| **29** | **0.2602** | **79.62%** | **0.5078** | **66.11%** | **91.11%** |
| 30 | 0.2584 | 79.31% | 0.5215 | 65.00% | 90.93% |
| 35 | 0.2430 | 80.34% | 0.5345 | 65.63% | 91.06% |
| 40 | 0.2310 | 81.04% | 0.5433 | 65.21% | 91.01% |
| 44 | 0.2215 | 81.94% | 0.5514 | 65.78% | 91.12% |

**关键观察**：
- **最佳验证集 mIoU** 出现在 **Epoch 29**（66.11%），而非最后一轮
- Epoch 30 之后，训练集指标持续提升，但验证集指标停滞不前
- 这表明模型在 Epoch 29 后开始**过拟合**

### 5.3 结果分析

#### 5.3.1 收敛性分析

- **训练损失**：从 1.86 持续下降到 0.22，收敛良好
- **验证损失**：前期快速下降，后期在 ~0.50-0.55 附近波动
- **mIoU**：验证集 mIoU 在 Epoch 29 达到峰值 66.11%

#### 5.3.2 过拟合分析

| 阶段 | Epoch | Train mIoU | Val mIoU | Gap |
|------|-------|-----------|----------|-----|
| Phase 2 结束 | 29 | 79.62% | 66.11% | 13.51% |
| Phase 3 结束 | 44 | 81.94% | 65.78% | **16.16%** |

**过拟合证据**：

1. **训练/验证差距扩大**：从 13.51% 扩大到 16.16%（+2.65%）
2. **验证损失上升**：从 0.508 上升到 0.551（+8.5%）
3. **验证 mIoU 停滞**：Train mIoU 持续上升至 81.94%，而 Val mIoU 停滞在 65-66%

**结论**：Epoch 29 后模型开始过拟合，因此 **best_model.pth 保存的是 Epoch 29 的权重**，这是验证集表现最佳的模型。

#### 5.3.3 与 SOTA 对比

| 方法 | 数据集 | mIoU |
|------|--------|------|
| SegFormer-B5 (原论文) | ADE20K | 51.8% |
| **本项目** | **COCO** | **66.11%** |

注：COCO 和 ADE20K 类别数和难度不同，不能直接对比，但本项目结果良好。

#### 5.3.4 三阶段训练分析

本项目训练分为三个阶段，展示了典型的深度学习训练曲线特征：

| 阶段 | Epochs | Val mIoU 变化 | 特点 |
|------|--------|--------------|------|
| Phase 1 | 1-9 | 13.57% → 61.73% | 快速学习期 |
| Phase 2 | 10-29 | 61.87% → **66.11%** | 稳定提升期 |
| Phase 3 | 30-44 | 65.00% → 65.78% | 过拟合期 |

**各阶段分析**：

- **Phase 1 (快速学习期)**：
  - 模型从随机初始化的分类头开始学习
  - mIoU 提升 +48.16%（13.57% → 61.73%）
  - 损失快速下降，特征提取能力快速形成

- **Phase 2 (稳定提升期)**：
  - 模型在已学习特征的基础上精细调整
  - mIoU 提升 +4.38%（61.87% → 66.11%）
  - 在 Epoch 29 达到最佳验证性能

- **Phase 3 (过拟合期)**：
  - 继续训练15轮，但验证集指标未能超越 Epoch 29
  - 训练集 mIoU 持续上升（79.31% → 81.94%）
  - 验证集 mIoU 波动但无明显提升（65.00% → 65.78%）
  - 验证损失呈上升趋势，确认过拟合

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

1. **更强的数据增强**：如 CutMix、MixUp、RandAugment
2. **正则化技术**：Label Smoothing、Dropout 增强
3. **更大的输入分辨率**：从 512×512 提升到 640×640 或更高
4. **模型集成**：多模型投票或加权平均
5. **后处理**：如 CRF 细化边界
6. **类别平衡**：对稀有类别过采样或使用 Class-Balanced Loss

**注**：实验表明继续训练（超过 30 轮）会导致过拟合，因此不建议简单增加训练轮次。

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

### D. 完整训练数据（45轮）

以下是完整的 44 轮训练数据，可用于绘制 Loss 和 mIoU 曲线图。

#### Phase 1 (Epochs 1-9)

| Epoch | Train Loss | Val Loss | Train mIoU | Val mIoU | Pixel Acc |
|-------|-----------|----------|-----------|----------|-----------|
| 1 | 1.8597 | 1.2984 | 8.20% | 13.57% | 83.12% |
| 2 | 1.2017 | 0.9466 | 21.01% | 31.41% | 87.43% |
| 3 | 0.9124 | 0.7944 | 37.54% | 43.55% | 89.13% |
| 4 | 0.7793 | 0.7338 | 49.25% | 50.78% | 89.97% |
| 5 | 0.7207 | 0.7101 | 54.80% | 55.04% | 90.18% |
| 6 | 0.6771 | 0.6812 | 60.82% | 59.39% | 90.61% |
| 7 | 0.6487 | 0.6672 | 64.19% | 60.17% | 90.25% |
| 8 | 0.5738 | 0.5695 | 65.52% | 60.06% | 90.43% |
| 9 | 0.4880 | 0.5375 | 66.30% | 61.73% | 90.54% |

#### Phase 2 (Epochs 10-29)

| Epoch | Train Loss | Val Loss | Train mIoU | Val mIoU | Pixel Acc |
|-------|-----------|----------|-----------|----------|-----------|
| 10 | 0.4471 | 0.5089 | 68.68% | 61.87% | 90.61% |
| 11 | 0.4167 | 0.5018 | 69.64% | 62.56% | 90.69% |
| 12 | 0.3975 | 0.4984 | 70.68% | 62.35% | 90.66% |
| 13 | 0.3782 | 0.4925 | 72.11% | 63.32% | 90.83% |
| 14 | 0.3676 | 0.4936 | 72.77% | 63.31% | 90.82% |
| 15 | 0.3578 | 0.4899 | 73.90% | 64.13% | 90.94% |
| 16 | 0.3534 | 0.5005 | 74.13% | 64.11% | 90.70% |
| 17 | 0.3356 | 0.5022 | 75.23% | 64.30% | 90.77% |
| 18 | 0.3298 | 0.5057 | 75.59% | 64.25% | 90.98% |
| 19 | 0.3188 | 0.5098 | 75.82% | 63.69% | 90.62% |
| 20 | 0.3066 | 0.5135 | 76.36% | 63.98% | 90.70% |
| 21 | 0.3007 | 0.4953 | 76.78% | 64.86% | 90.82% |
| 22 | 0.2919 | 0.4976 | 77.50% | 65.21% | 90.94% |
| 23 | 0.2853 | 0.5089 | 77.84% | 64.80% | 90.93% |
| 24 | 0.2817 | 0.5113 | 77.83% | 65.27% | 90.92% |
| 25 | 0.2769 | 0.5127 | 78.53% | 65.16% | 90.94% |
| 26 | 0.2696 | 0.5048 | 78.76% | 65.28% | 90.94% |
| 27 | 0.2640 | 0.5045 | 79.35% | 65.76% | 91.02% |
| 28 | 0.2613 | 0.5114 | 79.75% | 64.91% | 90.98% |
| **29** | **0.2602** | **0.5078** | **79.62%** | **66.11%** | **91.11%** |

#### Phase 3 (Epochs 30-44)

| Epoch | Train Loss | Val Loss | Train mIoU | Val mIoU | Pixel Acc |
|-------|-----------|----------|-----------|----------|-----------|
| 30 | 0.2584 | 0.5215 | 79.31% | 65.00% | 90.93% |
| 31 | 0.2541 | 0.5352 | 79.70% | 65.96% | 91.12% |
| 32 | 0.2531 | 0.5334 | 79.72% | 64.80% | 90.68% |
| 33 | 0.2438 | 0.5294 | 80.10% | 65.17% | 90.91% |
| 34 | 0.2458 | 0.5311 | 79.96% | 65.66% | 90.95% |
| 35 | 0.2430 | 0.5345 | 80.34% | 65.63% | 91.06% |
| 36 | 0.2403 | 0.5320 | 80.51% | 65.71% | 90.97% |
| 37 | 0.2332 | 0.5293 | 81.10% | 66.07% | 91.00% |
| 38 | 0.2351 | 0.5374 | 81.11% | 65.58% | 91.09% |
| 39 | 0.2322 | 0.5340 | 81.19% | 65.91% | 91.06% |
| 40 | 0.2310 | 0.5433 | 81.04% | 65.21% | 91.01% |
| 41 | 0.2289 | 0.5431 | 81.18% | 65.50% | 91.10% |
| 42 | 0.2263 | 0.5399 | 81.59% | 65.75% | 91.04% |
| 43 | 0.2241 | 0.5386 | 81.69% | 65.61% | 91.07% |
| 44 | 0.2215 | 0.5514 | 81.94% | 65.78% | 91.12% |

#### 训练曲线绘图说明

使用上述数据可绘制以下曲线图：

1. **Loss 曲线**：
   - X 轴：Epoch (1-44)
   - Y 轴：Loss
   - 绘制 Train Loss 和 Val Loss 两条曲线

2. **mIoU 曲线**：
   - X 轴：Epoch (1-44)
   - Y 轴：mIoU (%)
   - 绘制 Train mIoU 和 Val mIoU 两条曲线
   - 标注最佳点 (Epoch 29, 66.11%)

3. **Pixel Accuracy 曲线**：
   - X 轴：Epoch (1-44)
   - Y 轴：Pixel Accuracy (%)
   - 展示像素级准确率的变化趋势
