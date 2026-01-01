"""
测试数据加载和模型是否正常工作
"""

import os
import sys
import yaml

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("Testing data loading...")
    print("=" * 60)

    from data import (
        COCOSegmentationDataset,
        TestDataset,
        get_train_transforms,
        get_val_transforms,
        NUM_CLASSES,
        COCO_CATEGORY_NAMES,
    )

    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_root = config['data']['root']

    # 测试训练集
    print("\n1. Testing training dataset...")
    train_transforms = get_train_transforms(config)
    train_dataset = COCOSegmentationDataset(
        root=data_root,
        ann_file=config['data']['train_ann'],
        image_dir=config['data']['train_images'],
        transform=train_transforms,
    )

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Num classes: {NUM_CLASSES}")

    # 加载一个样本
    image, mask = train_dataset[0]
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Unique labels in mask: {np.unique(mask.numpy())[:10]}...")

    # 测试验证集
    print("\n2. Testing validation dataset...")
    val_transforms = get_val_transforms(config)
    val_dataset = COCOSegmentationDataset(
        root=data_root,
        ann_file=config['data']['val_ann'],
        image_dir=config['data']['val_images'],
        transform=val_transforms,
    )
    print(f"   Validation samples: {len(val_dataset)}")

    # 测试测试集
    print("\n3. Testing test dataset...")
    test_dataset = TestDataset(
        root=data_root,
        image_dir=config['data']['test_images'],
        transform=val_transforms,
    )
    print(f"   Test samples: {len(test_dataset)}")

    image, name, size = test_dataset[0]
    print(f"   Sample image: {name}, original size: {size}")

    print("\n✓ Data loading test passed!")
    return True


def test_model():
    """测试模型"""
    print("\n" + "=" * 60)
    print("Testing model...")
    print("=" * 60)

    from models import create_model
    from data import NUM_CLASSES

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 创建模型
    print("\n1. Creating model...")
    model = create_model(config)
    print(f"   Model created: {config['model']['name']}")

    # 测试前向传播
    print("\n2. Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    model = model.to(device)
    model.eval()

    # 创建假数据
    batch_size = 2
    input_size = config['augmentation']['val']['input_size']
    x = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected shape: [{batch_size}, {NUM_CLASSES}, {input_size[0]}, {input_size[1]}]")

    assert output.shape == (batch_size, NUM_CLASSES, input_size[0], input_size[1]), \
        "Output shape mismatch!"

    print("\n✓ Model test passed!")
    return True


def test_loss():
    """测试损失函数"""
    print("\n" + "=" * 60)
    print("Testing loss function...")
    print("=" * 60)

    from models import create_loss
    from data import NUM_CLASSES

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建损失函数
    criterion = create_loss(config)
    criterion = criterion.to(device)

    # 创建假数据
    batch_size = 2
    h, w = 512, 512
    logits = torch.randn(batch_size, NUM_CLASSES, h, w).to(device)
    targets = torch.randint(0, NUM_CLASSES, (batch_size, h, w)).to(device)

    # 计算损失
    loss, loss_dict = criterion(logits, targets)

    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Loss components: {loss_dict}")

    print("\n✓ Loss function test passed!")
    return True


def test_metrics():
    """测试评估指标"""
    print("\n" + "=" * 60)
    print("Testing metrics...")
    print("=" * 60)

    from utils import SegmentationMetrics
    from data import NUM_CLASSES

    metrics = SegmentationMetrics(NUM_CLASSES)

    # 创建假数据
    batch_size = 2
    h, w = 64, 64
    preds = torch.randint(0, NUM_CLASSES, (batch_size, h, w))
    targets = torch.randint(0, NUM_CLASSES, (batch_size, h, w))

    # 更新指标
    metrics.update(preds, targets)
    results = metrics.compute()

    print(f"   mIoU: {results['mIoU']*100:.2f}%")
    print(f"   Pixel Accuracy: {results['pixel_acc']*100:.2f}%")
    print(f"   Mean Accuracy: {results['mean_acc']*100:.2f}%")

    print("\n✓ Metrics test passed!")
    return True


def visualize_sample():
    """可视化一个样本"""
    print("\n" + "=" * 60)
    print("Visualizing a sample...")
    print("=" * 60)

    from data import COCOSegmentationDataset, get_val_transforms
    from utils import mask_to_color

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_root = config['data']['root']
    val_transforms = get_val_transforms(config)

    dataset = COCOSegmentationDataset(
        root=data_root,
        ann_file=config['data']['train_ann'],
        image_dir=config['data']['train_images'],
        transform=val_transforms,
    )

    # 获取一个样本
    image, mask = dataset[100]

    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image.numpy().transpose(1, 2, 0)
    image_np = (image_np * std + mean) * 255
    image_np = image_np.astype(np.uint8)

    mask_np = mask.numpy()
    mask_color = mask_to_color(mask_np)

    # 保存可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_np)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150)
    print(f"   Saved visualization to sample_visualization.png")

    print("\n✓ Visualization test passed!")
    return True


def main():
    print("\n" + "#" * 60)
    print("# COCO Segmentation - Test Suite")
    print("#" * 60)

    tests = [
        ("Data Loading", test_data_loading),
        ("Model", test_model),
        ("Loss Function", test_loss),
        ("Metrics", test_metrics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 可视化测试 (可选)
    try:
        visualize_sample()
        results.append(("Visualization", True))
    except Exception as e:
        print(f"\n✗ Visualization test failed: {e}")
        results.append(("Visualization", False))

    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"   {name}: {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! The code is ready for training.")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == '__main__':
    main()
