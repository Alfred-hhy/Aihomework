# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **COCO Semantic Segmentation** project that converts COCO instance segmentation annotations to semantic segmentation masks and trains a SegFormer model using PyTorch. The project supports distributed training (DDP), mixed precision, and test-time augmentation.

## Essential Commands

### Setup
```bash
# Install dependencies
pip install -r coco_segmentation/requirements.txt

# Test the codebase (validates data loading, model, loss, metrics)
cd coco_segmentation && python test_code.py
```

### Training
```bash
# Single GPU training
cd coco_segmentation
python train.py --config configs/config.yaml
# Or use the script:
./scripts/train_single.sh

# Multi-GPU DDP training (default: 2 GPUs)
torchrun --nproc_per_node=2 --master_port=29500 train.py --config configs/config.yaml
# Or use the script with custom GPU count:
./scripts/train_ddp.sh 4  # for 4 GPUs
```

### Inference
```bash
# Run inference with TTA (Test Time Augmentation)
cd coco_segmentation
python inference.py --config configs/config.yaml --checkpoint <path_to_checkpoint> --output predictions --tta

# Or use the script:
./scripts/run_inference.sh <checkpoint_path> [output_dir]

# Run inference without TTA (faster)
python inference.py --config configs/config.yaml --checkpoint <path_to_checkpoint> --no-tta
```

## Architecture

### Data Pipeline (`coco_segmentation/data/`)

**Key Insight**: COCO has 80 non-contiguous category IDs (1-90 with gaps). The code maps these to contiguous labels 1-80, with 0 reserved for background, totaling 81 classes.

- `coco_dataset.py`:
  - `COCOSegmentationDataset`: Converts COCO instance annotations to semantic segmentation masks by iterating through annotations and drawing polygons/RLE masks
  - `TestDataset`: Handles test images without annotations
  - Important mappings: `COCO_CAT_TO_LABEL` (non-contiguous → contiguous), `COCO_CATEGORY_NAMES`

- `transforms.py`: Uses albumentations for data augmentation
  - Training: RandomResizedCrop, HorizontalFlip, ColorJitter, GaussianBlur
  - Validation/Test: Simple resize + normalize
  - TTA: Multiple scales (0.75, 1.0, 1.25) + optional horizontal flip

### Model (`coco_segmentation/models/`)

- `segformer.py`:
  - `SegFormerWrapper`: Wraps HuggingFace's SegformerForSemanticSegmentation
  - Loads pretrained weights from `nvidia/segformer-b5-finetuned-ade-640-640`
  - Automatically upsamples output to input size using bilinear interpolation
  - `get_param_groups()`: Provides differential learning rates (backbone: 0.1x, head: 1.0x)

- `losses.py`:
  - `DiceLoss`: Soft Dice coefficient loss
  - `FocalLoss`: For class imbalance
  - `CombinedLoss`: Weighted combination of CrossEntropy + Dice (default)
  - Configurable via `config.yaml` (`loss_type`: ce/dice/ce_dice/focal)

### Training (`coco_segmentation/train.py`)

**Distributed Training Architecture**:
- Uses PyTorch DDP (DistributedDataParallel) with NCCL backend
- Automatic detection: checks `LOCAL_RANK` environment variable (set by `torchrun`)
- SyncBatchNorm for synchronized batch normalization across GPUs
- DistributedSampler ensures each GPU sees different data

**Key Components**:
- Mixed precision training with `torch.cuda.amp.GradScaler`
- Gradient accumulation (default: 2 steps)
- Polynomial learning rate decay with warmup
- Checkpointing: saves model, optimizer, scheduler, scaler states
- Logging: TensorBoard integration (only rank 0 writes logs)

**Training Flow**:
1. Setup distributed environment (if multi-GPU)
2. Create datasets with distributed samplers
3. Wrap model in DDP
4. Train with gradient accumulation and mixed precision
5. Validate and save checkpoints (best model based on mIoU)

### Inference (`coco_segmentation/inference.py`)

**Two Modes**:
1. **TTA Mode** (`--tta`):
   - Processes images one-by-one
   - Applies multiple augmentations (scales + optional flip)
   - Averages predictions across augmentations
   - Higher quality but slower

2. **Batch Mode** (`--no-tta`):
   - Processes images in batches
   - Faster inference
   - Single forward pass per image

**Output**: Single-channel grayscale PNG masks (0-80 class labels)

### Utilities (`coco_segmentation/utils/`)

- `metrics.py`:
  - `SegmentationMetrics`: Accumulates confusion matrix and computes mIoU, pixel accuracy, mean accuracy
  - Updates incrementally during training/validation

- `visualization.py`:
  - `mask_to_color()`: Converts class labels to RGB colormap (PASCAL VOC style)
  - `visualize_prediction()`: Side-by-side comparison of image/GT/prediction
  - `overlay_mask()`: Semi-transparent mask overlay

## Configuration (`coco_segmentation/configs/config.yaml`)

Central configuration file for all hyperparameters:
- **Data paths**: Relative to config file's parent directory (default: `../`)
- **Model**: SegFormer variant, num_classes (81), pretrained weights
- **Training**: batch_size, learning_rate, optimizer, scheduler, loss_type
- **Augmentation**: input_size, scale_range, flip, color_jitter
- **Inference**: TTA settings (use_tta, tta_scales, tta_flip)
- **Checkpointing**: save_dir, monitoring metric (val_mIoU)

## Data Structure Expected

```
/home/hehuiyang/file/homework/
├── Aihomework/
│   └── coco_segmentation/
│       ├── configs/
│       ├── data/
│       ├── models/
│       ├── scripts/
│       ├── utils/
│       ├── train.py
│       ├── inference.py
│       └── test_code.py
├── train2014/              # COCO training images
├── val2014/                # COCO validation images
├── test/                   # Test images (no annotations)
└── annotations/
    ├── instances_train2014.json
    └── instances_val2014.json
```

## Key Implementation Details

1. **Category Mapping**: Always use `COCO_CAT_TO_LABEL` when converting COCO category IDs to training labels. The model outputs 81 classes (0=background, 1-80=COCO categories).

2. **Mask Generation** (`coco_dataset.py:86-125`): Handles both polygon and RLE annotation formats. Annotations are sorted by area (largest first) so smaller objects overwrite larger ones in overlapping regions.

3. **Learning Rate Scheduling** (`train.py:59-70`): Polynomial decay with warmup. The scheduler steps **per iteration**, not per epoch. Total iterations = epochs × batches_per_epoch.

4. **Gradient Accumulation** (`train.py:121-125`): Loss is scaled by `1/accumulation_steps`, then gradients are accumulated. Optimizer steps only every N iterations.

5. **Checkpoint Loading** (`inference.py:30-49`): Handles both raw state_dicts and wrapped checkpoint dicts. Automatically strips `module.` prefix from DDP-saved models.

6. **TTA Implementation** (`inference.py:89-146`): Each augmentation is applied, prediction is made, then inverse transform is applied (e.g., flip back). All predictions are averaged in probability space before argmax.

## Common Workflows

**Adding a new loss function**:
1. Implement loss class in `models/losses.py`
2. Add case in `create_loss()` function
3. Update `config.yaml` with new `loss_type` option

**Changing the model architecture**:
1. Implement model in `models/` directory
2. Add case in `models/segformer.py:create_model()`
3. Update `config.yaml` model section

**Modifying data augmentation**:
- Edit `data/transforms.py`: `get_train_transforms()` for training augmentations
- Update `config.yaml` augmentation section to add new hyperparameters

**Debugging training**:
- Use `test_code.py` to validate data loading and model forward pass
- Check TensorBoard logs: `tensorboard --logdir coco_segmentation/logs`
- Reduce batch_size if OOM errors occur
- Disable mixed precision by setting `use_amp: false` in config
