#!/bin/bash
# 多卡 DDP 训练
# Usage: ./scripts/train_ddp.sh [NUM_GPUS]

cd "$(dirname "$0")/.."

NUM_GPUS=${1:-2}

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    train.py --config configs/config.yaml
