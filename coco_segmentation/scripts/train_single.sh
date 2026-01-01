#!/bin/bash
# 单卡训练
# Usage: ./scripts/train_single.sh

cd "$(dirname "$0")/.."

python train.py --config configs/config.yaml
