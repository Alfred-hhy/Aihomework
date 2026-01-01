#!/bin/bash
# 推理脚本
# Usage: ./scripts/run_inference.sh <checkpoint_path> [output_dir]

cd "$(dirname "$0")/.."

CHECKPOINT=$1
OUTPUT=${2:-predictions}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./scripts/run_inference.sh <checkpoint_path> [output_dir]"
    exit 1
fi

python inference.py \
    --config configs/config.yaml \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --tta
