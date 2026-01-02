#!/bin/bash
# 自动查找best checkpoint并推理
cd "$(dirname "$0")/.."

# 查找最新的best checkpoint
BEST_CKPT=$(ls -t checkpoints/*_best.pth 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "错误: 未找到best checkpoint文件"
    echo "请确保训练已完成并生成了 *_best.pth 文件"
    exit 1
fi

echo "找到最佳模型: $BEST_CKPT"
echo "开始推理..."

python inference.py \
    --config configs/config.yaml \
    --checkpoint "$BEST_CKPT" \
    --output predictions \
    --tta

echo "推理完成！结果保存在 predictions/ 目录"
echo "预测图片数量: $(ls predictions/*.png 2>/dev/null | wc -l)"
