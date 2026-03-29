#!/bin/bash

# 5G网络分析Web界面启动脚本

MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"
LORA_CKPT="./output/5g_network_analysis-*"  # 微调后会自动生成

# 查找最新的LoRA检查点
latest_ckpt=$(ls -td ./output/5g_network_analysis-* 2>/dev/null | head -1)

if [ -z "$latest_ckpt" ]; then
    echo "警告：未找到LoRA检查点，使用基础模型"
    LORA_CKPT=""
else
    echo "使用LoRA检查点: $latest_ckpt"
    LORA_CKPT="$latest_ckpt"
fi

cd "$(dirname "$0")"

python webui_qwen2.py \
    --model "$MODEL_PATH" \
    --ckpt "$LORA_CKPT"
