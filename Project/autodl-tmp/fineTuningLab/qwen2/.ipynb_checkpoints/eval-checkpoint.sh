#! /usr/bin/env bash

# 配置
MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"
DATA_PATH="/root/autodl-tmp/fineTuningLab/data/test.jsonl"
OUTPUT_DIR="./eval_results"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 查找最新的训练检查点
echo "查找最新的训练检查点..."
LATEST_CHECKPOINT=$(find ./output -type d -name "5g_network_analysis-*" 2>/dev/null | sort -r | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "警告: 未找到5g_network_analysis检查点，尝试查找其他检查点..."
    LATEST_CHECKPOINT=$(find ./output -type d -name "*" 2>/dev/null | sort -r | head -1)
fi

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "错误: 未找到任何检查点，请先运行训练"
    echo "使用基础模型进行评估..."
    CHECKPOINT_PATH=""
else
    # 查找checkpoint-*子目录
    CHECKPOINT_SUBDIR=$(find "$LATEST_CHECKPOINT" -type d -name "checkpoint-*" 2>/dev/null | sort -r | head -1)
    
    if [ -z "$CHECKPOINT_SUBDIR" ]; then
        echo "使用完整检查点目录: $LATEST_CHECKPOINT"
        CHECKPOINT_PATH="$LATEST_CHECKPOINT"
    else
        echo "使用最新检查点: $CHECKPOINT_SUBDIR"
        CHECKPOINT_PATH="$CHECKPOINT_SUBDIR"
    fi
fi

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/eval_results_${TIMESTAMP}.json"

echo "==================== 评估配置 ===================="
echo "基础模型: $MODEL_PATH"
echo "LoRA检查点: ${CHECKPOINT_PATH:-无}"
echo "测试数据: $DATA_PATH"
echo "输出文件: $OUTPUT_FILE"
echo "=================================================="

# 运行评估
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "评估基础模型..."
    CUDA_VISIBLE_DEVICES=0 python evaluate.py \
        --model "$MODEL_PATH" \
        --data "$DATA_PATH" \
        --output "$OUTPUT_FILE"
else
    echo "评估微调模型..."
    CUDA_VISIBLE_DEVICES=0 python evaluate.py \
        --model "$MODEL_PATH" \
        --ckpt "$CHECKPOINT_PATH" \
        --data "$DATA_PATH" \
        --output "$OUTPUT_FILE"
fi

# 显示结果
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "==================== 评估结果 ===================="
    python3 -c "
import json
with open('$OUTPUT_FILE', 'r', encoding='utf-8') as f:
    results = json.load(f)
    total = results.get('total', 0)
    correct = results.get('correct', 0)
    accuracy = results.get('accuracy', 0)
    print(f'总样本数: {total}')
    print(f'正确数: {correct}')
    print(f'准确率: {accuracy:.2%}')
    
    # 显示详细结果
    if 'results' in results and len(results['results']) > 0:
        print(f'\\n前3个样本结果:')
        for i, r in enumerate(results['results'][:3]):
            print(f'  样本 {i+1}:')
            print(f'    预测: {r.get(\"pred\", \"N/A\")}')
            print(f'    真实: {r.get(\"true\", \"N/A\")}')
            print(f'    正确: {r.get(\"correct\", False)}')
            print()
"
else
    echo "错误: 评估输出文件未生成"
fi