import pandas as pd
import json
import random
import os

# 配置
CSV_FILE = "/root/autodl-tmp/train.csv"
OUTPUT_DIR = "/root/autodl-tmp/fineTuningLab/data"
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train.jsonl")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val.jsonl")
VAL_RATIO = 0.1

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"读取训练集CSV: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

print(f"数据量: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 转换为对话格式，每行一个JSON字符串
conversations = []
for idx, row in df.iterrows():
    conversation = {
        "messages": [
            {
                "role": "system",
                "content": "你是一个5G无线网络优化专家。请根据提供的用户平面路测数据和工程参数数据，分析吞吐率下降的原因。答案必须用\\boxed{}格式包裹。"
            },
            {
                "role": "user",
                "content": str(row['question'])
            },
            {
                "role": "assistant",
                "content": f"\\boxed{{{row['answer']}}}"
            }
        ]
    }
    conversations.append(conversation)

print(f"转换后: {len(conversations)} 条")

# 随机拆分
random.shuffle(conversations)
split_idx = int(len(conversations) * (1 - VAL_RATIO))

train_data = conversations[:split_idx]
val_data = conversations[split_idx:]

print(f"\n拆分结果:")
print(f"  训练集: {len(train_data)} 条")
print(f"  验证集: {len(val_data)} 条")

# 保存为JSONL格式（每行一个JSON）
with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
    for conv in train_data:
        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
    for conv in val_data:
        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

print(f"\n保存成功:")
print(f"  训练集: {OUTPUT_TRAIN} (JSONL格式)")
print(f"  验证集: {OUTPUT_VAL} (JSONL格式)")

# 验证
print("\nJSONL格式验证:")
with open(OUTPUT_TRAIN, 'r') as f:
    lines = f.readlines()
    print(f"  第一行类型: {'JSON' if lines[0].strip().startswith('{') else '非JSON'}")
    print(f"  总行数: {len(lines)}")
    
print("\n训练集示例（第1条）:")
print(json.dumps(train_data[0], ensure_ascii=False, indent=2))
