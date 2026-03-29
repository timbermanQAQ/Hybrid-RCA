import pandas as pd
import json
import os

# 配置
CSV_FILE = "/root/autodl-tmp/phase_2_test.csv"
OUTPUT_DIR = "/root/autodl-tmp/fineTuningLab/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"读取测试集CSV: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

print(f"测试集数据量: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 转换为JSONL格式（每行一个JSON）
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for idx, row in df.iterrows():
        # 测试集只有system和user消息
        conversation = {
            "id": str(row['ID']),
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个5G无线网络优化专家。请根据提供的用户平面路测数据和工程参数数据，分析吞吐率下降的原因。答案必须用\\boxed{}格式包裹。"
                },
                {
                    "role": "user",
                    "content": str(row['question'])
                }
                # 注意：没有assistant部分，让模型生成
            ]
        }
        f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

print(f"\n保存成功: {OUTPUT_FILE}")
print(f"  数据量: {len(df)} 条")
print(f"  文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

# 显示示例
print("\n测试集示例（第1条）:")
with open(OUTPUT_FILE, 'r') as f:
    first_line = f.readline().strip()
    print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))
