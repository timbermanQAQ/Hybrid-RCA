import json
import torch
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer,
    set_seed
)
from peft import LoraConfig, TaskType, get_peft_model
import os

print("=== 调试版本：找出标签问题 ===")

# 清理GPU
torch.cuda.empty_cache()
gc.collect()

# 设置tokenizer并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置随机种子
set_seed(42)

# 配置
model_path = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
train_file = "../data/train.jsonl"

print("1. 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2. 加载一条数据并调试...")
with open(train_file, 'r') as f:
    first_line = f.readline()
    item = json.loads(first_line)

print("原始数据格式:")
print(json.dumps(item, ensure_ascii=False, indent=2))

print("\n3. 构建完整对话文本...")
# 按照你的build_full_text方法
full_text = ""
for msg in item["messages"]:
    full_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

print(f"完整对话文本长度: {len(full_text)}")
print(f"完整对话文本前500字符:\n{full_text[:500]}")

print("\n4. 编码完整对话文本...")
encoding = tokenizer(
    full_text,
    max_length=512,  # 先用较短的长度测试
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

input_ids = encoding["input_ids"][0]
print(f"input_ids长度: {len(input_ids)}")
print(f"input_ids前20个: {input_ids[:20].tolist()}")

# 解码前20个token看看
print(f"前20个token解码: {tokenizer.decode(input_ids[:20])}")

print("\n5. 查找assistant标记...")
# 先编码"<|im_start|>assistant\n"
assistant_marker = "<|im_start|>assistant\n"
assistant_tokens = tokenizer.encode(assistant_marker, add_special_tokens=False)
print(f"assistant标记token: {assistant_tokens}")

# 在input_ids中查找这个序列
input_ids_list = input_ids.tolist()
for i in range(len(input_ids_list) - len(assistant_tokens) + 1):
    if input_ids_list[i:i+len(assistant_tokens)] == assistant_tokens:
        print(f"找到assistant标记在位置: {i}")
        break
else:
    print("未找到assistant标记！")

print("\n6. 手动查找字符串位置...")
assistant_start = full_text.find("<|im_start|>assistant\n")
print(f"字符串查找assistant开始位置: {assistant_start}")

if assistant_start > 0:
    before_assistant = full_text[:assistant_start]
    print(f"assistant之前文本长度: {len(before_assistant)}")
    print(f"assistant之前文本: {before_assistant[-100:] if len(before_assistant) > 100 else before_assistant}")
    
    # 编码assistant之前的部分
    before_tokens = tokenizer.encode(before_assistant, add_special_tokens=False)
    print(f"assistant之前token数量: {len(before_tokens)}")
    print(f"assistant之前token: {before_tokens[:20]}")
    
    # 编码整个文本（无特殊标记）
    full_tokens_no_special = tokenizer.encode(full_text, add_special_tokens=False)
    print(f"完整文本token数量（无特殊）: {len(full_tokens_no_special)}")
    
    # 检查前before_tokens个token是否匹配
    if before_tokens == full_tokens_no_special[:len(before_tokens)]:
        print("✅ assistant之前部分匹配正确")
    else:
        print("❌ assistant之前部分不匹配！")
        print(f"前{before_tokens}个token应该为: {before_tokens[:20]}")
        print(f"实际前{before_tokens}个token为: {full_tokens_no_special[:20]}")

print("\n7. 完整解码input_ids看看...")
full_decoded = tokenizer.decode(input_ids)
# 找到第一个<|im_start|>assistant的位置
decoded_assistant_pos = full_decoded.find("<|im_start|>assistant")
print(f"解码文本中assistant位置: {decoded_assistant_pos}")
if decoded_assistant_pos > 0:
    before_assistant_decoded = full_decoded[:decoded_assistant_pos]
    print(f"解码文本中assistant之前部分长度: {len(before_assistant_decoded)}")