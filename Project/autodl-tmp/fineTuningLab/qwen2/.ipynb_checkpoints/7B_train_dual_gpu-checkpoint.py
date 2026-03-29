import json
import torch
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
from data_preprocess import InputOutputDataset

print("=== 双4090D训练开始（Causal LM格式）===")
print(f"GPU数量: {torch.cuda.device_count()}")

# 清理GPU
torch.cuda.empty_cache()

# 设置随机种子
set_seed(42)

# 配置
model_path = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
train_file = "../data/train.jsonl"
dev_file = "../data/dev.jsonl"
output_dir = "output/dual_4090d_final"

print("1. 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 设置tokenizer并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("2. 加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    use_cache=False,
)

print("3. 配置LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("4. 创建args配置...")
class Args:
    def __init__(self):
        self.max_source_length = 1024  # 输入长度
        self.ignore_pad_token_for_loss = True

args = Args()

print("5. 加载数据...")
with open(train_file, 'r') as f:
    train_data = [json.loads(line) for line in f]

with open(dev_file, 'r') as f:
    dev_data = [json.loads(line) for line in f]

print(f"训练数据: {len(train_data)} 条")
print(f"验证数据: {len(dev_data)} 条")

print("6. 创建数据集...")
train_dataset = InputOutputDataset(train_data, tokenizer, args)
dev_dataset = InputOutputDataset(dev_data, tokenizer, args)

print(f"训练集: {len(train_dataset)} 条")
print(f"验证集: {len(dev_dataset)} 条")

# 测试一个样本
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"\n样本检查:")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"labels shape: {sample['labels'].shape}")
    
    valid_labels = (sample['labels'] != -100).sum().item()
    print(f"有效标签数量: {valid_labels}/{sample['labels'].shape[0]}")
    
    if valid_labels > 0:
        print("✅ 数据格式正确！")
        
        # 解码看看
        input_text = tokenizer.decode(sample['input_ids'][:100])
        labels = sample['labels']
        valid_indices = (labels != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) > 0:
            valid_labels_text = tokenizer.decode(labels[valid_indices])
            print(f"\n输入前100个token: {input_text}")
            print(f"目标文本: {valid_labels_text}")
            
            # 解码整个输入
            full_text = tokenizer.decode(sample['input_ids'])
            print(f"\n完整输入长度: {len(full_text)}")
            print(f"完整输入预览: {full_text[:500]}...")
    else:
        print("❌ 警告：有效标签为0！")

print("7. 配置训练参数...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    learning_rate=2e-4,
    bf16=True,
    report_to="tensorboard",
    logging_dir=f"{output_dir}/logs",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",
    save_strategy="steps",
    logging_strategy="steps",
    dataloader_pin_memory=False,
    load_best_model_at_end=True,
    warmup_steps=100,
)

print("8. 创建训练器...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
    ),
)

print("9. 开始训练...")
try:
    # 先测试一个批次
    print("测试前向传播...")
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_batch = next(iter(test_loader))
    
    print(f"测试批次形状:")
    print(f"  input_ids: {test_batch['input_ids'].shape}")
    print(f"  attention_mask: {test_batch['attention_mask'].shape}")
    print(f"  labels: {test_batch['labels'].shape}")
    
    # 检查形状是否匹配
    if test_batch['input_ids'].shape == test_batch['labels'].shape:
        print("✅ 输入和标签形状匹配！")
        
        # 移动到GPU
        test_batch = {k: v.to(model.device) for k, v in test_batch.items()}
        
        with torch.no_grad():
            outputs = model(**test_batch)
            loss_value = outputs.loss.item() if outputs.loss is not None else float('nan')
            print(f"测试成功！Loss: {loss_value}")
        
        if torch.isnan(outputs.loss):
            print("警告：Loss是nan，但数据格式已正确，可能是初始化问题，继续训练...")
        
        print("\n开始正式训练...")
        trainer.train()
        
        print("\n10. 保存模型...")
        trainer.save_model()
        print(f"✅ 模型保存到: {output_dir}")
    else:
        print(f"❌ 输入和标签形状不匹配！")
        print(f"  输入形状: {test_batch['input_ids'].shape}")
        print(f"  标签形状: {test_batch['labels'].shape}")
    
except Exception as e:
    print(f"训练失败: {str(e)}")
    import traceback
    traceback.print_exc()