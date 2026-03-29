import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer,
    set_seed,
    BitsAndBytesConfig  # 新增：用于4-bit量化
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # 新增k-bit训练支持
import os
from data_preprocess import InputOutputDataset

print("=== 单GPU Qwen2.5-1.5B训练开始 ===")
print(f"可用GPU: {torch.cuda.get_device_name(0)}")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 清理GPU
torch.cuda.empty_cache()

# 设置随机种子
set_seed(42)

# ==================== 配置修改 ====================
model_path = "/root/autodl-tmp/Qwen2.5-1.5B-Instruct"
train_file = "../data/train.jsonl"
dev_file = "../data/dev.jsonl"
output_dir = "output/qwen2.5-1.5b"  # 修改输出目录

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

print("2. 加载模型（使用4-bit量化节省显存）...")
# 配置4-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用4-bit量化
    bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bfloat16
    bnb_4bit_use_double_quant=True,  # 双重量化进一步节省显存
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,  # 应用量化配置
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",  # 自动分配到GPU
    use_cache=False,
)

# 准备模型进行k-bit训练
model = prepare_model_for_kbit_training(model)

print("3. 配置LoRA（适配1.5B模型结构）...")
# Qwen2.5-1.5B的注意力模块名称可能与7B不同
# 以下是1.5B模型常见的LoRA目标模块
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA秩，1.5B模型可以适当降低
    lora_alpha=16,  # 降低alpha值
    lora_dropout=0.05,  # 降低dropout率
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力投影层
        "gate_proj", "up_proj", "down_proj",     # MLP层
        # "qkv_proj", "out_proj", "fc_in", "fc_out"  # 备用模块名，如果上述不匹配可尝试
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("4. 创建args配置...")
class Args:
    def __init__(self):
        self.max_source_length = 512  # 1.5B模型可适当降低上下文长度
        self.ignore_pad_token_for_loss = True

args = Args()

print("5. 加载数据...")
# [数据加载部分保持不变，根据你的实际情况调整]
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

# 测试一个样本（保持原样，用于验证数据格式）
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
        input_text = tokenizer.decode(sample['input_ids'][:50])
        labels = sample['labels']
        valid_indices = (labels != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) > 0:
            valid_labels_text = tokenizer.decode(labels[valid_indices[:50]])
            print(f"\n输入前50个token: {input_text}")
            print(f"目标文本: {valid_labels_text}")

print("7. 配置训练参数（单GPU优化）...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,  # 单GPU可增加batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # 降低梯度累积步数
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    learning_rate=3e-4,  # 1.5B模型可使用稍高学习率
    bf16=True,
    report_to="tensorboard",
    logging_dir=f"{output_dir}/logs",
    remove_unused_columns=False,
    dataloader_num_workers=2,  # 单GPU可设置少量workers
    gradient_checkpointing=True,  # 梯度检查点节省显存
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",  # 使用8-bit优化器
    save_strategy="steps",
    logging_strategy="steps",
    dataloader_pin_memory=True,  # 单GPU可启用pin memory
    load_best_model_at_end=True,
    warmup_steps=50,  # 减少warmup步数
    fp16=False,  # 确保使用bf16而非fp16
    no_cuda=False,
    ddp_find_unused_parameters=False,  # 单GPU训练，禁用ddp相关参数
    dataloader_drop_last=True,  # 避免最后批次尺寸不一致
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
    
    # 显存使用监控
    print(f"训练前显存: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
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
        
        print(f"前向传播后显存: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        
        if torch.isnan(outputs.loss):
            print("警告：Loss是nan，可能是初始化问题，尝试调整学习率...")
        
        print("\n开始正式训练...")
        trainer.train()
        
        print("\n10. 保存模型...")
        # 保存完整的适配器权重
        trainer.save_model()
        
        # 也可单独保存LoRA权重
        model.save_pretrained(f"{output_dir}/lora_weights")
        print(f"✅ 模型保存到: {output_dir}")
        print(f"✅ LoRA权重保存到: {output_dir}/lora_weights")
        
    else:
        print(f"❌ 输入和标签形状不匹配！")
        print(f"  输入形状: {test_batch['input_ids'].shape}")
        print(f"  标签形状: {test_batch['labels'].shape}")
    
except Exception as e:
    print(f"训练失败: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # 显存清理
    print(f"\n训练结束，最终显存使用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    torch.cuda.empty_cache()