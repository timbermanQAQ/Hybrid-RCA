# 在 finetune.py 中添加多GPU支持
import json
import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq, 
    HfArgumentParser,
    TrainingArguments, 
    Trainer,
    set_seed
)
from peft import LoraConfig, TaskType, get_peft_model
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from data_preprocess import InputOutputDataset


def main():
    set_seed(42)
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    
    # 检查可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    
    # 根据GPU数量调整batch_size
    if num_gpus > 1:
        print(f"使用多GPU训练: {num_gpus}个GPU")
        training_args._n_gpu = num_gpus
        training_args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("加载模型中...")
    
    # 多GPU配置
    if num_gpus > 1:
        # 多GPU使用自动device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",  # 自动分配多GPU
        )
    else:
        # 单GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_args.lora_rank,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=[],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 启用梯度检查点（节省显存）
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
    )
    
    # 加载训练数据
    if training_args.do_train:
        print(f"加载训练数据: {data_args.train_file}")
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
        train_dataset = InputOutputDataset(train_data, tokenizer, data_args)
        print(f"训练集大小: {len(train_dataset)}")
    
    # 加载验证数据
    if training_args.do_eval:
        print(f"加载验证数据: {data_args.validation_file}")
        with open(data_args.validation_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]
        eval_dataset = InputOutputDataset(eval_data, tokenizer, data_args)
        print(f"验证集大小: {len(eval_dataset)}")
    
    # 多GPU训练的特殊处理
    if num_gpus > 1:
        # 设置分布式训练
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
        training_args.dataloader_num_workers = 2 * num_gpus  # 增加数据加载workers
        
        # 根据GPU数量调整batch_size
        original_batch_size = training_args.per_device_train_batch_size
        training_args.per_device_train_batch_size = original_batch_size
        training_args.gradient_accumulation_steps = max(training_args.gradient_accumulation_steps // num_gpus, 1)
        
        print(f"多GPU配置:")
        print(f"  每个GPU batch_size: {training_args.per_device_train_batch_size}")
        print(f"  梯度累积步数: {training_args.gradient_accumulation_steps}")
        print(f"  有效总batch_size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_gpus}")
    
    # 训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
    )
    
    # 训练
    if training_args.do_train:
        print("开始训练...")
        trainer.train()
        
        # 只在主进程保存模型
        if training_args.local_rank in [-1, 0]:
            trainer.save_model()
            print(f"模型保存到: {training_args.output_dir}")
    
    # 评估
    if training_args.do_eval:
        print("最终评估...")
        metrics = trainer.evaluate()
        print(f"评估结果: {metrics}")


if __name__ == "__main__":
    # 设置多GPU环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    
    # 如果使用accelerate，初始化
    if torch.cuda.device_count() > 1:
        from accelerate import Accelerator
        accelerator = Accelerator()
        print(f"使用Accelerate，设备: {accelerator.device}")
    
    main()