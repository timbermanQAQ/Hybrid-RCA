from qwen_5g_optimizer import Qwen5GOptimizerModel

# 初始化模型
model = Qwen5GOptimizerModel(
    base_model_path="model/Qwen2.5-1.5B-Instruct",
    adapter_path="qwen2/15B_output/qwen2.5-1.5b/lora_weights"
)

# 处理文件
stats = model.process_file(
    input_file_path="/data/test.jsonl",
    output_file_path="output15.jsonl",
    # max_items=3,  # 可选：限制处理数量
    require_analysis=True  # 生成分析
)

print(f"处理完成，统计信息: {stats}")