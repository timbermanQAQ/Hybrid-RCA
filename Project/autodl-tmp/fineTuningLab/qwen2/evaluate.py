import json
import torch
import argparse
import re
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path, checkpoint_path=None):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    if checkpoint_path:
        model = PeftModel.from_pretrained(model, checkpoint_path)
    
    model.eval()
    return tokenizer, model


def extract_answer(text):
    """从模型输出中提取答案（如\boxed{C1}）"""
    # 匹配\boxed{}格式
    match = re.search(r'\\boxed{([^}]+)}', text)
    if match:
        return match.group(1)
    
    # 匹配其他常见格式
    patterns = [
        r'答案[：:]\s*([A-Z]\d+)',
        r'选项[：:]\s*([A-Z]\d+)',
        r'最终答案[：:]\s*([A-Z]\d+)',
        r'[Cc]\d+'  # 直接匹配C1, C2等
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None


class QwenEvaluator:
    def __init__(self, tokenizer, model, data_path):
        self.tokenizer = tokenizer
        self.model = model
        self.data_path = data_path
    
    def evaluate_accuracy(self):
        """评估答案准确率"""
        # 加载测试数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            if self.data_path.endswith('.jsonl'):
                test_data = [json.loads(line) for line in f]
            else:
                test_data = json.load(f)
        
        total = len(test_data)
        correct = 0
        results = []
        
        print(f"开始评估，共 {total} 条测试数据")
        
        for item in tqdm(test_data, desc="评估进度"):
            # 构建提示词
            prompt = self.build_prompt(item)
            
            # 生成回答
            response = self.generate_response(prompt)
            
            # 提取预测答案
            pred_answer = extract_answer(response)
            
            # 获取真实答案
            true_answer = self.extract_true_answer(item)
            
            # 计算是否正确
            is_correct = (pred_answer == true_answer)
            if is_correct:
                correct += 1
            
            # 记录结果
            results.append({
                "pred": pred_answer,
                "true": true_answer,
                "correct": is_correct,
                "response": response[:200] + "..." if len(response) > 200 else response
            })
        
        # 计算指标
        accuracy = correct / total if total > 0 else 0
        
        # 输出结果
        print(f"\n{'='*50}")
        print(f"评估结果:")
        print(f"  总样本数: {total}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.2%}")
        print(f"{'='*50}")
        
        # 显示错误示例
        wrong_examples = [r for r in results if not r["correct"]]
        if wrong_examples:
            print(f"\n错误示例 (前3个):")
            for i, ex in enumerate(wrong_examples[:3]):
                print(f"  示例 {i+1}:")
                print(f"    预测: {ex['pred']}")
                print(f"    真实: {ex['true']}")
                print(f"    生成内容: {ex['response']}")
                print()
        
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "results": results
        }
    
    def build_prompt(self, item):
        """根据数据格式构建提示词"""
        # 对话格式
        if "messages" in item:
            prompt = ""
            for msg in item["messages"]:
                if msg["role"] != "assistant":  # 测试时不包含assistant消息
                    prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt
        
        # 简单格式
        elif "question" in item:
            return f"""<|im_start|>system
你是5G无线网络优化专家。请分析吞吐率下降的原因，答案用\\boxed{{}}格式包裹。<|im_end|>
<|im_start|>user
{item['question']}<|im_end|>
<|im_start|>assistant\n"""
        
        else:
            return str(item)
    
    def generate_response(self, prompt):
        """生成模型响应"""
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        return response
    
    def extract_true_answer(self, item):
        """从数据项中提取真实答案"""
        # 从对话格式中提取
        if "messages" in item:
            for msg in reversed(item["messages"]):
                if msg["role"] == "assistant":
                    return extract_answer(msg["content"])
        
        # 从简单格式中提取
        elif "answer" in item:
            return item["answer"]
        
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--ckpt", type=str, help="LoRA检查点路径（可选）")
    parser.add_argument("--data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--output", type=str, default="eval_results.json", help="输出结果文件")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    tokenizer, model = load_model(args.model, args.ckpt)
    
    # 创建评估器
    evaluator = QwenEvaluator(tokenizer, model, args.data)
    
    # 进行评估
    results = evaluator.evaluate_accuracy()
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()