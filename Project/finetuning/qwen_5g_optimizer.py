import json
import torch
import re
import os
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class Qwen5GOptimizerModel:
    """Qwen2.5-1.5B微调模型封装模块"""
    
    def __init__(self, 
                 base_model_path: str,
                 adapter_path: str,
                 device: str = "auto",
                 torch_dtype: str = "bfloat16"):
        """
        初始化5G网络优化微调模型
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: LoRA适配器路径
            device: 设备设置，"auto"或"cuda"
            torch_dtype: 数据类型，"bfloat16"或"float16"
        """
        print("初始化5G网络优化模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 设置数据类型
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # 加载基础模型
        print(f"加载基础模型: {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        
        # 加载LoRA适配器
        print(f"加载LoRA适配器: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        # 默认生成参数
        self.default_generation_params = {
            "max_new_tokens": 350,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "num_beams": 1,
        }
        
        print("模型初始化完成")
    
    def extract_available_options(self, user_content: str) -> List[str]:
        """从用户问题中提取可用的选项"""
        options = []
        
        # 模式1: M1: ... M5:
        m_pattern = r'(M[1-5])\s*[:：]'
        m_matches = re.findall(m_pattern, user_content)
        if m_matches:
            options.extend(sorted(set(m_matches)))
        
        # 模式2: 1: ... 8:
        num_pattern = r'^(\d+)\s*[:：]'
        num_matches = re.findall(num_pattern, user_content, re.MULTILINE)
        if num_matches:
            options.extend(sorted(set(num_matches), key=int))
        
        # 模式3: C1-C8
        c_pattern = r'\b(C\d+)\b'
        c_matches = re.findall(c_pattern, user_content)
        if c_matches:
            options.extend(sorted(set(c_matches)))
        
        # 模式4: 纯数字列表
        if not options:
            num_list_pattern = r'选项\s*[：:]\s*(\d+(?:\s*,\s*\d+)*)'
            list_match = re.search(num_list_pattern, user_content)
            if list_match:
                numbers = re.findall(r'\d+', list_match.group(1))
                options.extend(numbers)
        
        return options
    
    def format_input_text(self, 
                         system_content: str, 
                         user_content: str,
                         require_analysis: bool = True,
                         max_length: int = 512) -> str:
        """格式化输入文本"""
        
        # 提取可用选项
        available_options = self.extract_available_options(user_content)
        
        # 构建增强的system prompt
        if require_analysis and available_options:
            option_str = "、".join(available_options[:5]) + "等" if len(available_options) > 5 else "、".join(available_options)
            enhanced_system = f"""{system_content}

重要要求：
1. 请先进行详细的技术分析
2. 必须从{option_str}中选择最可能的原因
3. 最终答案必须使用\\boxed{{}}格式包裹
4. 格式示例：\\boxed{{M3}} 或 \\boxed{{4}}"""
        else:
            enhanced_system = system_content
        
        # 构建输入文本
        if require_analysis:
            input_text = f"<|im_start|>system\n{enhanced_system}<|im_end|>\n"
            input_text += f"<|im_start|>user\n{user_content}<|im_end|>\n"
            input_text += "<|im_start|>assistant\n分析："
        else:
            input_text = f"<|im_start|>system\n{enhanced_system}<|im_end|>\n"
            input_text += f"<|im_start|>user\n{user_content}<|im_end|>\n"
            input_text += "<|im_start|>assistant\n"
        
        # 截断处理
        tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        
        if len(tokens) > max_length:
            system_part = f"<|im_start|>system\n{enhanced_system}<|im_end|>\n"
            assistant_start = "<|im_start|>assistant\n分析：" if require_analysis else "<|im_start|>assistant\n"
            
            system_tokens = self.tokenizer.encode(system_part + assistant_start, add_special_tokens=False)
            available_tokens = max_length - len(system_tokens) - 10
            
            user_tokens = self.tokenizer.encode(user_content, add_special_tokens=False)
            if len(user_tokens) > available_tokens:
                truncated_user_tokens = user_tokens[:available_tokens]
                truncated_user = self.tokenizer.decode(truncated_user_tokens)
                
                if truncated_user.endswith("...") or truncated_user.endswith("…"):
                    truncated_user = truncated_user[:-1]
                
                user_content = truncated_user + " [数据截断]"
            
            input_text = f"<|im_start|>system\n{enhanced_system}<|im_end|>\n"
            input_text += f"<|im_start|>user\n{user_content}<|im_end|>\n"
            if require_analysis:
                input_text += "<|im_start|>assistant\n分析："
            else:
                input_text += "<|im_start|>assistant\n"
        
        return input_text
    
    def extract_answer(self, response: str) -> str:
        """从响应中提取答案"""
        # 方法1：提取\boxed{}中的内容
        box_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(box_pattern, response)
        if match:
            return match.group(1)
        
        # 方法2：提取数字（如4, 8等）
        number_pattern = r'\b(\d+)\b'
        numbers = re.findall(number_pattern, response)
        if numbers:
            return numbers[-1]
        
        # 方法3：提取M1, M2等格式
        m_pattern = r'\b(M\d+)\b'
        m_match = re.search(m_pattern, response, re.IGNORECASE)
        if m_match:
            return m_match.group(1).upper()
        
        # 方法4：提取选项模式
        option_patterns = [
            r'选项[：:\s]*([A-Za-z0-9]+)',
            r'答案[：:\s]*([A-Za-z0-9]+)',
            r'选择[：:\s]*([A-Za-z0-9]+)',
        ]
        
        for pattern in option_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        return ""
    
    def extract_analysis(self, response: str) -> str:
        """从响应中提取分析部分"""
        # 移除答案部分
        response_clean = response
        
        # 移除\boxed{}部分
        box_pattern = r'\\boxed\{[^}]*\}'
        response_clean = re.sub(box_pattern, '', response_clean)
        
        # 移除其他答案标记
        answer_patterns = [
            r'答案[：:\s]*[A-Za-z0-9]+',
            r'选项[：:\s]*[A-Za-z0-9]+',
            r'选择[：:\s]*[A-Za-z0-9]+',
            r'\b[M]?\d+\b(?=\s*$)',
        ]
        
        for pattern in answer_patterns:
            response_clean = re.sub(pattern, '', response_clean)
        
        # 清理空白字符
        response_clean = response_clean.strip()
        
        # 如果以"分析："开头，去掉
        if response_clean.startswith("分析："):
            response_clean = response_clean[3:].strip()
        
        return response_clean
    
    def generate_response(self, 
                         input_text: str,
                         generation_params: Optional[Dict] = None) -> str:
        """生成响应"""
        if generation_params is None:
            generation_params = self.default_generation_params
        
        # 编码
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_params,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取assistant生成的内容
        assistant_marker = "<|im_start|>assistant"
        if assistant_marker in full_response:
            parts = full_response.split(assistant_marker)
            if len(parts) > 1:
                generated = parts[-1]
                generated = generated.replace("<|im_end|>", "").strip()
                # 如果以"分析："开头，去掉前面的标记
                if generated.startswith("分析："):
                    generated = generated[3:].strip()
                elif generated.startswith("\n"):
                    generated = generated[1:].strip()
            else:
                generated = full_response[len(input_text):].strip()
        else:
            generated = full_response[len(input_text):].strip()
        
        return generated
    
    def ensure_boxed_format(self, answer: str) -> str:
        """确保答案以\boxed{}格式输出"""
        if not answer:
            return "\\boxed{}"
        
        # 如果已经是\boxed{}格式，直接返回
        if answer.startswith("\\boxed{") and answer.endswith("}"):
            return answer
        
        # 清理答案，移除特殊字符
        clean_answer = re.sub(r'[^\w\d]', '', answer)
        if clean_answer:
            return f"\\boxed{{{clean_answer}}}"
        else:
            return f"\\boxed{{{answer}}}"
    
    def process_single_item(self, 
                           item: Dict, 
                           require_analysis: bool = True) -> Dict:
        """处理单个数据项"""
        item_id = item.get("id", "unknown")
        
        # 提取消息内容
        system_content = ""
        user_content = ""
        
        for msg in item["messages"]:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content']
        
        # 格式化输入
        input_text = self.format_input_text(system_content, user_content, require_analysis)
        
        # 生成响应
        response = self.generate_response(input_text)
        
        # 提取分析和答案
        analysis = self.extract_analysis(response)
        answer_raw = self.extract_answer(response)
        
        # 确保答案以\boxed{}格式
        answer = self.ensure_boxed_format(answer_raw)
        
        # 如果分析为空但响应不为空，使用完整响应作为分析
        if not analysis and response:
            analysis = response
        
        # 清理分析文本，移除可能的\boxed{}格式
        if "\\boxed{" in analysis:
            analysis = re.sub(r'\\boxed\{[^}]*\}', '', analysis).strip()
        
        return {
            "id": item_id,
            "analysis": analysis,
            "answer": answer,
            "full_response": response,
            "input_preview": input_text[-200:] if len(input_text) > 200 else input_text,
        }
    
    def process_file(self, 
                    input_file_path: str, 
                    output_file_path: str,
                    require_analysis: bool = True,
                    max_items: Optional[int] = None,
                    output_format: str = "jsonl") -> Dict:
        """
        处理输入文件并生成输出文件
        
        Args:
            input_file_path: 输入文件路径
            output_file_path: 输出文件路径
            require_analysis: 是否要求生成分析
            max_items: 最大处理项数，None表示处理所有
            output_format: 输出格式，"jsonl"或"json"
        
        Returns:
            处理统计信息
        """
        print(f"开始处理文件: {input_file_path}")
        
        # 加载输入数据
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_data = [json.loads(line) for line in f if line.strip()]
        
        if max_items and max_items > 0:
            input_data = input_data[:max_items]
        
        total_items = len(input_data)
        print(f"找到 {total_items} 个数据项")
        
        # 处理每个数据项
        results = []
        processed_count = 0
        
        import time
        start_time = time.time()
        
        for i, item in enumerate(input_data):
            try:
                result = self.process_single_item(item, require_analysis)
                results.append(result)
                processed_count += 1
                
                # 进度显示
                if (i + 1) % 10 == 0 or i == 0 or i == total_items - 1:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1) if (i + 1) > 0 else 0
                    remaining = avg_time * (total_items - i - 1)
                    print(f"进度: {i+1}/{total_items} | 平均时间: {avg_time:.2f}s/项 | 预计剩余: {remaining:.0f}s")
                    
            except Exception as e:
                print(f"处理第 {i+1} 项时出错 (ID: {item.get('id', 'unknown')}): {e}")
                # 添加错误项
                results.append({
                    "id": item.get("id", f"error_{i}"),
                    "analysis": f"处理出错: {str(e)}",
                    "answer": "\\boxed{}",
                    "full_response": "",
                    "error": True,
                    "error_message": str(e),
                })
        
        # 计算处理时间
        total_time = time.time() - start_time
        
        # 保存结果
        print(f"保存结果到: {output_file_path}")
        
        if output_format.lower() == "jsonl":
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for result in results:
                    # 只保存必要字段
                    output_item = {
                        "id": result["id"],
                        "analysis": result["analysis"],
                        "answer": result["answer"],
                    }
                    f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        else:
            # JSON格式
            output_data = []
            for result in results:
                output_item = {
                    "id": result["id"],
                    "analysis": result["analysis"],
                    "answer": result["answer"],
                }
                output_data.append(output_item)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 生成统计信息
        successful = sum(1 for r in results if "error" not in r or not r["error"])
        has_answer = sum(1 for r in results if r["answer"] and r["answer"] != "\\boxed{}")
        has_analysis = sum(1 for r in results if r["analysis"] and len(r["analysis"]) > 10)
        
        stats = {
            "total_items": total_items,
            "processed_count": processed_count,
            "successful_count": successful,
            "items_with_answer": has_answer,
            "items_with_analysis": has_analysis,
            "answer_rate": has_answer / total_items * 100 if total_items > 0 else 0,
            "analysis_rate": has_analysis / total_items * 100 if total_items > 0 else 0,
            "total_time_seconds": total_time,
            "average_time_per_item": total_time / total_items if total_items > 0 else 0,
            "output_file": output_file_path,
            "output_format": output_format,
        }
        
        # 打印统计信息
        print("\n" + "="*60)
        print("处理完成统计")
        print("="*60)
        print(f"总数据项: {stats['total_items']}")
        print(f"成功处理: {stats['successful_count']}")
        print(f"包含答案: {stats['items_with_answer']} ({stats['answer_rate']:.1f}%)")
        print(f"包含分析: {stats['items_with_analysis']} ({stats['analysis_rate']:.1f}%)")
        print(f"总用时: {stats['total_time_seconds']:.1f}秒")
        print(f"平均每项: {stats['average_time_per_item']:.2f}秒")
        print(f"输出文件: {stats['output_file']}")
        
        # 保存统计信息
        stats_file = output_file_path.replace(".json", "_stats.json").replace(".jsonl", "_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计信息: {stats_file}")
        
        return stats


# 使用示例和命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="5G网络优化模型批量处理工具")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径 (JSONL格式)")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--base_model", type=str, 
                       default="/root/autodl-tmp/Qwen2.5-1.5B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--adapter", type=str,
                       default="/root/autodl-tmp/fineTuningLab/qwen2/15B_output/qwen2.5-1.5b/lora_weights",
                       help="LoRA适配器路径")
    parser.add_argument("--max_items", type=int, default=None, help="最大处理项数")
    parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "json"], 
                       help="输出格式")
    parser.add_argument("--no_analysis", action="store_true", help="不生成分析，只生成答案")
    parser.add_argument("--device", type=str, default="auto", help="设备设置")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                       help="数据类型")
    
    args = parser.parse_args()
    
    # 创建模型实例
    print("="*60)
    print("5G网络优化模型批量处理器")
    print("="*60)
    
    model = Qwen5GOptimizerModel(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        device=args.device,
        torch_dtype=args.dtype
    )
    
    # 处理文件
    stats = model.process_file(
        input_file_path=args.input,
        output_file_path=args.output,
        require_analysis=not args.no_analysis,
        max_items=args.max_items,
        output_format=args.format
    )
    
    # 输出示例
    print("\n" + "="*60)
    print("输出示例 (前3项):")
    print("="*60)
    
    # 读取并显示前几项输出
    try:
        with open(args.output, 'r', encoding='utf-8') as f:
            if args.format == "jsonl":
                lines = f.readlines()[:3]
                for line in lines:
                    item = json.loads(line.strip())
                    print(f"ID: {item['id']}")
                    print(f"分析: {item['analysis'][:100]}..." if len(item['analysis']) > 100 else f"分析: {item['analysis']}")
                    print(f"答案: {item['answer']}")
                    print("-" * 40)
            else:
                data = json.load(f)
                for item in data[:3]:
                    print(f"ID: {item['id']}")
                    print(f"分析: {item['analysis'][:100]}..." if len(item['analysis']) > 100 else f"分析: {item['analysis']}")
                    print(f"答案: {item['answer']}")
                    print("-" * 40)
    except Exception as e:
        print(f"读取输出文件时出错: {e}")
    
    print("\n处理完成!")