# data_preprocess.py
import json
import re
from torch.utils.data import Dataset


class InputOutputDataset(Dataset):
    """5G网络分析数据集类 - 正确的Causal LM格式"""
    def __init__(self, data, tokenizer, args):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args.max_source_length
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建完整对话，并进行智能截断
        full_text = self.build_truncated_conversation(item)
        
        # 编码完整对话
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        
        # 创建标签 - 复制input_ids
        labels = input_ids.clone()
        
        # 找到assistant部分开始的位置，将之前的标签设为-100
        # 这意味着模型只预测assistant部分的输出
        text = full_text
        assistant_start = text.find("<|im_start|>assistant\n")
        
        if assistant_start > 0:
            # 计算assistant之前的token数量
            before_assistant = text[:assistant_start]
            before_tokens = self.tokenizer.encode(before_assistant, add_special_tokens=False)
            before_length = len(before_tokens)
            
            # 将assistant之前的所有token设为-100
            labels[:before_length] = -100
        
        # 将padding部分的标签设为-100
        if self.ignore_pad_token_for_loss:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def build_truncated_conversation(self, item):
        """构建截断的对话，确保assistant部分在max_length内"""
        if isinstance(item, dict) and "messages" in item:
            messages = item["messages"]
            
            # 提取system、user、assistant内容
            system_content = ""
            user_content = ""
            assistant_content = ""
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_content = msg['content']
                elif msg['role'] == 'user':
                    user_content = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_content = msg['content']
            
            # 构建system部分
            system_part = f"<|im_start|>system\n{system_content}<|im_end|>\n"
            
            # 构建assistant部分（包括标记和内容）
            assistant_part = f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
            
            # 计算system和assistant部分的token数量
            system_assistant_tokens = len(self.tokenizer.encode(
                system_part + assistant_part,
                add_special_tokens=False
            ))
            
            # 剩余给user内容的token数量
            remaining_tokens = self.max_length - system_assistant_tokens - 10  # 留一些余量
            
            if remaining_tokens > 0:
                # 编码user内容
                user_tokens = self.tokenizer.encode(user_content, add_special_tokens=False)
                
                # 如果user内容太长，进行截断
                if len(user_tokens) > remaining_tokens:
                    truncated_user_tokens = user_tokens[:remaining_tokens]
                    truncated_user_content = self.tokenizer.decode(truncated_user_tokens)
                    # 确保解码不会产生不完整的内容
                    if truncated_user_content.endswith("...") or truncated_user_content.endswith("…"):
                        truncated_user_content = truncated_user_content[:-1]
                else:
                    truncated_user_content = user_content
                
                # 构建user部分
                user_part = f"<|im_start|>user\n{truncated_user_content}<|im_end|>\n"
                
                # 完整对话
                full_text = system_part + user_part + assistant_part
            else:
                # 如果连system和assistant都放不下，只保留system和极短的user提示
                user_part = "<|im_start|>user\nAnalyze the 5G network data: [DATA TRUNCATED]<|im_end|>\n"
                full_text = system_part + user_part + assistant_part
            
            return full_text
        
        # 其他格式处理（保持原样）
        elif "instruction" in item and "input" in item and "output" in item:
            full_text = f"""<|im_start|>system
你是5G网络优化专家。<|im_end|>
<|im_start|>user
{item['instruction']}
{item['input']}<|im_end|>
<|im_start|>assistant
{item['output']}<|im_end|>"""
            return full_text
        
        else:
            # 简单分割，前80%作为输入，后20%作为回答
            text = str(item)
            split_point = int(len(text) * 0.8)
            input_text = text[:split_point]
            target_text = text[split_point:]
            return f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{target_text}<|im_end|>"


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
        r'最终答案[：:]\s*([A-Z]\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return "未找到答案"


def load_and_process_data(file_path, tokenizer, args):
    """加载并处理数据文件"""
    data = []
    
    # 判断文件格式
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    # 创建数据集
    dataset = InputOutputDataset(data, tokenizer, args)
    return dataset