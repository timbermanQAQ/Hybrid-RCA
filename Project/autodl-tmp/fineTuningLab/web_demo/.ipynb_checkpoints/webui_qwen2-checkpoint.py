import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../qwen2')
import json
import torch
import argparse
import gradio as gr
from evaluate import load_model

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, required=True, help="基础模型路径")
parser.add_argument("--ckpt", type=str, default=None, required=True, help="LoRA检查点路径")
args = parser.parse_args()

# 加载模型
tokenizer, model = load_model(args.model, args.ckpt)

def analyze_5g_network(question):
    """分析5G网络问题"""
    # 构建系统提示词
    system_prompt = "你是一个5G无线网络优化专家。请根据提供的用户平面路测数据和工程参数数据，分析吞吐率下降的原因。答案必须用\\boxed{}格式包裹。"
    
    # 构建完整提示词
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    # 生成回答
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response = tokenizer.decode(outputs[:, inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
    
    return response

def extract_answer(response):
    """从回复中提取答案（如\boxed{C1}）"""
    import re
    # 查找\boxed{}格式的答案
    match = re.search(r'\\boxed{([^}]+)}', response)
    if match:
        return match.group(1)
    return "未找到答案"

def chat_interface(question, history):
    """聊天界面处理函数"""
    if not question.strip():
        return history, ""
    
    # 生成回答
    answer = analyze_5g_network(question)
    
    # 提取答案
    extracted = extract_answer(answer)
    
    # 添加到历史
    history.append((question, answer))
    
    return history, extracted

def batch_analysis(input_text):
    """批量分析模式：输入多个问题，每行一个"""
    questions = [q.strip() for q in input_text.split('\n') if q.strip()]
    results = []
    
    for i, question in enumerate(questions, 1):
        answer = analyze_5g_network(question)
        extracted = extract_answer(answer)
        results.append(f"问题 {i}: {question[:50]}...")
        results.append(f"答案: {extracted}")
        results.append(f"完整分析:\n{answer}")
        results.append("-" * 50)
    
    return "\n".join(results)

def main():
    with gr.Blocks(title="5G网络分析专家系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛰️ 5G无线网络分析专家系统")
        gr.Markdown("上传5G路测数据问题，模型将分析原因并给出答案")
        
        with gr.Tabs():
            with gr.TabItem("💬 交互分析"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="分析对话", height=500)
                        question_input = gr.Textbox(
                            label="输入5G网络问题",
                            placeholder="例如：分析以下5G路测数据，找出吞吐率下降原因...",
                            lines=3
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("分析", variant="primary")
                            clear_btn = gr.Button("清空")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 分析结果")
                        answer_output = gr.Textbox(
                            label="提取的答案",
                            placeholder="将显示提取的答案（如C1, C2等）",
                            lines=2
                        )
                        gr.Markdown("#### 示例问题格式")
                        gr.Markdown("""
                        ```
                        Analyze the 5G wireless network drive-test data...
                        Identify the reason for throughput dropping...
                        From the following 8 potential root causes...
                        ```
                        """)
            
            with gr.TabItem("📁 批量分析"):
                gr.Markdown("### 批量处理多个问题")
                batch_input = gr.Textbox(
                    label="输入多个问题（每行一个）",
                    placeholder="输入第一个问题...\n输入第二个问题...",
                    lines=10
                )
                batch_output = gr.Textbox(
                    label="批量分析结果",
                    lines=15
                )
                batch_btn = gr.Button("开始批量分析", variant="primary")
            
            with gr.TabItem("⚙️ 模型配置"):
                gr.Markdown("### 模型参数设置")
                temperature = gr.Slider(0, 1, value=0.7, label="温度 (Temperature)")
                max_tokens = gr.Slider(100, 2048, value=1024, step=100, label="最大生成长度")
                
                gr.Markdown(f"""
                **当前模型配置**
                - 基础模型: `{args.model}`
                - LoRA检查点: `{args.ckpt}`
                - 设备: `{"cuda" if torch.cuda.is_available() else "cpu"}`
                """)
        
        # 事件处理
        submit_btn.click(
            chat_interface,
            inputs=[question_input, chatbot],
            outputs=[chatbot, answer_output]
        )
        
        question_input.submit(
            chat_interface,
            inputs=[question_input, chatbot],
            outputs=[chatbot, answer_output]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, answer_output]
        )
        
        batch_btn.click(
            batch_analysis,
            inputs=batch_input,
            outputs=batch_output
        )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
        debug=False
    )

if __name__ == "__main__":
    main()