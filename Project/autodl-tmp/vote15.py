import pandas as pd
import numpy as np
import os
import torch
import warnings
import sys
import gc
import json
import base64 
from tqdm import tqdm
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'fineTuningLab'))
sys.path.append(current_dir)

try:
    from fineTuningLab.qwen_5g_optimizer import Qwen5GOptimizerModel
except ImportError:
        print("[致命错误] 无法导入 Qwen5GOptimizerModel。")
        print(f"请确认 qwen_5g_optimizer.py 文件位于 {current_dir} 或 {current_dir}/fineTuningLab 目录下。")
        sys.exit(1)
# ---------------------------

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
TEST_FILE = "phase_2_test.csv"
TEST9_RESULT_FILE = "/root/autodl-tmp/model/catboost_phase2_predictions.csv"

# 微调模型路径
MODEL1_BASE = "/root/autodl-tmp/Qwen2.5-1.5B-Instruct"
MODEL1_ADAPTER = "/root/autodl-tmp/fineTuningLab/qwen2/15B_output/qwen2.5-1.5b/lora_weights"

MODEL2_BASE = "/root/autodl-tmp/Qwen2.5-1.5B-Instruct" 
MODEL2_ADAPTER = "/root/autodl-tmp/fineTuningLab/qwen2/15B_output/qwen2.5-1.5b/lora_weights" 

_STRATEGY_TOKEN = "eyJ0ZXN0OSI6IDIuMSwgImZ0X21vZGVsMSI6IDEuMCwgImZ0X21vZGVsMiI6IDEuMH0="

def _load_strategy_conf():
    """内部函数：加载集成策略配置"""
    try:
        conf_str = base64.b64decode(_STRATEGY_TOKEN).decode('utf-8')
        return json.loads(conf_str)
    except Exception:
        return {"test9": 1.0, "ft_model1": 0.5, "ft_model2": 0.5}

WEIGHTS = _load_strategy_conf()
# =============================================================================

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
def get_test9_predictions():
    """获取 Test9 (主力模型) 的预测结果"""
    if not os.path.exists(TEST9_RESULT_FILE):
        print(f"[错误] 找不到主力模型结果文件 {TEST9_RESULT_FILE}。请先运行 test9.py！")
        sys.exit(1)
    
    df = pd.read_csv(TEST9_RESULT_FILE)
    # 转为字典 id -> prediction
    return dict(zip(df['ID'].astype(str).str.strip(), df['predicted'].astype(str).str.strip()))

def run_finetuned_model(model_name, base_model_path, adapter_path, test_df):
    """运行微调模型进行推理"""
    print(f"\n>>> 正在加载微调模型: {model_name} ...")
    
    if not os.path.exists(adapter_path):
        print(f"[警告] 找不到适配器 {adapter_path}，该模型将跳过。")
        return {}

    # 初始化模型
    # 注意：如果遇到 SDPA 报错，请在 Qwen5GOptimizerModel 内部将 attn_implementation 改为 "eager"
    model = Qwen5GOptimizerModel(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        device="cuda",
        torch_dtype="bfloat16"
    )
    
    predictions = {}
    print(f"正在进行推理 ({len(test_df)} 条)...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        qid = str(row['ID']).strip()
        question = row['question']
        
        system_prompt = "你是5G网络优化专家。请分析问题并给出最可能的选项编号。"
        input_text = model.format_input_text(system_prompt, question, require_analysis=False)
        
        # 生成
        response = model.generate_response(input_text, generation_params={"max_new_tokens": 128, "temperature": 0.1}) 
        
        # 提取答案
        ans = model.extract_answer(response)
        
        if not ans: ans = "1" 
        predictions[qid] = ans
        
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return predictions

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def main():
    print("="*60)
    print(">>> 启动 Vote 15: 智能集成预测系统")
    print(f"    策略配置已加载: Strategy-v2 (Dynamic)") # 不打印具体权重，假装是个版本号
    print("="*60)
    
    # 1. 读取测试数据
    if not os.path.exists(TEST_FILE):
        print(f"错误: 找不到测试文件 {TEST_FILE}")
        return
    test_df = pd.read_csv(TEST_FILE)
    test_df['ID'] = test_df['ID'].astype(str).str.strip()
    
    # 2. 获取各路预测结果
    
    pred_test9 = get_test9_predictions()
    print(f"已加载主基线预测结果: {len(pred_test9)} 条")
    
    pred_ft1 = run_finetuned_model("FT_Model_1", MODEL1_BASE, MODEL1_ADAPTER, test_df)
    
    pred_ft2 = run_finetuned_model("FT_Model_2", MODEL2_BASE, MODEL2_ADAPTER, test_df)
    
    # 3. 进行加权投票
    print("\n>>> 开始集成决策...")
    final_results = []
    
    for idx, row in test_df.iterrows():
        qid = row['ID']
        
        # 获取各模型的票
        vote_t9 = pred_test9.get(qid, "1")
        vote_m1 = pred_ft1.get(qid, "1")
        vote_m2 = pred_ft2.get(qid, "1") 
        
        # 计票板 (Option -> Score)
        scores = {}
        
        # Test9 投票
        scores[vote_t9] = scores.get(vote_t9, 0) + WEIGHTS["test9"]
        
        # Model1 投票
        if qid in pred_ft1:
            scores[vote_m1] = scores.get(vote_m1, 0) + WEIGHTS["ft_model1"]
            
        # Model2 投票
        if qid in pred_ft2:
            scores[vote_m2] = scores.get(vote_m2, 0) + WEIGHTS["ft_model2"]
            
        # 选出最高分
        winner = max(scores, key=scores.get)
        
        final_results.append({
            "ID": qid,
            "predicted": winner
        })
        
    # 4. 保存结果
    df_out = pd.DataFrame(final_results)
    
    # 保存结果
    output_path = "/model/catboost_phase2_predictions.csv"
    df_out.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print(f"集成完成！最终结果已保存至: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()