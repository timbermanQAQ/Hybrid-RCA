import pandas as pd
import numpy as np
import re
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import warnings
import gc
import sys
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# 忽略警告
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. 依赖检查与配置
# -----------------------------------------------------------------------------
try:
    from rule_based_classifier import process_single_question
    from catboost_classifier import (
        aggregate_features, 
        train_models, 
        predict_ensemble, 
        EASY_CLASSES
    )
    # 导入特征工程逻辑
    from rule_feature_engineering import (
        parse_drive_test_data,
        parse_engineering_params,
        calculate_features,
        classify_from_features
    )
except ImportError as e:
    print("错误: 无法导入依赖文件。")
    exit()

# 配置
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = "./model_cache"
OPTIMIZED_RATIONALE_FILE = "optimized_rationales.csv" 

C_DESCRIPTIONS = {
    "C1": "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "C2": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    "C3": "A neighboring cell provides higher throughput.",
    "C4": "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    "C5": "Frequent handovers degrade performance.",
    "C6": "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    "C7": "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    "C8": "Average scheduled RBs are below 160, affecting throughput."
}

TELECOM_KEYWORDS = [
    'rsrp', 'sinr', 'throughput', 'cell', 'handover', 'pci', 'rb', 'gnodeb',
    'downtilt', 'beam', 'neighbor', 'ul', 'dl', '5g', 'kpi', 'arfcn', 'coverage'
]

# -----------------------------------------------------------------------------
# 2. 核心辅助函数
# -----------------------------------------------------------------------------

def parse_options(question_text):
    options = {}
    lines = question_text.split('\n')
    pattern = re.compile(r'^\s*([A-Za-z0-9]{1,2})\s*[:\.]\s+(.*)$')
    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            oid, desc = match.groups()
            if len(desc.strip()) > 0 and oid.upper() != "GIVEN":
                options[oid.upper()] = desc.strip()
    if not options:
        inline_pat = re.compile(r'(?:(?<=\s)|^)([A-Za-z0-9]{1,2})\s*[:\.]\s*([^:]+?)(?=(\s+[A-Za-z0-9]{1,2}\s*[:\.])|$)')
        for m in inline_pat.finditer(question_text):
            oid, desc = m.group(1), m.group(2).strip()
            if desc and oid.upper() != "GIVEN":
                options[oid.upper()] = desc
    return options

def format_options(options):
    return "\n".join([f"{k}: {v}" for k, v in options.items()])

def robust_parse_output(raw_output, options):
    """强力解析器"""
    clean_text = raw_output.strip()
    box_match = re.search(r'\\boxed\s*\{([^}]+)\}', clean_text, re.IGNORECASE)
    if box_match: return robust_parse_output(box_match.group(1).strip(), options)
    
    clean_text = re.sub(r'\b(Option|Answer|The correct option is|The answer is)\b', '', clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.replace("**", "").replace("`", "").replace("'", "").replace('"', "").strip(".: ")
    clean_upper = clean_text.upper()

    if clean_upper in options: return clean_upper
    
    start_match = re.match(r'^([A-Z0-9]+)', clean_text)
    if start_match:
        potential_id = start_match.group(1).upper()
        if potential_id in options: return potential_id

    for oid, desc in options.items():
        if clean_text.replace(" ", "") == desc.replace(" ", ""): return oid
        if clean_text.isdigit() and desc.isdigit():
             if int(clean_text) == int(desc): return oid
             
    best_ratio, best_oid = 0, None
    for oid, desc in options.items():
        if desc in clean_text: return oid
        ratio = SequenceMatcher(None, clean_text, desc).ratio()
        if ratio > best_ratio: best_ratio, best_oid = ratio, oid
    if best_ratio > 0.8: return best_oid

    if len(clean_text) > 0:
        first_word = clean_text.split()[0].upper()
        if first_word in options: return first_word
            
    return list(options.keys())[0] if options else "1"

# -----------------------------------------------------------------------------
# 3. 核心工具类：Qwen 推理引擎
# -----------------------------------------------------------------------------

class QwenSolver:
    def __init__(self, model_name=MODEL_NAME):
        print(f"\n[Qwen] 正在初始化高精度模型 (强制离线): {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(CACHE_DIR):
            print(f"[错误] 缓存目录 {CACHE_DIR} 不存在！")
            self.model = None
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=CACHE_DIR, local_files_only=True
            )
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
            print(f"[Qwen] 模型加载成功！推理精度: {dtype}")
        except Exception as e:
            print(f"[Fatal] 模型加载失败: {e}")
            self.model = None

    def clear_memory_context(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print("   [系统] 显存已清理。")

    def generate_raw(self, prompt, max_new_tokens=128):
        if self.model is None: return "1"
        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.0
            )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def solve_math_with_rationale(self, question, options, rationale):
        """数学题专用"""
        clean_rat = re.sub(r'\\boxed\s*\{[^}]+\}', '[Calculated Value]', rationale)
        patterns = [
            r'(?i)(Therefore|Thus|So|Hence|Consequently),?\s*(the|it|we).*?(correct|answer|option).*', 
            r'(?i)(The correct answer is|The answer is|Option).+',
            r'(?i)Conclusion\s*:?.+',
            r'(?i)Final Answer\s*:?.*'
        ]
        for pat in patterns:
            clean_rat = re.sub(pat, "\n[Analysis Ends Here]", clean_rat)
        
        options_text = format_options(options)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Expert Analysis:\n{clean_rat}\n\n"
            f"Based on the analysis above, select the correct option.\n"
            f"Instruction: Start your response with the Option ID."
        )
        raw_out = self.generate_raw(prompt, max_new_tokens=128)
        return robust_parse_output(raw_out, options)

# -----------------------------------------------------------------------------
# 4. 其他逻辑函数
# -----------------------------------------------------------------------------

def is_telecom_question(text):
    text_lower = text.lower()
    score = 0
    for k in TELECOM_KEYWORDS:
        if k in text_lower:
            score += 1
            if k in ['rsrp', 'sinr', 'gnodeb', 'downtilt', 'pci', 'handover']:
                return True
    return score >= 2

def find_best_match_id_v2(predicted_c_label, phase2_options):
    if not phase2_options: return "1"
    if predicted_c_label not in C_DESCRIPTIONS:
        return list(phase2_options.keys())[0]
    target_desc = C_DESCRIPTIONS[predicted_c_label]
    option_ids = list(phase2_options.keys())
    option_descs = list(phase2_options.values())
    target_numbers = set(re.findall(r'\d+', target_desc))
    all_texts = [target_desc] + option_descs
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except ValueError:
        return option_ids[0]
    final_scores = []
    for i, desc in enumerate(option_descs):
        base_score = cosine_sim[i]
        option_numbers = set(re.findall(r'\d+', desc))
        common = len(target_numbers.intersection(option_numbers))
        final_scores.append(base_score + (common * 0.4)) 
    best_idx = np.argmax(final_scores)
    return option_ids[best_idx]

def build_train_data(train_path, phase1_test_path, phase1_truth_path):
    print(f"正在构建 CatBoost 训练集...")
    if not os.path.exists(train_path): return pd.DataFrame(), pd.Series()
    train_df = pd.read_csv(train_path)
    extra_data = []
    if os.path.exists(phase1_test_path) and os.path.exists(phase1_truth_path):
        p1_truth = pd.read_csv(phase1_truth_path)
        p1_test = pd.read_csv(phase1_test_path).set_index("ID")["question"].to_dict()
        for _, row in p1_truth.iterrows():
            bid = row["ID"].rsplit("_", 1)[0] if "_" in row["ID"] else row["ID"]
            if bid in p1_test: extra_data.append((p1_test[bid], row["Qwen3-32B"])) 
    X, y = [], []
    for _, row in train_df.iterrows():
        if row["answer"] not in EASY_CLASSES:
            f = aggregate_features(row["question"])
            if f:
                X.append(f)
                y.append(row["answer"])
    for q, ans in extra_data:
        if ans not in EASY_CLASSES:
            f = aggregate_features(q)
            if f:
                X.append(f)
                y.append(ans)
    return pd.DataFrame(X), pd.Series(y)

# -----------------------------------------------------------------------------
# 5. 主程序
# -----------------------------------------------------------------------------

def main():
    print("="*60)
    print(">>> 启动 Test 10: 终极混合架构 (修正规则引擎调用逻辑)")
    print("="*60)
    
    # 0. 加载思维链
    math_rationales = {}
    target_rat_file = OPTIMIZED_RATIONALE_FILE if os.path.exists(OPTIMIZED_RATIONALE_FILE) else "intermediate_rationales.csv"
    if os.path.exists(target_rat_file):
        print(f"正在加载思维链库: {target_rat_file} ...")
        df_rat = pd.read_csv(target_rat_file)
        for _, row in df_rat.iterrows():
            rid = str(row['ID']).strip()
            rat_content = row.get('step1_rationale', row.get('rationale', ''))
            if pd.notna(rat_content):
                math_rationales[rid] = rat_content
        print(f"成功加载 {len(math_rationales)} 条黄金思维链。")
    
    # A. 训练 CatBoost
    X_train, y_train = build_train_data("train.csv", "phase_1_test.csv", "phase_1_test_truth.csv")
    models = None
    if len(X_train) > 0:
        counts = y_train.value_counts().to_dict()
        weights = {c: len(y_train)/(len(counts)*v) for c, v in counts.items()}
        print(f"CatBoost 训练集: {len(X_train)} 条")
        models = train_models(X_train, y_train, weights)

    # B. 加载高精度 Qwen
    qwen = QwenSolver()

    # C. 读取测试集
    test_path = "phase_2_test.csv"
    if not os.path.exists(test_path):
        print("错误: 未找到 phase_2_test.csv")
        return
    test_df = pd.read_csv(test_path)
    
    results_store = {} 
    cb_indices = []
    cb_features = []
    
    print(f"正在处理 {len(test_df)} 条数据...")
    math_count = 0
    telecom_count = 0

    # 阶段 1: 初步分类处理
    for idx, row in test_df.iterrows():
        qid = str(row['ID']).strip()
        q_text = row['question']
        options = parse_options(q_text)
        
        # 检查是否有黄金思维链
        rationale = math_rationales.get(qid)
        
        if rationale:
            math_count += 1
            ans = qwen.solve_math_with_rationale(q_text, options, rationale)
            results_store[idx] = {'pred_id': ans, 'c_label': 'Math_CoT', 'type': 'Math_CoT'}
        
        elif not is_telecom_question(q_text):
            math_count += 1
            ans = qwen.solve_math_with_rationale(q_text, options, "") 
            results_store[idx] = {'pred_id': ans, 'c_label': 'Math_Fallback', 'type': 'Math_Fallback'}
            
        else:
            telecom_count += 1
            rule_pred = process_single_question(q_text)
            
            if rule_pred in EASY_CLASSES:
                ans = find_best_match_id_v2(rule_pred, options)
                results_store[idx] = {'pred_id': ans, 'c_label': rule_pred, 'type': 'Telecom_Easy'}
            else:
                feats = aggregate_features(q_text)
                if feats and models:
                    cb_indices.append(idx)
                    cb_features.append(feats)
                    results_store[idx] = {'pred_id': None, 'c_label': None, 'type': 'Telecom_WaitCB'} 
                else:
                    fallback_label = rule_pred if rule_pred != "Unknown" else "Unknown"
                    ans = find_best_match_id_v2(fallback_label, options)
                    results_store[idx] = {'pred_id': ans, 'c_label': fallback_label, 'type': 'Telecom_Rule_Fallback'}

    print(f"初步分类: 电信题 {telecom_count}, 数学/CoT题 {math_count}")

    # CatBoost 批处理
    if cb_features and models:
        print(f"CatBoost 批量推理: {len(cb_features)} 条")
        preds, _ = predict_ensemble(models, pd.DataFrame(cb_features))
        for i, idx in enumerate(cb_indices):
            c_label = preds[i]
            q_text = test_df.loc[idx, 'question']
            options = parse_options(q_text)
            ans = find_best_match_id_v2(c_label, options)
            results_store[idx] = {'pred_id': ans, 'c_label': c_label, 'type': 'Telecom_CatBoost'}

    # 隔离清理
    print("\n" + "="*60)
    print(">>> 中场休息: 清理显存...")
    print("="*60)
    qwen.clear_memory_context() 

    # --------------------------------------------------------------------------------
    # 阶段 2: 低置信度诊断与 【特征工程规则修正】
    # --------------------------------------------------------------------------------
    print(f">>> 阶段 2: 启动低置信度诊断与特征工程规则修正")
    fix_count = 0
    
    for idx, row in test_df.iterrows():
        res = results_store.get(idx)

        if not res or 'Math' in res['type']: 
            continue
            
        q_text = row['question']
        options = parse_options(q_text)
        current_c_label = res['c_label']
        
        # 诊断逻辑：检查语义匹配度
        is_low_confidence = False
        if current_c_label in C_DESCRIPTIONS:
            target_desc = C_DESCRIPTIONS[current_c_label]
            best_ratio = 0
            if options:
                best_ratio = max([SequenceMatcher(None, target_desc, opt).ratio() for opt in options.values()])
            if best_ratio < 0.35:
                is_low_confidence = True
        elif current_c_label == "Unknown" or current_c_label is None:
            is_low_confidence = True
            
        # === 核心替换：使用特征工程规则进行修正 ===
        if is_low_confidence:
            try:
                # 1. 提取表格和工程参数
                drive_df = parse_drive_test_data(q_text)
                eng_df = parse_engineering_params(q_text)
                
                # 2. 计算特征
                feat_df = calculate_features(drive_df, eng_df)
                
                # 3. 白盒规则分类 (注意：这里返回的是 Option ID，例如 'A' 或 '1')
                rule_feat_label = classify_from_features(feat_df, options, qid=row['ID'])
                
                # 4. 【修复】直接使用返回的 Label，不再映射！
                if rule_feat_label and rule_feat_label in options:
                    new_ans = rule_feat_label
                    results_store[idx]['pred_id'] = new_ans
                    results_store[idx]['type'] += '_FixedByRuleFeat'
                    fix_count += 1
                
                if fix_count % 10 == 0:
                    print(f"\r[特征规则修正] 已处理: {fix_count}", end="")
                    
            except Exception as e:
                pass

    print(f"\n\n>>> 修正完成。共检测并修正 {fix_count} 条。")

    # 保存
    output = []
    for idx, row in test_df.iterrows():
        res = results_store.get(idx, {'pred_id': '1'})
        output.append({
            "ID": row["ID"],
            "predicted": res['pred_id']
        })
    
    df_out = pd.DataFrame(output)
    df_out.to_csv("catboost_phase2_predictions.csv", index=False)
    print("\n结果已保存至 catboost_phase2_predictions.csv")

if __name__ == "__main__":
    main()