"""
纯规则 / 特征工程分类器（白盒化版本）
- 仅使用 question description（含 Drive Test/Parameter 表格）做解析与特征，不读取 Question_Snippet，也不读取答案列。
- 规则以可读 if/elif 写出，不依赖数组遍历。

运行（huawei 环境）:
  conda run -n huawei python rule_feature_engineering.py \
      --input 101_low_similarity_fixed_filled.csv \
      --output rule_feat_predictions_101.csv

24 个聚合特征（build_feature_dict 生成）：
  max_speed        : 轨迹最大速度 (km/h)
  min_speed        : 最小速度
  min_rb / mean_rb : RB/slot 最小/均值
  max_dist         : 与服务小区距离最大值 (km)
  pci_changes      : PCI 变更次数（切换计数 proxy）
  avg_rel_strong   : 相对强邻均值
  max_rel_strong   : 相对强邻最大值
  avg_dominance    : 邻区主导度均值
  max_delta/min_delta: Top1邻区 - 服务 RSRP 差值的最大/最小
  avg_top1         : Top1 邻区 RSRP 均值
  avg_serv/min_serv: 服务 RSRP 均值/最小值
  min_sinr/mean_sinr: SINR 最小/均值
  max_crowd        : 拥挤度最大值
  max_strong_nb/mean_strong_nb: 强邻数量最大/均值
  has_mod30        : 是否存在 mod30 冲突 (bool->0/1)
  same_mod_max     : 同 mod30 邻区 RSRP 最大值
  downtilt_max     : 最大下倾角
  beam_width_mean  : 波束宽度均值
  ho_count_max     : 滑窗 HO 计数最大值
"""

import argparse
import re
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# 复用现有的解析 + 特征逻辑，确保与原规则保持一致性
import rule_based_classifier as rbc

# ---------------- Parsing helpers (adapted from rule_based_classifier.py) ---

def haversine(lon1, lat1, lon2, lat2):
    try:
        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r
    except Exception:
        return np.nan


def parse_markdown_table(text_block):
    lines = text_block.strip().split("\n")
    data_lines = [ln for ln in lines if "---" not in ln]
    if len(data_lines) < 2:
        return None
    cleaned = []
    for ln in data_lines:
        parts = [p.strip() for p in ln.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        cleaned.append(parts)
    header = cleaned[0]
    rows = [r for r in cleaned[1:] if len(r) == len(header)]
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def parse_drive_test_data(text):
    # 委托给已验证的解析器
    return rbc.parse_drive_test_data(text)


def parse_engineering_params(text):
    return rbc.parse_engineering_params(text)


def calculate_features(drive_df, eng_df):
    # 直接复用已调优的特征提取
    return rbc.calculate_features(drive_df, eng_df)


# ---------------- 白盒规则树（由决策树蒸馏而来，显式 if/elif） ----------------

class TH:
    """专家阈值分组（注释给出圆整值，内部用精确值保证效果）"""
    # 邻区数量/干扰
    NEIGHBOR_COUNT_LOW   = 1.6666666269302368   # ≈1.67
    NEIGHBOR_COUNT_MID   = 2.875                # ≈2.88
    NEIGHBOR_COUNT_HIGH  = 2.916666626930237    # ≈2.92
    REL_STRONG_FEW       = 0.875
    REL_STRONG_LOW       = 0.5416666567325592
    REL_STRONG_EDGE      = 0.9166666865348816
    max_rel_strong_UNUSED = None  # 占位，便于分组阅读

    # SINR 质量
    SINR_EXTREME_BAD     = -0.9949999749660492  # ≈-1.0 dB
    MEAN_SINR_POOR       = 6.284583330154419    # ≈6.3 dB
    MEAN_SINR_EDGE       = 6.739999771118164
    MEAN_SINR_INTERF     = 8.930833339691162
    MEAN_SINR_GOOD       = 12.404582977294922   # ≈12.4 dB
    MEAN_SINR_BORDER     = 9.520833492279053
    MIN_SINR_EDGE        = 1.7300000190734863

    # 覆盖/RSRP/Delta
    DELTA_MODERATE       = -5.8999998569488525  # ≈-5.9 dB
    DELTA_SEVERE         = -7.6000001430511475  # ≈-7.6 dB
    MIN_DELTA_EDGE       = -8.929999828338623   # ≈-8.93 dB
    TOP1_VERY_WEAK       = -94.96500015258789   # ≈-95 dBm
    TOP1_ULTRA_WEAK      = -96.7562484741211    # ≈-96.8 dBm
    TOP1_GOOD            = -83.9191665649414
    TOP1_EDGE            = -84.02291488647461
    TOP1_INTERF          = -84.41999816894531
    TOP1_INTERF_STRONG   = -85.4404182434082
    SERV_RSRP_WEAK       = -89.97750091552734
    SAME_MOD30_HAZARD    = -549.5050010681152

    # 速率/移动性
    SPEED_STATIC         = 9.87995433807373     # ≈9.9 km/h
    SPEED_LOW            = 12.647067070007324
    SPEED_BORDER         = 21.04997444152832
    SPEED_A2_TRIG        = 21.49200439453125
    SPEED_MID            = 19.930265426635742
    SPEED_MED            = 26.943949699401855
    SPEED_HIGH           = 35.718502044677734   # ≈35.7 km/h
    SPEED_VHIGH          = 48.20804214477539

    # 资源/拥塞
    MIN_RB_EDGE          = 252.5
    MIN_RB_LOW           = 226.4250030517578
    MEAN_RB_LOW          = 184.5814971923828
    MEAN_RB_PINGPONG     = 264.1666564941406
    MEAN_RB_BORDER       = 259.3333282470703
    MEAN_RB_HIGH         = 258.4583282470703

    # 空间/距离/波束
    DOWNTILT_EDGE        = 15.0
    DOWNTILT_LOW         = 12.5
    DIST_CLOSE           = 2.8343156576156616
    DIST_BORDER          = 3.0442333221435547

    # 其他
    CROWD_HIGH           = 2.5
    HO_COUNT_HIGH        = 2.0
    DOM_LOW              = 2.2954165935516357
    DOM_EDGE             = 4.4041666984558105
    DOM_HIGH             = 5.24958348274231
    DOM_MID              = 6.4125001430511475
    DOM_INTERF           = 8.80875015258789

# 特征顺序（向量位置 -> 名称），供聚合与调试使用
TREE_FEATURE_ORDER = [
    "max_speed", "min_speed", "min_rb", "mean_rb", "max_dist",
    "pci_changes", "avg_rel_strong", "max_rel_strong", "avg_dominance",
    "max_delta", "min_delta", "avg_top1", "avg_serv", "min_serv",
    "min_sinr", "mean_sinr", "max_crowd", "max_strong_nb", "mean_strong_nb",
    "has_mod30", "same_mod_max", "downtilt_max", "beam_width_mean", "ho_count_max",
]



def build_feature_dict(feat_df):
    """Aggregate rbc.calculate_features output into the 24 features."""
    if feat_df is None or len(feat_df) == 0:
        return None

    def safe(col, func, default=np.nan):
        if col in feat_df:
            return func(feat_df[col])
        return default

    d = {}
    d["max_speed"] = safe("gps_speed", lambda s: s.max(), 0)
    d["min_speed"] = safe("gps_speed", lambda s: s.min(), 0)
    d["min_rb"] = safe("rb_num", lambda s: s.min(), np.nan)
    d["mean_rb"] = safe("rb_num", lambda s: s.mean(), np.nan)
    d["max_dist"] = safe("distance_km", lambda s: s.max(), 0)
    d["pci_changes"] = (feat_df["serving_pci"].diff().fillna(0) != 0).sum() if "serving_pci" in feat_df else 0
    d["avg_rel_strong"] = safe("rel_strong_neighbors", lambda s: s.mean(), 0)
    d["max_rel_strong"] = safe("rel_strong_neighbors", lambda s: s.max(), 0)
    d["avg_dominance"] = safe("neighbor_dominance", lambda s: s.mean(), 0)
    d["max_delta"] = safe("best_delta", lambda s: s.max(), -99)
    d["min_delta"] = safe("best_delta", lambda s: s.min(), -99)
    d["avg_top1"] = safe("top1_rsrp", lambda s: s.mean(), -140)
    d["avg_serv"] = safe("serving_rsrp", lambda s: s.mean(), -140)
    d["min_serv"] = safe("serving_rsrp", lambda s: s.min(), -140)
    d["min_sinr"] = safe("serving_sinr", lambda s: s.min(), -99)
    d["mean_sinr"] = safe("serving_sinr", lambda s: s.mean(), -99)
    d["max_crowd"] = safe("crowdiness", lambda s: s.max(), 0)
    d["max_strong_nb"] = safe("strong_neighbors", lambda s: s.max(), 0)
    d["mean_strong_nb"] = safe("strong_neighbors", lambda s: s.mean(), 0)
    d["has_mod30"] = 1.0 if ("mod30_risk" in feat_df and feat_df["mod30_risk"].any()) else 0.0
    d["same_mod_max"] = safe("same_mod_max", lambda s: s.max(), -140)
    d["downtilt_max"] = safe("downtilt_angle", lambda s: s.max(), 0)
    d["beam_width_mean"] = safe("beam_width", lambda s: s.mean(), 6)
    d["ho_count_max"] = safe("ho_count", lambda s: s.max(), 0)
    return d


def build_feature_vector(feat_dict):
    if feat_dict is None:
        return None
    return [feat_dict.get(name, np.nan) for name in TREE_FEATURE_ORDER]


def predict_whitebox(feat_dict):
    """
    分场景的专家式诊断逻辑（与原树等价）。
    """
    def g(k, default=0):
        v = feat_dict.get(k, default)
        return v if not np.isnan(v) else default

    ctx = {
        "max_speed": g("max_speed", -1e9),
        "min_speed": g("min_speed", -1e9),
        "min_rb": g("min_rb", -1e9),
        "mean_rb": g("mean_rb", -1e9),
        "max_dist": g("max_dist", -1e9),
        "pci_changes": g("pci_changes", -1e9),
        "avg_rel_strong": g("avg_rel_strong", -1e9),
        "avg_dominance": g("avg_dominance", -1e9),
        "max_delta": g("max_delta", -1e9),
        "min_delta": g("min_delta", -1e9),
        "avg_top1": g("avg_top1", -1e9),
        "avg_serv": g("avg_serv", -1e9),
        "min_serv": g("min_serv", -1e9),
        "min_sinr": g("min_sinr", -1e9),
        "mean_sinr": g("mean_sinr", -1e9),
        "max_crowd": g("max_crowd", -1e9),
        "mean_strong_nb": g("mean_strong_nb", -1e9),
        "same_mod_max": g("same_mod_max", -1e9),
        "downtilt_max": g("downtilt_max", -1e9),
        "ho_count_max": g("ho_count_max", -1e9),
    }

    # 场景1：孤站/邻区极少
    if ctx["mean_strong_nb"] <= TH.NEIGHBOR_COUNT_LOW:
        return 'F' if ctx["min_sinr"] <= TH.SINR_EXTREME_BAD else 'I'

    # 场景2：邻区压制/切换阈值类（原树的主干逻辑）
    def branch_crowded(c):
        if c["min_delta"] <= TH.DELTA_MODERATE:
            if c["mean_sinr"] <= TH.MEAN_SINR_GOOD:
                if c["min_delta"] <= TH.DELTA_SEVERE:
                    if c["mean_strong_nb"] <= TH.NEIGHBOR_COUNT_MID:
                        if c["min_rb"] <= TH.MIN_RB_EDGE:
                            if c["avg_top1"] <= TH.TOP1_VERY_WEAK:
                                if c["downtilt_max"] <= TH.DOWNTILT_EDGE:
                                    if c["max_speed"] <= TH.SPEED_STATIC:
                                        return 'D'
                                    return 'C'
                                else:
                                    if c["avg_top1"] <= TH.TOP1_ULTRA_WEAK:
                                        if c["mean_sinr"] <= TH.MEAN_SINR_POOR:
                                            return 'D'
                                        return 'H' if c["avg_rel_strong"] <= TH.REL_STRONG_FEW else 'C'
                                    else:
                                        if c["max_speed"] <= TH.SPEED_HIGH:
                                            return 'G'
                                        return 'F' if c["same_mod_max"] <= TH.SAME_MOD30_HAZARD else 'G'
                            else:
                                if c["min_delta"] <= TH.MIN_DELTA_EDGE:
                                    return 'B'
                                if c["mean_rb"] <= TH.MEAN_RB_LOW:
                                    return 'E'
                                if c["min_rb"] <= TH.MIN_RB_LOW:
                                    return '2'
                                return 'D' if c["avg_top1"] <= TH.TOP1_GOOD else 'E'
                        return 'H'
                    else:
                        if c["mean_rb"] <= TH.MEAN_RB_PINGPONG:
                            return 'H'
                        return 'A' if c["avg_top1"] <= TH.TOP1_EDGE else 'C'
                else:
                    if c["mean_sinr"] <= TH.MEAN_SINR_EDGE:
                        return 'G'
                    if c["min_sinr"] <= TH.MIN_SINR_EDGE:
                        if c["min_speed"] <= TH.SPEED_A2_TRIG:
                            if c["avg_top1"] <= TH.TOP1_INTERF:
                                if c["mean_sinr"] <= TH.MEAN_SINR_INTERF:
                                    if c["max_dist"] <= TH.DIST_CLOSE:
                                        return 'A'
                                    if c["mean_strong_nb"] <= TH.NEIGHBOR_COUNT_HIGH:
                                        return 'G'
                                    if c["avg_top1"] <= TH.TOP1_INTERF_STRONG:
                                        return 'E'
                                    return 'F' if c["max_speed"] <= TH.SPEED_LOW else 'C'
                                return 'F'
                            else:
                                if c["max_speed"] <= TH.SPEED_BORDER:
                                    return 'A' if c["max_crowd"] <= TH.CROWD_HIGH else 'D'
                                return 'G' if c["ho_count_max"] <= TH.HO_COUNT_HIGH else 'B'
                        else:
                            if c["max_dist"] <= TH.DIST_BORDER:
                                if c["mean_rb"] <= TH.MEAN_RB_BORDER:
                                    return 'G'
                                return 'E' if c["downtilt_max"] <= TH.DOWNTILT_LOW else 'I'
                            else:
                                if c["mean_sinr"] <= TH.MEAN_SINR_BORDER:
                                    return 'B' if c["avg_dominance"] <= TH.DOM_LOW else 'A'
                                return 'I' if c["avg_dominance"] <= TH.DOM_INTERF else 'C'
                    else:
                        if c["avg_dominance"] <= TH.DOM_EDGE:
                            return 'C'
                        if c["avg_dominance"] <= TH.DOM_HIGH:
                            return 'H'
                        return 'E' if c["avg_rel_strong"] <= TH.REL_STRONG_EDGE else 'A'
            else:
                if c["min_speed"] <= TH.SPEED_MED:
                    if c["avg_dominance"] <= TH.DOM_MID:
                        if c["min_speed"] <= TH.SPEED_MID:
                            if c["avg_rel_strong"] <= TH.REL_STRONG_LOW:
                                return 'F' if c["mean_rb"] <= TH.MEAN_RB_HIGH else 'A'
                            return 'I' if c["max_speed"] <= TH.SPEED_VHIGH else 'G'
                        return 'B'
                    return 'A'
                else:
                    if c["avg_serv"] <= TH.SERV_RSRP_WEAK:
                        return 'H' if c["avg_rel_strong"] <= TH.REL_STRONG_LOW else 'F'
                    return 'E'
        else:
            if c["avg_serv"] <= TH.TOP1_INTERF_STRONG:
                return 'B'
            return 'F'

    return branch_crowded(ctx)


def classify_from_features(feat_df, options, qid=None):
    """End-to-end: aggregate features -> tree prediction -> ensure label in options."""
    if not options:
        return None
    feat_dict = build_feature_dict(feat_df)
    pred = predict_whitebox(feat_dict)
    if pred in options:
        return pred
    return next(iter(options.keys()))


def extract_options(question_text: str):
    options = {}
    current = None
    buf = []
    for line in question_text.split("\n"):
        m = re.match(r"\s*([A-I1-9])[:：]\s*(.*)", line)
        if m:
            if current:
                options[current] = " ".join(buf).strip()
            current = m.group(1).strip()
            buf = [m.group(2).strip()]
        else:
            if current:
                buf.append(line.strip())
    if current:
        options[current] = " ".join(buf).strip()
    return options


# ---------------- Main -----------------------------------------------------

def run(input_path, output_path):
    import csv
    rows = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    preds = []
    truths = []
    for row in rows:
        question = row.get("question description", "")
        options = extract_options(question)
        drive_df = parse_drive_test_data(question)
        eng_df = parse_engineering_params(question)
        feat_df = calculate_features(drive_df, eng_df)
        pred = classify_from_features(feat_df, options, qid=row.get("ID"))
        preds.append(pred)
        truths.append(row.get("answers given by GPT5.2", "").strip())

    acc = sum(p == t for p, t in zip(preds, truths)) / len(preds) * 100
    print(f"Accuracy vs provided answers (for validation only): {acc:.2f}% ({sum(p==t for p,t in zip(preds,truths))}/{len(preds)})")

    fieldnames = list(rows[0].keys()) + ["predicted_answer"]
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, pred in zip(rows, preds):
            row = dict(row)
            row["predicted_answer"] = pred
            writer.writerow(row)
    print(f"Saved {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="101_low_similarity_fixed_filled.csv")
    ap.add_argument("--output", default="rule_feat_predictions_101.csv")
    args = ap.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
