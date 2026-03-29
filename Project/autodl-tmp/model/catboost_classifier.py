import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

from rule_based_classifier import (
    parse_drive_test_data,
    parse_engineering_params,
    calculate_features,
    process_single_question,
)

EASY_CLASSES = {"C2", "C5", "C7", "C8"}


def aggregate_features(question_text):
    """
    Turn a single question into a flat numeric feature vector for CatBoost.
    Includes variance/quantile stats and missing flags.
    """
    ddf = parse_drive_test_data(question_text)
    edf = parse_engineering_params(question_text)
    fdf = calculate_features(ddf, edf)

    if fdf is None or len(fdf) == 0:
        return None

    def safe_stat(series, func, default=np.nan):
        try:
            val = func(series)
            return val if pd.notna(val) else default
        except Exception:
            return default

    q = lambda s, p, default=np.nan: default if s is None else safe_stat(s, lambda x: np.nanpercentile(x, p))

    eng_missing = int(edf is None or len(edf) == 0)
    sinr_missing = int("serving_sinr" not in fdf.columns or fdf["serving_sinr"].isna().all())
    rows_len = len(fdf)

    rel_str = fdf.get("rel_strong_neighbors")
    strong_rel_ge1 = float((rel_str >= 1).sum()) / rows_len if rel_str is not None else np.nan
    strong_rel_ge2 = float((rel_str >= 2).sum()) / rows_len if rel_str is not None else np.nan
    strong_rel_ge3 = float((rel_str >= 3).sum()) / rows_len if rel_str is not None else np.nan

    strong_abs = fdf.get("strong_neighbors")
    strong_abs_ge1 = float((strong_abs >= 1).sum()) / rows_len if strong_abs is not None else np.nan
    strong_abs_ge2 = float((strong_abs >= 2).sum()) / rows_len if strong_abs is not None else np.nan

    pci_unique = (
        ddf["5G KPI PCell RF Serving PCI"].nunique(dropna=True)
        if ddf is not None and "5G KPI PCell RF Serving PCI" in ddf.columns
        else 0
    )

    features = {
        "rows": rows_len,
        "eng_missing": eng_missing,
        "sinr_missing": sinr_missing,
        # RSRP
        "serving_rsrp_mean": safe_stat(fdf["serving_rsrp"], np.mean),
        "serving_rsrp_min": safe_stat(fdf["serving_rsrp"], np.min),
        "serving_rsrp_max": safe_stat(fdf["serving_rsrp"], np.max),
        "serving_rsrp_p25": q(fdf["serving_rsrp"], 25),
        "serving_rsrp_p50": q(fdf["serving_rsrp"], 50),
        "serving_rsrp_p75": q(fdf["serving_rsrp"], 75),
        "serving_rsrp_std": safe_stat(fdf["serving_rsrp"], np.std),
        "serving_rsrp_range": safe_stat(fdf["serving_rsrp"], lambda s: s.max() - s.min()),
        # SINR
        "serving_sinr_mean": safe_stat(fdf.get("serving_sinr"), np.mean),
        "serving_sinr_min": safe_stat(fdf.get("serving_sinr"), np.min),
        "serving_sinr_p25": q(fdf.get("serving_sinr"), 25),
        "serving_sinr_p75": q(fdf.get("serving_sinr"), 75),
        "serving_sinr_std": safe_stat(fdf.get("serving_sinr"), np.std),
        "serving_sinr_range": safe_stat(fdf.get("serving_sinr"), lambda s: s.max() - s.min()),
        # Throughput
        "throughput_mean": safe_stat(fdf.get("throughput"), np.mean),
        "throughput_min": safe_stat(fdf.get("throughput"), np.min),
        "throughput_max": safe_stat(fdf.get("throughput"), np.max),
        "throughput_p25": q(fdf.get("throughput"), 25),
        "throughput_p75": q(fdf.get("throughput"), 75),
        "throughput_std": safe_stat(fdf.get("throughput"), np.std),
        # RB
        "rb_mean": safe_stat(fdf["rb_num"], np.mean),
        "rb_min": safe_stat(fdf["rb_num"], np.min),
        "rb_p25": q(fdf["rb_num"], 25),
        # Speed
        "gps_speed_max": safe_stat(fdf["gps_speed"], np.max),
        "gps_speed_mean": safe_stat(fdf["gps_speed"], np.mean),
        # Distance
        "distance_max": safe_stat(fdf["distance_km"], np.max),
        "distance_mean": safe_stat(fdf["distance_km"], np.mean),
        "distance_p75": q(fdf["distance_km"], 75),
        "distance_p90": q(fdf["distance_km"], 90),
        # Downtilt
        "downtilt_max": safe_stat(fdf["downtilt_angle"], np.max),
        "downtilt_mean": safe_stat(fdf["downtilt_angle"], np.mean),
        # Delta
        "best_delta_max": safe_stat(fdf["best_delta"], np.max),
        "best_delta_mean": safe_stat(fdf["best_delta"], np.mean),
        "best_delta_p75": q(fdf["best_delta"], 75),
        "best_delta_range": safe_stat(fdf["best_delta"], lambda s: s.max() - s.min()),
        # Neighbor power
        "top1_rsrp_mean": safe_stat(fdf["top1_rsrp"], np.mean),
        "top1_rsrp_max": safe_stat(fdf["top1_rsrp"], np.max),
        "top1_rsrp_p75": q(fdf["top1_rsrp"], 75),
        "top1_minus_serving_mean": safe_stat(fdf["top1_rsrp"] - fdf["serving_rsrp"], np.mean),
        "neighbor_dominance_mean": safe_stat(fdf["neighbor_dominance"], np.mean),
        "neighbor_dominance_max": safe_stat(fdf["neighbor_dominance"], np.max),
        # Crowdiness
        "strong_neighbors_mean": safe_stat(fdf["strong_neighbors"], np.mean),
        "strong_neighbors_max": safe_stat(fdf["strong_neighbors"], np.max),
        "strong_neighbors_p75": q(fdf["strong_neighbors"], 75),
        "rel_strong_neighbors_mean": safe_stat(fdf.get("rel_strong_neighbors"), np.mean),
        "rel_strong_neighbors_p75": q(fdf.get("rel_strong_neighbors"), 75),
        "crowdiness_max": safe_stat(fdf.get("crowdiness"), np.max),
        "crowdiness_mean": safe_stat(fdf.get("crowdiness"), np.mean),
        "crowdiness_p90": q(fdf.get("crowdiness"), 90),
        "rel_strong_frac_ge1": strong_rel_ge1,
        "rel_strong_frac_ge2": strong_rel_ge2,
        "rel_strong_frac_ge3": strong_rel_ge3,
        "strong_abs_frac_ge1": strong_abs_ge1,
        "strong_abs_frac_ge2": strong_abs_ge2,
        # Beam/tilt
        "beam_width_mean": safe_stat(fdf.get("beam_width"), np.mean),
        "tilt_over_beam": safe_stat(fdf["downtilt_angle"], np.mean) / (safe_stat(fdf.get("beam_width"), np.mean) or 1),
        "tilt_times_distance": safe_stat(fdf["downtilt_angle"], np.mean) * safe_stat(fdf["distance_km"], np.mean),
        # Mod30
        "mod30_any": int(fdf["mod30_risk"].any()),
        "mod30_frac": safe_stat(fdf["mod30_risk"], np.mean, default=0.0),
        "same_mod_max": safe_stat(fdf.get("same_mod_max"), np.max, default=-140),
        "same_mod_count_max": safe_stat(fdf.get("same_mod_count"), np.max, default=0),
        # Mobility
        "pci_change_count": int((fdf["serving_pci"].diff().fillna(0) != 0).sum()),
        "pci_change_rate": int((fdf["serving_pci"].diff().fillna(0) != 0).sum()) / max(len(fdf), 1),
        "ho_count_max": safe_stat(fdf.get("ho_count"), np.max, default=0),
        "pci_unique_count": pci_unique,
    }

    for k, v in list(features.items()):
        if pd.isna(v):
            features[k] = -999

    return features


def build_datasets(train_path, phase1_test_path, phase1_truth_path):
    train_df = pd.read_csv(train_path)
    val_truth = pd.read_csv(phase1_truth_path)
    test_questions = pd.read_csv(phase1_test_path).set_index("ID")["question"].to_dict()

    X_train, y_train = [], []
    for _, row in train_df.iterrows():
        ans = row["answer"]
        if ans in EASY_CLASSES:
            continue
        feats = aggregate_features(row["question"])
        if feats is None:
            continue
        X_train.append(feats)
        y_train.append(ans)

    val_records = []
    for _, row in val_truth.iterrows():
        tid = row["ID"]
        label = row["Qwen3-32B"]
        base_id = tid.rsplit("_", 1)[0] if "_" in tid else tid
        question = test_questions.get(base_id)
        if question is None:
            continue
        rule_pred = process_single_question(question)
        feats = aggregate_features(question) if label not in EASY_CLASSES else None
        val_records.append(
            {
                "ID": tid,
                "label": label,
                "rule_pred": rule_pred,
                "needs_model": label not in EASY_CLASSES,
                "features": feats,
            }
        )

    return pd.DataFrame(X_train), pd.Series(y_train), val_records


def train_models(X_train, y_train, class_weights):
    params_list = [
        dict(
            loss_function="MultiClass",
            depth=8,
            learning_rate=0.08,
            iterations=500,
            eval_metric="Accuracy",
            random_seed=42,
            class_weights=class_weights,
            bootstrap_type="Bernoulli",
            subsample=0.9,
            colsample_bylevel=0.8,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            od_type="Iter",
            od_wait=60,
            verbose=100,
        ),
        dict(
            loss_function="MultiClass",
            depth=8,
            learning_rate=0.075,
            iterations=600,
            eval_metric="Accuracy",
            random_seed=7,
            class_weights=class_weights,
            bootstrap_type="Bernoulli",
            subsample=0.9,
            colsample_bylevel=0.8,
            l2_leaf_reg=4.5,
            random_strength=0.6,
            od_type="Iter",
            od_wait=60,
            verbose=100,
        ),
        dict(
            loss_function="MultiClass",
            depth=7,
            learning_rate=0.085,
            iterations=500,
            eval_metric="Accuracy",
            random_seed=99,
            class_weights=class_weights,
            bootstrap_type="Bernoulli",
            subsample=0.9,
            colsample_bylevel=0.85,
            l2_leaf_reg=4.0,
            random_strength=0.7,
            od_type="Iter",
            od_wait=60,
            verbose=100,
        ),
    ]

    models = []
    train_pool = Pool(X_train, y_train)
    for params in params_list:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, use_best_model=True)
        models.append(model)
    return models


def predict_ensemble(models, X_df):
    if len(X_df) == 0:
        return [], []
    class_order = models[0].classes_
    prob_sum = np.zeros((len(X_df), len(class_order)))
    pool = Pool(X_df)
    for model in models:
        probs = model.predict_proba(pool)
        prob_sum += probs
    prob_avg = prob_sum / len(models)
    pred_indices = np.argmax(prob_avg, axis=1)
    max_probs = prob_avg.max(axis=1)
    preds = [class_order[i] for i in pred_indices]
    return preds, max_probs


def train_and_eval():
    X_train, y_train, val_records = build_datasets(
        "train.csv",
        "phase_1_test.csv",
        "phase_1_test_truth.csv",
    )

    class_counts = y_train.value_counts().to_dict()
    total = len(y_train)
    class_weights = {cls: total / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}

    models = train_models(X_train, y_train, class_weights)

    # Prepare validation features for difficult classes
    val_features = []
    val_idx = []
    for i, rec in enumerate(val_records):
        if rec["needs_model"] and rec["features"] is not None:
            val_features.append(rec["features"])
            val_idx.append(i)

    val_preds_model, _ = predict_ensemble(models, pd.DataFrame(val_features))

    # Combine rule + model
    combined_preds = []
    model_ptr = 0
    for rec in val_records:
        if rec["rule_pred"] in EASY_CLASSES:
            combined_preds.append(rec["rule_pred"])
        elif rec["needs_model"] and rec["features"] is not None:
            combined_preds.append(val_preds_model[model_ptr])
            model_ptr += 1
        else:
            combined_preds.append(rec["rule_pred"])

    actual = [rec["label"] for rec in val_records]
    val_acc = np.mean([p == a for p, a in zip(combined_preds, actual)]) * 100
    print(f"Validation accuracy (rule+2xCatBoost) on phase1 truth: {val_acc:.2f}% (n={len(actual)})")

    pred_df = pd.DataFrame(
        list(zip([rec["ID"] for rec in val_records], combined_preds, actual)),
        columns=["ID", "predicted", "actual"],
    )
    pred_df.to_csv("catboost_phase1_predictions.csv", index=False)
    print("Saved validation predictions to catboost_phase1_predictions.csv")

    # Save first model
    models[0].save_model("catboost_model.cbm")
    print("Saved model to catboost_model.cbm")


if __name__ == "__main__":
    train_and_eval()
