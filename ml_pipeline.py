"""
AI vs Real Images — Full ML Pipeline
=====================================
Covers:
  • Data Loading & EDA
  • Feature Engineering
  • Logistic Regression (with regularization tuning)
  • Decision Tree (with pruning / max_depth tuning)
  • Random Forest (ensemble baseline)
  • Preprocessing: StandardScaler, PCA
  • Cross-Validation (StratifiedKFold)
  • Hyperparameter search (GridSearchCV)
  • Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC
  • Confusion Matrix
  • Feature Importance (tree + permutation)
  • Learning Curves
  • Calibration Curves
  • Saving models + JSON results for API
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ── Sklearn ──────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV,
    learning_curve, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# ── Visualization (optional) ─────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

from feature_extraction import extract_all_features, get_feature_names

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("dataset")          # Kaggle dataset root
OUTPUT_DIR  = Path("output")
MODEL_DIR   = Path("models")
SEED        = 42
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10
N_SPLITS    = 5
PCA_VARIANCE = 0.95                    # Retain 95% variance

for d in [OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(root: Path):
    """
    Expects:
      dataset/
        train/
          AI/   *.jpg
          Real/ *.jpg
        test/
          AI/
          Real/
    """
    print("\n[1] Loading dataset …")
    records = []
    for split in ["train", "test"]:
        for label in ["AI", "Real"]:
            folder = root / split / label
            if not folder.exists():
                print(f"  ⚠ Missing: {folder}")
                continue
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    records.append({"path": str(img_path), "label": label, "split": split})

    df = pd.DataFrame(records)
    print(f"  Total images found: {len(df)}")
    print(df.groupby(["split", "label"]).size().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame, cache_file: str = "output/features.npz"):
    """Extract or load cached features."""
    if os.path.exists(cache_file):
        print(f"\n[2] Loading cached features from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data["X"], data["y"], data["splits"]

    print("\n[2] Extracting features …")
    X_list, y_list, splits_list = [], [], []
    failed = 0

    for i, row in df.iterrows():
        try:
            feats = extract_all_features(row["path"])
            X_list.append(feats)
            y_list.append(1 if row["label"] == "AI" else 0)
            splits_list.append(row["split"])
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  ✗ {row['path']}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  … {i + 1}/{len(df)} processed")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    splits = np.array(splits_list)
    np.savez(cache_file, X=X, y=y, splits=splits)
    print(f"  Feature matrix: {X.shape}  |  Failed: {failed}")
    return X, y, splits


# ─────────────────────────────────────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────────────────────────────────────
def exploratory_analysis(X, y, feature_names):
    print("\n[3] Exploratory Data Analysis")
    df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
    df["label"] = y

    stats = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_distribution": {
            "AI": int((y == 1).sum()),
            "Real": int((y == 0).sum())
        },
        "class_balance_ratio": round(float((y == 1).sum() / len(y)), 4),
        "feature_stats": {
            "mean_range": [round(float(X.mean(axis=0).min()), 4),
                           round(float(X.mean(axis=0).max()), 4)],
            "std_range":  [round(float(X.std(axis=0).min()), 4),
                           round(float(X.std(axis=0).max()), 4)],
            "nan_count": int(np.isnan(X).sum()),
            "inf_count": int(np.isinf(X).sum()),
        },
        "top_discriminative_features": []
    }

    # Mann-Whitney U-like score per feature (mean diff)
    ai_mean   = X[y == 1].mean(axis=0)
    real_mean = X[y == 0].mean(axis=0)
    std_all   = X.std(axis=0) + 1e-10
    effect_size = np.abs(ai_mean - real_mean) / std_all
    top_idx = effect_size.argsort()[::-1][:10]
    fn = feature_names[:X.shape[1]]
    stats["top_discriminative_features"] = [
        {"feature": fn[i], "effect_size": round(float(effect_size[i]), 4)}
        for i in top_idx
    ]

    print(f"  Samples: {stats['n_samples']}  |  Features: {stats['n_features']}")
    print(f"  Class balance — AI: {stats['class_distribution']['AI']}, "
          f"Real: {stats['class_distribution']['Real']}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(X_train, X_test):
    """Scale + PCA."""
    print("\n[4] Preprocessing …")
    # Handle NaN / Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test  = np.nan_to_num(X_test,  nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    pca = PCA(n_components=PCA_VARIANCE, random_state=SEED)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)

    print(f"  After PCA ({PCA_VARIANCE*100:.0f}% var): {X_train_p.shape[1]} components")
    return X_train_s, X_test_s, X_train_p, X_test_p, scaler, pca


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    """Full evaluation suite."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    fpr, tpr, thresholds = (roc_curve(y_test, y_prob) if y_prob is not None
                            else (None, None, None))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "log_loss":  round(log_loss(y_test, y_prob), 4) if y_prob is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {
            "fpr": fpr.tolist()[::5] if fpr is not None else [],
            "tpr": tpr.tolist()[::5] if tpr is not None else [],
        }
    }
    print(f"\n  ── {model_name} ──")
    print(f"  Acc: {metrics['accuracy']:.4f}  Prec: {metrics['precision']:.4f}  "
          f"Rec: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}  "
          f"AUC: {metrics['auc_roc']}")
    return metrics


def cross_validate_model(model, X, y, model_name, cv):
    """StratifiedKFold cross-validation."""
    scoring = ['accuracy', 'f1', 'roc_auc']
    cv_results = {}
    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        cv_results[metric] = {
            "scores": [round(s, 4) for s in scores.tolist()],
            "mean":   round(scores.mean(), 4),
            "std":    round(scores.std(), 4)
        }
    print(f"  CV {model_name}: "
          f"Acc={cv_results['accuracy']['mean']:.4f}±{cv_results['accuracy']['std']:.4f}  "
          f"AUC={cv_results['roc_auc']['mean']:.4f}±{cv_results['roc_auc']['std']:.4f}")
    return cv_results


def compute_learning_curve(model, X, y, model_name):
    """Learning curve (train-size vs. score)."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1, random_state=SEED
    )
    return {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).round(4).tolist(),
        "val_scores_mean":   val_scores.mean(axis=1).round(4).tolist(),
        "train_scores_std":  train_scores.std(axis=1).round(4).tolist(),
        "val_scores_std":    val_scores.std(axis=1).round(4).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train, X_test, y_test, cv):
    print("\n[5] Logistic Regression …")

    # Hyperparameter search
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    base_lr = LogisticRegression(max_iter=1000, random_state=SEED)
    gs = GridSearchCV(base_lr, param_grid, cv=cv, scoring="roc_auc",
                      n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best_lr = gs.best_estimator_

    print(f"  Best params: {gs.best_params_}  |  Best CV AUC: {gs.best_score_:.4f}")

    metrics = evaluate_model(best_lr, X_test, y_test, "Logistic Regression")
    cv_res  = cross_validate_model(best_lr, X_train, y_train, "LR", cv)
    lc      = compute_learning_curve(best_lr, X_train, y_train, "LR")

    # Coefficient analysis
    coefs = best_lr.coef_[0]
    coef_data = sorted(
        [{"index": int(i), "coef": round(float(c), 6)} for i, c in enumerate(coefs)],
        key=lambda x: abs(x["coef"]), reverse=True
    )[:20]

    # Regularization path (C vs. AUC from grid search)
    reg_path = []
    c_vals = param_grid["C"]
    for res in gs.cv_results_['params']:
        idx = list(gs.cv_results_['params']).index(res)
        reg_path.append({
            "C": res["C"], "penalty": res["penalty"],
            "mean_auc": round(gs.cv_results_['mean_test_score'][idx], 4)
        })

    result = {
        "metrics": metrics,
        "cv_results": cv_res,
        "learning_curve": lc,
        "best_params": gs.best_params_,
        "best_cv_auc": round(gs.best_score_, 4),
        "top_coefficients": coef_data,
        "regularization_path": reg_path
    }
    return best_lr, result


# ─────────────────────────────────────────────────────────────────────────────
# 7. DECISION TREE
# ─────────────────────────────────────────────────────────────────────────────
def train_decision_tree(X_train, y_train, X_test, y_test, cv, feature_names=None):
    print("\n[6] Decision Tree …")

    param_grid = {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "criterion": ["gini", "entropy"]
    }
    base_dt = DecisionTreeClassifier(random_state=SEED)
    gs = GridSearchCV(base_dt, param_grid, cv=cv, scoring="roc_auc",
                      n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best_dt = gs.best_estimator_

    print(f"  Best params: {gs.best_params_}  |  Best CV AUC: {gs.best_score_:.4f}")

    metrics = evaluate_model(best_dt, X_test, y_test, "Decision Tree")
    cv_res  = cross_validate_model(best_dt, X_train, y_train, "DT", cv)
    lc      = compute_learning_curve(best_dt, X_train, y_train, "DT")

    # Feature importance
    fi = best_dt.feature_importances_
    fn = feature_names[:len(fi)] if feature_names else [f"f_{i}" for i in range(len(fi))]
    top_fi = sorted(
        [{"feature": fn[i], "importance": round(float(fi[i]), 6)} for i in range(len(fi))],
        key=lambda x: x["importance"], reverse=True
    )[:20]

    # Depth vs AUC
    depth_auc = {}
    for item in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
        d = str(item[0]['max_depth'])
        depth_auc.setdefault(d, []).append(float(item[1]))
    depth_auc_summary = [
        {"max_depth": k, "mean_auc": round(np.mean(v), 4)}
        for k, v in depth_auc.items()
    ]

    # Text representation of top levels
    tree_text = export_text(best_dt, max_depth=3,
                            feature_names=fn[:best_dt.n_features_in_])

    # Permutation importance
    perm_imp = permutation_importance(best_dt, X_test, y_test,
                                      n_repeats=5, random_state=SEED, n_jobs=-1)
    top_perm = sorted(
        [{"feature": fn[i], "importance": round(float(perm_imp.importances_mean[i]), 6),
          "std": round(float(perm_imp.importances_std[i]), 6)}
         for i in range(len(fn))],
        key=lambda x: x["importance"], reverse=True
    )[:15]

    result = {
        "metrics": metrics,
        "cv_results": cv_res,
        "learning_curve": lc,
        "best_params": gs.best_params_,
        "best_cv_auc": round(gs.best_score_, 4),
        "top_feature_importances": top_fi,
        "permutation_importances": top_perm,
        "depth_vs_auc": depth_auc_summary,
        "tree_text": tree_text
    }
    return best_dt, result


# ─────────────────────────────────────────────────────────────────────────────
# 8. RANDOM FOREST (ensemble baseline)
# ─────────────────────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test, cv, feature_names=None):
    print("\n[7] Random Forest …")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                min_samples_split=5, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    cv_res  = cross_validate_model(rf, X_train, y_train, "RF", cv)
    lc      = compute_learning_curve(rf, X_train, y_train, "RF")

    fi = rf.feature_importances_
    fn = feature_names[:len(fi)] if feature_names else [f"f_{i}" for i in range(len(fi))]
    top_fi = sorted(
        [{"feature": fn[i], "importance": round(float(fi[i]), 6)} for i in range(len(fi))],
        key=lambda x: x["importance"], reverse=True
    )[:20]

    result = {
        "metrics": metrics,
        "cv_results": cv_res,
        "learning_curve": lc,
        "top_feature_importances": top_fi
    }
    return rf, result


# ─────────────────────────────────────────────────────────────────────────────
# 9. CALIBRATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def calibration_analysis(models_probs, y_test):
    """Check if predicted probabilities are well-calibrated."""
    results = {}
    for name, y_prob in models_probs.items():
        fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        brier = brier_score_loss(y_test, y_prob)
        results[name] = {
            "fraction_of_positives": fraction_pos.round(4).tolist(),
            "mean_predicted_value":  mean_pred.round(4).tolist(),
            "brier_score": round(brier, 4)
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    print("=" * 60)
    print(" AI vs Real Images — Predictive Analytics Pipeline")
    print("=" * 60)

    feature_names = get_feature_names()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # ── Load & extract ───────────────────────────────────────────────────────
    df = load_dataset(DATA_ROOT)
    if df.empty:
        print("\n⚠  No images found. Please place the Kaggle dataset at ./dataset/")
        print("   Expected structure: dataset/train/AI/*.jpg  dataset/train/Real/*.jpg")
        return

    X, y, splits = build_feature_matrix(df)

    # ── EDA ──────────────────────────────────────────────────────────────────
    eda_stats = exploratory_analysis(X, y, feature_names)

    # ── Split ────────────────────────────────────────────────────────────────
    train_mask = splits == "train"
    X_train_raw, X_test_raw = X[train_mask], X[~train_mask]
    y_train,     y_test     = y[train_mask], y[~train_mask]

    if len(X_test_raw) == 0:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )

    # ── Preprocess ───────────────────────────────────────────────────────────
    X_tr_s, X_te_s, X_tr_p, X_te_p, scaler, pca = preprocess(X_train_raw, X_test_raw)

    # ── Train models ─────────────────────────────────────────────────────────
    lr_model,  lr_results  = train_logistic_regression(X_tr_p, y_train, X_te_p, y_test, cv)
    dt_model,  dt_results  = train_decision_tree(X_tr_s, y_train, X_te_s, y_test, cv, feature_names)
    rf_model,  rf_results  = train_random_forest(X_tr_s, y_train, X_te_s, y_test, cv, feature_names)

    # ── Calibration ──────────────────────────────────────────────────────────
    models_probs = {
        "Logistic Regression": lr_model.predict_proba(X_te_p)[:, 1],
        "Decision Tree":       dt_model.predict_proba(X_te_s)[:, 1],
        "Random Forest":       rf_model.predict_proba(X_te_s)[:, 1],
    }
    calibration = calibration_analysis(models_probs, y_test)

    # ── Model comparison ─────────────────────────────────────────────────────
    comparison = [
        {**m["metrics"], "cv_auc_mean": m["cv_results"]["roc_auc"]["mean"],
         "cv_auc_std": m["cv_results"]["roc_auc"]["std"]}
        for m in [lr_results, dt_results, rf_results]
    ]

    # ── Save models ───────────────────────────────────────────────────────────
    for name, obj in [("scaler", scaler), ("pca", pca),
                      ("lr", lr_model), ("dt", dt_model), ("rf", rf_model)]:
        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    # ── Save JSON results ─────────────────────────────────────────────────────
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": round(time.time() - start, 1),
        "eda": eda_stats,
        "logistic_regression": lr_results,
        "decision_tree": dt_results,
        "random_forest": rf_results,
        "calibration": calibration,
        "model_comparison": comparison
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f" Pipeline complete in {results['runtime_seconds']}s")
    print(f" Results saved → {out_path}")
    print(f"{'=' * 60}\n")
    return results


if __name__ == "__main__":
    main()
