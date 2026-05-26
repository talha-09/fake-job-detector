"""
  Fake Job Posting Detection — Model Training Script

  Models:
    1. XGBoost (primary)  — gradient boosted trees with scale_pos_weight
    2. Logistic Regression (secondary) — with SMOTE oversampling

  Both trained on the SAME 5,009-dimensional feature space:
    - 5,000 TF-IDF text features (unigrams + bigrams)
    - 9 structural features (has_salary, has_logo, etc.)
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import nltk
import scipy.sparse as sp
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "fake_job_postings.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# NLTK
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# STEP 1: Load & Inspect Data
print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)
print(f"  Rows     : {df.shape[0]:,}")
print(f"  Columns  : {df.shape[1]}")
print(f"  Real jobs: {(df['fraudulent'] == 0).sum():,}")
print(f"  Fake jobs: {(df['fraudulent'] == 1).sum():,}  ({df['fraudulent'].mean()*100:.1f}%)")


# STEP 2: Combine Text Columns
print("\nCombining text fields...")

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
df["combined_text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
print(f"  Combined columns: {TEXT_COLS}")


# STEP 3: Clean Text
print("\nCleaning text...")

def clean_text(text: str) -> str:
    """Lowercase, strip HTML, remove non-letters, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters only
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

df["cleaned_text"] = df["combined_text"].apply(clean_text)
print("  Text cleaning done.")


# STEP 4: Structural Features
print("\nBuilding structural (non-text) features...")

df["has_salary"]       = df["salary_range"].notna().astype(int)
df["has_logo"]         = df["has_company_logo"].astype(int)
df["has_questions"]    = df["has_questions"].notna().astype(int)
df["telecommuting"]    = df["telecommuting"].fillna(0).astype(int)
df["title_len"]        = df["title"].fillna("").apply(len)
df["desc_len"]         = df["description"].fillna("").apply(len)
df["profile_len"]      = df["company_profile"].fillna("").apply(len)
df["short_desc"]       = (df["desc_len"] < 200).astype(int)
df["no_profile"]       = (df["profile_len"] == 0).astype(int)

STRUCT_COLS = [
    "has_salary", "has_logo", "has_questions", "telecommuting",
    "title_len", "desc_len", "profile_len", "short_desc", "no_profile"
]

struct_features = df[STRUCT_COLS].values
print(f"  Structural features built: {STRUCT_COLS}")
print(f"  Jobs with salary range   : {df['has_salary'].sum():,} / {len(df):,}")
print(f"  Jobs with company logo   : {df['has_logo'].sum():,} / {len(df):,}")


# STEP 5: TF-IDF Vectorization
print("\nVectorizing text with TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

X_tfidf = tfidf.fit_transform(df["cleaned_text"])
print(f"  TF-IDF shape     : {X_tfidf.shape}")

# Combine TF-IDF with structural features using sparse hstack
X_struct_sparse = sp.csr_matrix(struct_features)
X = sp.hstack([X_tfidf, X_struct_sparse])
y = df["fraudulent"].values
print(f"  Combined shape   : {X.shape}  (TF-IDF + {len(STRUCT_COLS)} structural features)")


# STEP 6: Train / Test Split
print("\nSplitting the dataset (train/val/test = 60/20/20)...")

# Keep the same test split size (20%) as your existing evaluation code.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split remaining 80% into train and validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25,          # 0.25 * 0.80 = 0.20 overall
    random_state=42,
    stratify=y_trainval
)

print(f"  Train samples: {X_train.shape[0]:,}")
print(f"  Val   samples: {X_val.shape[0]:,}")
print(f"  Test  samples: {X_test.shape[0]:,}")
print(f"  Train fake   : {int(y_train.sum())}")
print(f"  Val   fake   : {int(y_val.sum())}")
print(f"  Test  fake   : {int(y_test.sum())}")

# Class imbalance ratio — used by XGBoost (train only)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
imbalance_ratio = round(neg / pos, 2)
print(f"  Imbalance ratio (real/fake) [train]: {imbalance_ratio}")


# Balance class distribution (Logistic Regression only)
print("\nBalancing training data with SMOTE (for LR only)...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  Before SMOTE — Real: {(y_train == 0).sum()}, Fake: {(y_train == 1).sum()}")
print(f"  After  SMOTE — Real: {(y_train_bal == 0).sum()}, Fake: {(y_train_bal == 1).sum()}")


# Train Logistic Regression (threshold tuned on validation)
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=3000,
    C=1.0,
    solver="lbfgs",
    random_state=42
)
lr_model.fit(X_train_bal, y_train_bal)

# Threshold selection on VAL (not test)
lr_probs_val = lr_model.predict_proba(X_val)[:, 1]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, lr_probs_val)
f1_vals = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8)
best_idx = f1_vals[:-1].argmax()
best_threshold = thresholds[best_idx]
print(f"  LR best threshold (max F1 on validation): {best_threshold:.3f}")

# Final predictions on TEST
lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_pred = (lr_probs >= best_threshold).astype(int)

lr_acc  = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, zero_division=0)
lr_rec  = recall_score(y_test, lr_pred, zero_division=0)
lr_f1   = f1_score(y_test, lr_pred, zero_division=0)
lr_roc  = roc_auc_score(y_test, lr_probs)
lr_pr   = average_precision_score(y_test, lr_probs)

print("\nLogistic Regression — final on TEST")
print(f"  Accuracy : {lr_acc*100:.2f}%")
print(f"  Precision: {lr_prec*100:.2f}%")
print(f"  Recall   : {lr_rec*100:.2f}%")
print(f"  F1 Score : {lr_f1*100:.2f}%")
print(f"  ROC-AUC  : {lr_roc:.4f}")
print(f"  PR-AUC   : {lr_pr:.4f}")
print("  Classification report:")
print(classification_report(y_test, lr_pred, target_names=["Real", "Fake"]))


# Train XGBoost (threshold tuned on validation)
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    scale_pos_weight=imbalance_ratio,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

xgb_probs_val = xgb_model.predict_proba(X_val)[:, 1]
xgb_prec_vals, xgb_rec_vals, xgb_thresholds = precision_recall_curve(y_val, xgb_probs_val)
xgb_f1_vals = 2 * xgb_prec_vals * xgb_rec_vals / (xgb_prec_vals + xgb_rec_vals + 1e-8)
xgb_best_idx = xgb_f1_vals[:-1].argmax()
xgb_best_threshold = xgb_thresholds[xgb_best_idx]
print(f"  XGB best threshold (max F1 on validation): {xgb_best_threshold:.3f}")

# Final predictions on TEST
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_probs >= xgb_best_threshold).astype(int)

xgb_acc  = accuracy_score(y_test, xgb_pred)
xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
xgb_rec  = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1   = f1_score(y_test, xgb_pred, zero_division=0)
xgb_roc  = roc_auc_score(y_test, xgb_probs)
xgb_pr   = average_precision_score(y_test, xgb_probs)

print("\nXGBoost — final on TEST")
print(f"  Accuracy : {xgb_acc*100:.2f}%")
print(f"  Precision: {xgb_prec*100:.2f}%")
print(f"  Recall   : {xgb_rec*100:.2f}%")
print(f"  F1 Score : {xgb_f1*100:.2f}%")
print(f"  ROC-AUC  : {xgb_roc:.4f}")
print(f"  PR-AUC   : {xgb_pr:.4f}")
print("  Classification report:")
print(classification_report(y_test, xgb_pred, target_names=["Real", "Fake"]))


# STEP 10: Model Comparison
print("\nModel comparison (using TEST set PR-AUC as the tie-breaker)...")
print(f"  Logistic Regression: F1={lr_f1*100:.2f}%, Precision={lr_prec*100:.2f}%, Recall={lr_rec*100:.2f}%, PR-AUC={lr_pr:.4f}")
print(f"  XGBoost            : F1={xgb_f1*100:.2f}%, Precision={xgb_prec*100:.2f}%, Recall={xgb_rec*100:.2f}%, PR-AUC={xgb_pr:.4f}")

best_model_name = "xgboost" if xgb_pr > lr_pr else "logistic_regression"
print(f"  Recommended default model: {best_model_name.upper()}")


# STEP 11: Save Models + Vectorizer + Config
print("\nSaving artifacts (models, vectorizer, and thresholds)...")

joblib.dump(lr_model,   os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
joblib.dump(xgb_model,  os.path.join(MODELS_DIR, "xgboost_model.pkl"))
joblib.dump(tfidf,      os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

# Save thresholds and structural column names so the backend knows what to use
config = {
    "lr_threshold"      : round(float(best_threshold), 4),
    "xgb_threshold"     : round(float(xgb_best_threshold), 4),
    "best_model"        : best_model_name,
    "struct_cols"       : STRUCT_COLS,
    "n_tfidf_features"  : 5000
}
with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("  Saved: models/logistic_regression_model.pkl")
print("  Saved: models/xgboost_model.pkl")
print("  Saved: models/tfidf_vectorizer.pkl")
print("  Saved: models/model_config.json")


# STEP 12: Save Metrics JSON
print("\nSaving benchmark metrics for the dashboard...")

lr_cm  = confusion_matrix(y_test, lr_pred)
xgb_cm = confusion_matrix(y_test, xgb_pred)

metrics = {
    "xgboost": {
        "accuracy"        : round(float(xgb_acc),  4),
        "precision"       : round(float(xgb_prec), 4),
        "recall"          : round(float(xgb_rec),  4),
        "f1_score"        : round(float(xgb_f1),   4),
        "roc_auc"         : round(float(xgb_roc),  4),
        "pr_auc"          : round(float(xgb_pr),   4),
        "threshold"       : round(float(xgb_best_threshold), 4),
        "confusion_matrix": xgb_cm.tolist()
    },
    "logistic_regression": {
        "accuracy"        : round(float(lr_acc),  4),
        "precision"       : round(float(lr_prec), 4),
        "recall"          : round(float(lr_rec),  4),
        "f1_score"        : round(float(lr_f1),   4),
        "roc_auc"         : round(float(lr_roc),  4),
        "pr_auc"          : round(float(lr_pr),   4),
        "threshold"       : round(float(best_threshold), 4),
        "confusion_matrix": lr_cm.tolist()
    },
    "dataset": {
        "total_rows"   : int(df.shape[0]),
        "real_postings": int((df["fraudulent"] == 0).sum()),
        "fake_postings": int((df["fraudulent"] == 1).sum()),
        "fake_percent" : round(float(df["fraudulent"].mean() * 100), 2)
    },
    "best_model": best_model_name
}

metrics_path = os.path.join(STATIC_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: static/model_metrics.json")


# STEP 13: Suspicious Keywords (from LR coefficients)
print("\nExtracting top suspicious keywords (from LR coefficients)...")

feature_names = np.array(
    tfidf.get_feature_names_out().tolist() + STRUCT_COLS
)

# LR coefficients: positive = associated with FAKE (class=1)
lr_coefs = lr_model.coef_[0]
top_indices = lr_coefs.argsort()[-50:][::-1]
top_keywords = feature_names[top_indices].tolist()
top_scores   = lr_coefs[top_indices].tolist()

# Separate out structural vs text features for explainability
top_text_kw  = [(kw, sc) for kw, sc in zip(top_keywords, top_scores) if kw not in STRUCT_COLS][:30]
top_struct_kw = [(kw, sc) for kw, sc in zip(top_keywords, top_scores) if kw in STRUCT_COLS]

keywords_path = os.path.join(STATIC_DIR, "suspicious_keywords.json")
with open(keywords_path, "w") as f:
    json.dump({
        "keywords"           : [kw for kw, _ in top_text_kw],
        "keywords_with_scores": [{"word": kw, "score": round(sc, 4)} for kw, sc in top_text_kw],
        "structural_signals" : [{"feature": kw, "score": round(sc, 4)} for kw, sc in top_struct_kw]
    }, f, indent=2)

print(f"  Top 10 suspicious keywords : {[kw for kw, _ in top_text_kw[:10]]}")
print(f"  Top structural signals     : {[kw for kw, _ in top_struct_kw]}")
print(f"  Saved: static/suspicious_keywords.json")


# DONE
print("\nTraining complete.")
print(f"\n  Logistic Regression  — F1: {lr_f1*100:.2f}%  PR-AUC: {lr_pr:.4f}")
print(f"  XGBoost              — F1: {xgb_f1*100:.2f}%  PR-AUC: {xgb_pr:.4f}")
print(f"\n  Best model: {best_model_name.upper()}")
print("\n  Files saved:")
print("    models/logistic_regression_model.pkl")
print("    models/xgboost_model.pkl")
print("    models/tfidf_vectorizer.pkl")
print("    models/model_config.json")
print("    static/model_metrics.json")
print("    static/suspicious_keywords.json")
print("\n  --> Next: run evaluate.py to generate charts")
