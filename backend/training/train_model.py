"""
  Fake Job Posting Detection — Model Training Script
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

#  Paths 
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "fake_job_postings.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
STATIC_DIR  = os.path.join(BASE_DIR, "..", "static")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

#  NLTK 
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# STEP 1: Load & Inspect Data
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Rows     : {df.shape[0]:,}")
print(f"  Columns  : {df.shape[1]}")
print(f"  Real jobs: {(df['fraudulent'] == 0).sum():,}")
print(f"  Fake jobs: {(df['fraudulent'] == 1).sum():,}  ({df['fraudulent'].mean()*100:.1f}%)")


# STEP 2: Combine Text Columns
print("\n" + "=" * 60)
print("STEP 2: Combining Text Columns")
print("=" * 60)

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
df["combined_text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
print(f"  Combined columns: {TEXT_COLS}")
print(f"  Sample (first 200 chars): {df['combined_text'].iloc[0][:200]}")


# STEP 3: Clean Text
print("\n" + "=" * 60)
print("STEP 3: Cleaning Text")
print("=" * 60)

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
print(f"  Sample cleaned: {df['cleaned_text'].iloc[0][:200]}")


# STEP 4: TF-IDF Vectorization
print("\n" + "=" * 60)
print("STEP 4: TF-IDF Vectorization")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,    # apply log normalization
    min_df=2              # ignore terms appearing in fewer than 2 docs
)

X = tfidf.fit_transform(df["cleaned_text"])
y = df["fraudulent"].values
print(f"  Feature matrix shape: {X.shape}")
print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")


# STEP 5: Train / Test Split
print("\n" + "=" * 60)
print("STEP 5: Train / Test Split (80 / 20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep class ratio in both splits
)
print(f"  Train samples: {X_train.shape[0]:,}")
print(f"  Test  samples: {X_test.shape[0]:,}")
print(f"  Train fake   : {y_train.sum()}")
print(f"  Test  fake   : {y_test.sum()}")


# STEP 6: Handle Class Imbalance with SMOTE
print("\n" + "=" * 60)
print("STEP 6: Applying SMOTE to Balance Training Set")
print("=" * 60)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  Before SMOTE — Real: {(y_train == 0).sum()}, Fake: {(y_train == 1).sum()}")
print(f"  After  SMOTE — Real: {(y_train_bal == 0).sum()}, Fake: {(y_train_bal == 1).sum()}")


# STEP 7: Train Naive Bayes
print("\n" + "=" * 60)
print("STEP 7: Training Naive Bayes")
print("=" * 60)

nb_model = MultinomialNB(alpha=0.5)
nb_model.fit(X_train_bal, y_train_bal)
nb_pred = nb_model.predict(X_test)

nb_acc  = accuracy_score(y_test, nb_pred)
nb_prec = precision_score(y_test, nb_pred, zero_division=0)
nb_rec  = recall_score(y_test, nb_pred, zero_division=0)
nb_f1   = f1_score(y_test, nb_pred, zero_division=0)

print(f"  Accuracy : {nb_acc*100:.2f}%")
print(f"  Precision: {nb_prec*100:.2f}%")
print(f"  Recall   : {nb_rec*100:.2f}%")
print(f"  F1 Score : {nb_f1*100:.2f}%")
print("\n  Full Classification Report:")
print(classification_report(y_test, nb_pred, target_names=["Real", "Fake"]))


# STEP 8: Train Logistic Regression
print("\n" + "=" * 60)
print("STEP 8: Training Logistic Regression")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
lr_model.fit(X_train_bal, y_train_bal)
lr_pred = lr_model.predict(X_test)

lr_acc  = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, zero_division=0)
lr_rec  = recall_score(y_test, lr_pred, zero_division=0)
lr_f1   = f1_score(y_test, lr_pred, zero_division=0)

print(f"  Accuracy : {lr_acc*100:.2f}%")
print(f"  Precision: {lr_prec*100:.2f}%")
print(f"  Recall   : {lr_rec*100:.2f}%")
print(f"  F1 Score : {lr_f1*100:.2f}%")
print("\n  Full Classification Report:")
print(classification_report(y_test, lr_pred, target_names=["Real", "Fake"]))


# STEP 9: Save Models + Vectorizer
print("\n" + "=" * 60)
print("STEP 9: Saving Models & Vectorizer")
print("=" * 60)

joblib.dump(nb_model,  os.path.join(MODELS_DIR, "naive_bayes_model.pkl"))
joblib.dump(lr_model,  os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
joblib.dump(tfidf,     os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
print("  Saved: models/naive_bayes_model.pkl")
print("  Saved: models/logistic_regression_model.pkl")
print("  Saved: models/tfidf_vectorizer.pkl")


# STEP 10: Save Metrics to JSON (used by backend /api/metrics)
print("\n" + "=" * 60)
print("STEP 10: Saving Metrics JSON")
print("=" * 60)

# Confusion matrix values
nb_cm = confusion_matrix(y_test, nb_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

metrics = {
    "naive_bayes": {
        "accuracy" : round(float(nb_acc),  4),
        "precision": round(float(nb_prec), 4),
        "recall"   : round(float(nb_rec),  4),
        "f1_score" : round(float(nb_f1),   4),
        "confusion_matrix": nb_cm.tolist()
    },
    "logistic_regression": {
        "accuracy" : round(float(lr_acc),  4),
        "precision": round(float(lr_prec), 4),
        "recall"   : round(float(lr_rec),  4),
        "f1_score" : round(float(lr_f1),   4),
        "confusion_matrix": lr_cm.tolist()
    },
    "dataset": {
        "total_rows"   : int(df.shape[0]),
        "real_postings": int((df["fraudulent"] == 0).sum()),
        "fake_postings": int((df["fraudulent"] == 1).sum()),
        "fake_percent" : round(float(df["fraudulent"].mean() * 100), 2)
    }
}

metrics_path = os.path.join(STATIC_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: static/model_metrics.json")


# STEP 11: Save top suspicious keywords (for explainability)
print("\n" + "=" * 60)
print("STEP 11: Extracting Top Suspicious Keywords")
print("=" * 60)

feature_names = np.array(tfidf.get_feature_names_out())

# From Naive Bayes: log prob of each word given class=1 (fake)
fake_log_prob = nb_model.feature_log_prob_[1]
real_log_prob = nb_model.feature_log_prob_[0]

# Difference = words most associated with FAKE vs REAL
diff = fake_log_prob - real_log_prob
top_indices = diff.argsort()[-50:][::-1]
top_keywords = feature_names[top_indices].tolist()

keywords_path = os.path.join(STATIC_DIR, "suspicious_keywords.json")
with open(keywords_path, "w") as f:
    json.dump({"keywords": top_keywords}, f, indent=2)

print(f"  Top 10 suspicious keywords: {top_keywords[:10]}")
print(f"  Saved: static/suspicious_keywords.json")


# COMPLETED 
print("\n" + "=" * 60)
print("[DONE] TRAINING COMPLETE - Phase 1 Step 9 done!")
print("=" * 60)
print(f"\n  Naive Bayes Accuracy      : {nb_acc*100:.2f}%")
print(f"  Logistic Regression Accuracy: {lr_acc*100:.2f}%")
print("\n  Files saved to:")
print("    backend/models/naive_bayes_model.pkl")
print("    backend/models/logistic_regression_model.pkl")
print("    backend/models/tfidf_vectorizer.pkl")
print("    backend/static/model_metrics.json")
print("    backend/static/suspicious_keywords.json")
print("  --> Next: run evaluate.py to generate charts")
