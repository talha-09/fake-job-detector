"""
  Fake Job Posting Detection — Evaluation & Chart Generator
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

# Paths 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
DATA_PATH  = os.path.join(BASE_DIR, "data", "fake_job_postings.csv")

os.makedirs(STATIC_DIR, exist_ok=True)

# Style 
plt.rcParams.update({
    "figure.facecolor" : "#1e1e2e",
    "axes.facecolor"   : "#1e1e2e",
    "axes.edgecolor"   : "#444",
    "text.color"       : "#cdd6f4",
    "axes.labelcolor"  : "#cdd6f4",
    "xtick.color"      : "#cdd6f4",
    "ytick.color"      : "#cdd6f4",
    "grid.color"       : "#333",
    "axes.titlecolor"  : "#cdd6f4",
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 11,
})

INDIGO  = "#6366f1"
GREEN   = "#10b981"
RED     = "#ef4444"
AMBER   = "#f59e0b"
PURPLE  = "#a855f7"
CYAN    = "#06b6d4"

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# Helper: Rebuild test set (same random_state as training)
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def rebuild_test_set():
    """Reload data and reproduce exact test split used during training."""
    df = pd.read_csv(DATA_PATH)
    TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
    df["combined_text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
    df["cleaned_text"]  = df["combined_text"].apply(clean_text)

    tfidf  = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    X      = tfidf.transform(df["cleaned_text"])
    y      = df["fraudulent"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test, df


print("Loading models & rebuilding test set...")
nb_model = joblib.load(os.path.join(MODELS_DIR, "naive_bayes_model.pkl"))
lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
tfidf    = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

X_test, y_test, df = rebuild_test_set()
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Load saved metrics
with open(os.path.join(STATIC_DIR, "model_metrics.json")) as f:
    metrics = json.load(f)

nb_m = metrics["naive_bayes"]
lr_m = metrics["logistic_regression"]

print("Models loaded. Generating charts...\n")


# CHART 1: Confusion Matrix — Naive Bayes
print("  [1/6] Confusion Matrix — Naive Bayes")
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, nb_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Naive Bayes — Confusion Matrix", pad=14)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
# Style the text
for text in ax.texts:
    text.set_color("#1e1e2e")
    text.set_fontsize(14)
    text.set_fontweight("bold")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "nb_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()


# CHART 2: Confusion Matrix — Logistic Regression
print("  [2/6] Confusion Matrix — Logistic Regression")
fig, ax = plt.subplots(figsize=(6, 5))
cm2 = confusion_matrix(y_test, lr_pred)
disp2 = ConfusionMatrixDisplay(cm2, display_labels=["Real", "Fake"])
disp2.plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title("Logistic Regression — Confusion Matrix", pad=14)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
for text in ax.texts:
    text.set_color("#1e1e2e")
    text.set_fontsize(14)
    text.set_fontweight("bold")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "lr_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()


# CHART 3: Model Accuracy Comparison Bar Chart
print("  [3/6] Accuracy Comparison Chart")
fig, ax = plt.subplots(figsize=(7, 5))

models     = ["Naive Bayes", "Logistic\nRegression"]
accuracies = [nb_m["accuracy"] * 100, lr_m["accuracy"] * 100]
colors     = [INDIGO, GREEN]

bars = ax.bar(models, accuracies, color=colors, width=0.45, zorder=3, edgecolor="none")
ax.set_ylim(80, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison", pad=14)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

for bar, acc in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{acc:.1f}%",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", color="white"
    )

plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()


# CHART 4: Metrics Comparison (Precision, Recall, F1, Accuracy)
print("  [4/6] Metrics Comparison Chart")
fig, ax = plt.subplots(figsize=(9, 5))

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
nb_vals = [nb_m["accuracy"], nb_m["precision"], nb_m["recall"], nb_m["f1_score"]]
lr_vals = [lr_m["accuracy"], lr_m["precision"], lr_m["recall"], lr_m["f1_score"]]

x      = np.arange(len(metric_names))
width  = 0.35

b1 = ax.bar(x - width/2, [v*100 for v in nb_vals], width, label="Naive Bayes",          color=INDIGO, zorder=3)
b2 = ax.bar(x + width/2, [v*100 for v in lr_vals], width, label="Logistic Regression",   color=GREEN,  zorder=3)

ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Metrics", pad=14)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 110)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(facecolor="#2a2a3e", edgecolor="#444")

for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.1f}",
        ha="center", va="bottom", fontsize=9, color="white"
    )

plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()


# CHART 5: Class Distribution (Real vs Fake)
print("  [5/6] Class Distribution Chart")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

real_count = int((df["fraudulent"] == 0).sum())
fake_count = int((df["fraudulent"] == 1).sum())

# Pie chart
axes[0].pie(
    [real_count, fake_count],
    labels=["Real", "Fake"],
    colors=[GREEN, RED],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"color": "white", "fontsize": 12},
    wedgeprops={"edgecolor": "#1e1e2e", "linewidth": 2}
)
axes[0].set_title("Class Distribution", pad=14)

# Bar chart
axes[1].bar(["Real", "Fake"], [real_count, fake_count], color=[GREEN, RED], zorder=3)
axes[1].set_ylabel("Count")
axes[1].set_title("Real vs Fake Job Postings", pad=14)
axes[1].yaxis.grid(True, zorder=0)
axes[1].set_axisbelow(True)
for i, v in enumerate([real_count, fake_count]):
    axes[1].text(i, v + 100, f"{v:,}", ha="center", fontsize=12, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()


# CHART 6: Top 20 Suspicious Keywords
print("  [6/6] Suspicious Keywords Chart")

with open(os.path.join(STATIC_DIR, "suspicious_keywords.json")) as f:
    kw_data = json.load(f)

top_keywords = kw_data["keywords"][:20]
feature_names = np.array(tfidf.get_feature_names_out())
fake_log_prob = nb_model.feature_log_prob_[1]
real_log_prob = nb_model.feature_log_prob_[0]
diff = fake_log_prob - real_log_prob

# Get scores for the top keywords
kw_indices = [np.where(feature_names == kw)[0][0] for kw in top_keywords if kw in feature_names]
kw_scores  = diff[kw_indices]
kw_words   = feature_names[kw_indices]

# Sort by score
sorted_idx   = kw_scores.argsort()
kw_words     = kw_words[sorted_idx]
kw_scores    = kw_scores[sorted_idx]

fig, ax = plt.subplots(figsize=(8, 7))
colors_bar = [RED if s > 0 else GREEN for s in kw_scores]
bars = ax.barh(kw_words, kw_scores, color=colors_bar, zorder=3)
ax.set_xlabel("Log-Probability Difference (Fake - Real)")
ax.set_title("Top Suspicious Keywords\n(Higher = More Indicative of Fake Posting)", pad=14)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

red_patch   = mpatches.Patch(color=RED,   label="Spam Indicator")
green_patch = mpatches.Patch(color=GREEN, label="Legit Indicator")
ax.legend(handles=[red_patch, green_patch], facecolor="#2a2a3e", edgecolor="#444")

plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "suspicious_keywords.png"), dpi=150, bbox_inches="tight")
plt.close()



print()
print("=" * 60)
print("EVALUATION COMPLETE — All charts saved!")
print("=" * 60)
print("\n  Charts saved to backend/static/:")
charts = [
    "nb_confusion_matrix.png",
    "lr_confusion_matrix.png",
    "accuracy_comparison.png",
    "metrics_comparison.png",
    "class_distribution.png",
    "suspicious_keywords.png",
]
for c in charts:
    path = os.path.join(STATIC_DIR, c)
    exists = "YES" if os.path.exists(path) else "NO"
    print(f"    {exists} {c}")

print("\n  ➡  Phase 1 COMPLETE — Ready for Phase 2 (FastAPI backend)!")
