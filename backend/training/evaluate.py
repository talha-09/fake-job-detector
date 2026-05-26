"""
  Fake Job Posting Detection — Evaluation & Chart Generator

  Generates 7 dark-themed charts comparing XGBoost and Logistic Regression:
    1. Side-by-side Confusion Matrices
    2. Precision-Recall Curves
    3. ROC Curves
    4. Metrics Bar Chart Comparison
    5. Top 20 Suspicious Keywords
    6. Dataset Distribution (donut)
    7. Structural Feature Signal Strengths
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
DATA_PATH  = os.path.join(BASE_DIR, "data", "fake_job_postings.csv")

os.makedirs(STATIC_DIR, exist_ok=True)

# NLTK
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# Dark theme palette
DARK_BG      = "#0D0D0D"
PANEL_BG     = "#1A1A1A"
GOLD         = "#F5A623"   # Logistic Regression
GREEN        = "#4CAF50"   # Real (class 0)
RED          = "#F44336"   # Fake (class 1)
TEAL         = "#00BCD4"   # XGBoost
WHITE        = "#E8E8E8"
GRID_COLOR   = "#2E2E2E"
MONO_FONT    = "DejaVu Sans Mono"

def apply_dark_style():
    """Apply dark theme globally to all subsequent matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor"       : DARK_BG,
        "axes.facecolor"         : PANEL_BG,
        "axes.edgecolor"         : GRID_COLOR,
        "axes.labelcolor"        : WHITE,
        "axes.titlecolor"        : WHITE,
        "axes.grid"              : True,
        "grid.color"             : GRID_COLOR,
        "grid.linewidth"         : 0.6,
        "xtick.color"            : WHITE,
        "ytick.color"            : WHITE,
        "xtick.labelsize"        : 9,
        "ytick.labelsize"        : 9,
        "text.color"             : WHITE,
        "font.family"            : "monospace",
        "font.monospace"         : [MONO_FONT, "Courier New", "monospace"],
        "legend.facecolor"       : PANEL_BG,
        "legend.edgecolor"       : GRID_COLOR,
        "legend.labelcolor"      : WHITE,
        "figure.titlesize"       : 13,
        "axes.titlesize"         : 11,
        "axes.labelsize"         : 10,
    })

apply_dark_style()


# Structural feature columns
STRUCT_COLS = [
    "has_salary", "has_logo", "has_questions", "telecommuting",
    "title_len", "desc_len", "profile_len", "short_desc", "no_profile"
]


# Text cleaning (must be same as in train_model.py)
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

    # Build structural features
    df["has_salary"]    = df["salary_range"].notna().astype(int)
    df["has_logo"]      = df["has_company_logo"].astype(int)
    df["has_questions"]  = df["has_questions"].notna().astype(int)
    df["telecommuting"] = df["telecommuting"].fillna(0).astype(int)
    df["title_len"]     = df["title"].fillna("").apply(len)
    df["desc_len"]      = df["description"].fillna("").apply(len)
    df["profile_len"]   = df["company_profile"].fillna("").apply(len)
    df["short_desc"]    = (df["desc_len"] < 200).astype(int)
    df["no_profile"]    = (df["profile_len"] == 0).astype(int)

    # Vectorize text
    tfidf  = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    X_tfidf = tfidf.transform(df["cleaned_text"])

    # Combine TF-IDF + structural
    struct_features = df[STRUCT_COLS].values
    X_struct_sparse = sp.csr_matrix(struct_features)
    X = sp.hstack([X_tfidf, X_struct_sparse])
    y = df["fraudulent"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test, df


# Load models & data
print("Loading models and preparing the test set...")
lr_model  = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
tfidf     = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

# Load config for thresholds
with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
    config = json.load(f)

lr_threshold  = config["lr_threshold"]
xgb_threshold = config["xgb_threshold"]

X_test, y_test, df = rebuild_test_set()

# Get probabilities
lr_probs  = lr_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# Apply thresholds
lr_pred  = (lr_probs >= lr_threshold).astype(int)
xgb_pred = (xgb_probs >= xgb_threshold).astype(int)

# Compute metrics for display
lr_acc   = float((lr_pred == y_test).mean())
xgb_acc  = float((xgb_pred == y_test).mean())
lr_roc   = roc_auc_score(y_test, lr_probs)
xgb_roc  = roc_auc_score(y_test, xgb_probs)
lr_pr    = average_precision_score(y_test, lr_probs)
xgb_pr   = average_precision_score(y_test, xgb_probs)

from sklearn.metrics import precision_score, recall_score, f1_score
lr_prec_val  = precision_score(y_test, lr_pred, zero_division=0)
lr_rec_val   = recall_score(y_test, lr_pred, zero_division=0)
lr_f1_val    = f1_score(y_test, lr_pred, zero_division=0)
xgb_prec_val = precision_score(y_test, xgb_pred, zero_division=0)
xgb_rec_val  = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1_val   = f1_score(y_test, xgb_pred, zero_division=0)

# Load saved metrics
with open(os.path.join(STATIC_DIR, "model_metrics.json")) as f:
    metrics = json.load(f)

print("Models are loaded. Generating charts...\n")


# CHART 1: Side-by-side Confusion Matrices
print("Making confusion matrices (LR vs XGBoost)...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Confusion Matrices — Fake Job Detection", fontsize=13, color=WHITE, y=1.02)

for ax, cm_data, title in zip(
    axes,
    [confusion_matrix(y_test, lr_pred), confusion_matrix(y_test, xgb_pred)],
    ["Logistic Regression", "XGBoost"],
):
    cm = np.array(cm_data)
    im = ax.imshow(cm, interpolation="nearest", cmap="YlOrRd", aspect="auto")
    im.set_clim(0, cm.max())

    labels = [["TN", "FP"], ["FN", "TP"]]
    cell_colors = [[GREEN, "#B71C1C"], ["#1B5E20", RED]]

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = count / cm.sum() * 100
            ax.text(
                j, i,
                f"{labels[i][j]}\n{count:,}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=cell_colors[i][j],
                fontfamily="monospace",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Real", "Predicted Fake"], color=WHITE)
    ax.set_yticklabels(["Actual Real", "Actual Fake"], color=WHITE)
    ax.set_title(title, color=WHITE, pad=10)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

plt.tight_layout()
cm_path = os.path.join(STATIC_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"    Saved: {cm_path}")


# CHART 2: Precision-Recall Curves
print("Plotting precision-recall curves...")
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Precision-Recall Curves", fontsize=13, color=WHITE)

for probs, label, color, threshold in [
    (lr_probs,  f"Logistic Regression (PR-AUC = {lr_pr:.3f})",  GOLD, lr_threshold),
    (xgb_probs, f"XGBoost             (PR-AUC = {xgb_pr:.3f})", TEAL, xgb_threshold),
]:
    precision_c, recall_c, thresholds_c = precision_recall_curve(y_test, probs)
    ax.plot(recall_c, precision_c, color=color, lw=2, label=label)

    idx = np.argmin(np.abs(thresholds_c - threshold))
    ax.scatter(
        recall_c[idx], precision_c[idx],
        color=color, s=100, zorder=5,
        edgecolors=WHITE, linewidths=0.8,
        label=f"  threshold = {threshold:.3f}",
    )

baseline = (y_test == 1).mean()
ax.axhline(baseline, color=RED, linestyle="--", lw=1.2,
           label=f"Random baseline ({baseline:.3f})")

ax.set_xlabel("Recall", color=WHITE)
ax.set_ylabel("Precision", color=WHITE)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.legend(loc="upper right", fontsize=8)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)

plt.tight_layout()
pr_path = os.path.join(STATIC_DIR, "pr_curves.png")
plt.savefig(pr_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"    Saved: {pr_path}")


# CHART 3: ROC Curves
print("Plotting ROC curves...")
fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("ROC Curves", fontsize=13, color=WHITE)

for probs, label, color in [
    (lr_probs,  f"Logistic Regression (AUC = {lr_roc:.3f})",  GOLD),
    (xgb_probs, f"XGBoost             (AUC = {xgb_roc:.3f})", TEAL),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=label)

ax.plot([0, 1], [0, 1], color=RED, linestyle="--", lw=1.2, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", color=WHITE)
ax.set_ylabel("True Positive Rate", color=WHITE)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.legend(loc="lower right", fontsize=9)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)

plt.tight_layout()
roc_path = os.path.join(STATIC_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"    Saved: {roc_path}")


# CHART 4: Metrics Bar Chart — side-by-side comparison
print("Building the metrics comparison bar chart...")
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
lr_vals  = [lr_acc,      lr_prec_val,  lr_rec_val,  lr_f1_val,  lr_roc,  lr_pr]
xgb_vals = [xgb_acc,     xgb_prec_val, xgb_rec_val, xgb_f1_val, xgb_roc, xgb_pr]

x      = np.arange(len(metric_labels))
width  = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Model Performance Comparison", fontsize=13, color=WHITE)

bars_lr  = ax.bar(x - width/2, lr_vals,  width, label="Logistic Regression", color=GOLD,  alpha=0.88)
bars_xgb = ax.bar(x + width/2, xgb_vals, width, label="XGBoost",             color=TEAL,  alpha=0.88)

for bar in bars_lr:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8,
            color=GOLD, fontfamily="monospace")

for bar in bars_xgb:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8,
            color=TEAL, fontfamily="monospace")

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, color=WHITE)
ax.set_ylim([0, 1.13])
ax.set_ylabel("Score", color=WHITE)
ax.legend(fontsize=9)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)

plt.tight_layout()
bar_path = os.path.join(STATIC_DIR, "metrics_comparison.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"    Saved: {bar_path}")


# CHART 5: Top 20 Suspicious Keywords (from LR coefficients)
print("Extracting the top suspicious keywords...")

with open(os.path.join(STATIC_DIR, "suspicious_keywords.json"), "r") as f:
    kw_data = json.load(f)

kw_scores = kw_data["keywords_with_scores"][:20]
words     = [item["word"]  for item in reversed(kw_scores)]
scores    = [item["score"] for item in reversed(kw_scores)]

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Top 20 Suspicious Keywords\n(Logistic Regression Coefficients → Fake)", fontsize=11, color=WHITE)

colors = [RED if s > 0.5 else GOLD for s in scores]
bars   = ax.barh(words, scores, color=colors, alpha=0.85, height=0.65)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
        f"{score:.3f}", va="center", ha="left",
        fontsize=8, color=WHITE, fontfamily="monospace",
    )

ax.set_xlabel("Coefficient (higher → stronger fake signal)", color=WHITE)
ax.set_facecolor(PANEL_BG)
ax.set_xlim([0, max(scores) * 1.18])
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)

plt.tight_layout()
kw_path = os.path.join(STATIC_DIR, "suspicious_keywords.png")
plt.savefig(kw_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"    Saved: {kw_path}")


# CHART 6: Dataset Distribution (donut chart)
print("Creating the dataset distribution chart...")

real_count = int((df["fraudulent"] == 0).sum())
fake_count = int((df["fraudulent"] == 1).sum())

fig, ax = plt.subplots(figsize=(6, 6))
fig.suptitle("Dataset Class Distribution", fontsize=13, color=WHITE)

wedges, texts, autotexts = ax.pie(
    [real_count, fake_count],
    labels=["Real Jobs", "Fake Jobs"],
    colors=[GREEN, RED],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"width": 0.5, "edgecolor": DARK_BG, "linewidth": 2},
    pctdistance=0.75,
)
for t in texts:
    t.set_color(WHITE)
    t.set_fontfamily("monospace")
    t.set_fontsize(10)
for at in autotexts:
    at.set_color(DARK_BG)
    at.set_fontfamily("monospace")
    at.set_fontweight("bold")
    at.set_fontsize(9)

ax.text(0, 0, f"{df.shape[0]:,}\nTotal", ha="center", va="center",
        fontsize=11, color=WHITE, fontfamily="monospace", fontweight="bold")

ax.set_facecolor(DARK_BG)
fig.set_facecolor(DARK_BG)

plt.tight_layout()
dist_path = os.path.join(STATIC_DIR, "dataset_distribution.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"    Saved: {dist_path}")


# CHART 7: Structural Feature Signal Strengths
print("Summarizing structural feature signals...")

struct_signals = kw_data.get("structural_signals", [])

if struct_signals:
    s_words  = [item["feature"] for item in reversed(struct_signals)]
    s_scores = [item["score"]   for item in reversed(struct_signals)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(s_words) * 0.55)))
    fig.suptitle("Structural Feature Signal Strengths\n(Logistic Regression Coefficients)", fontsize=11, color=WHITE)

    s_colors = [RED if s > 0 else GREEN for s in s_scores]
    s_bars   = ax.barh(s_words, s_scores, color=s_colors, alpha=0.85, height=0.6)

    ax.axvline(0, color=WHITE, linewidth=0.8, linestyle="--")

    for bar, score in zip(s_bars, s_scores):
        offset = 0.005 if score >= 0 else -0.005
        ha     = "left" if score >= 0 else "right"
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
            f"{score:+.3f}", va="center", ha=ha,
            fontsize=8, color=WHITE, fontfamily="monospace",
        )

    ax.set_xlabel("Coefficient  (positive → fake signal,  negative → real signal)", color=WHITE)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    struct_path = os.path.join(STATIC_DIR, "structural_signals.png")
    plt.savefig(struct_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    Saved: {struct_path}")
else:
    print("    No structural signals found — skipping Chart 7.")


# SUMMARY
print("\nCharts generation finished.")
charts = [
    "confusion_matrices.png",
    "pr_curves.png",
    "roc_curves.png",
    "metrics_comparison.png",
    "suspicious_keywords.png",
    "dataset_distribution.png",
    "structural_signals.png",
]
print("\nSaved files in backend/static/:")
for c in charts:
    path = os.path.join(STATIC_DIR, c)
    exists = "YES" if os.path.exists(path) else "NO"
    print(f"    {exists} {c}")

print("\nCharts are ready for the frontend dashboard.")
