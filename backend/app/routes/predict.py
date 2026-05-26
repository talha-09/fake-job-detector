"""
predict.py — POST /api/predict

Loads the trained .pkl models ONCE at module import time.
Every user request just calls model.predict_proba() — no retraining ever happens.

Models:
  - XGBoost (primary)           — gradient boosted trees, highest precision
  - Logistic Regression (secondary) — SMOTE-trained, higher recall
"""

import os
import re
import json
import joblib
import numpy as np
import scipy.sparse as sp
from fastapi import APIRouter, HTTPException

from app.schemas   import PredictRequest, PredictResponse
from app.database  import save_prediction

router = APIRouter()

# Resolve paths relative to THIS file
_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "models"))
_STATIC_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "static"))

# Load artefacts at startup (ONCE)
_MODELS: dict = {}
_TFIDF        = None
_KEYWORDS: list = []
_CONFIG: dict = {}

# Structural feature column names (must match training)
_STRUCT_COLS = [
    "has_salary", "has_logo", "has_questions", "telecommuting",
    "title_len", "desc_len", "profile_len", "short_desc", "no_profile"
]

def _load_artefacts():
    global _TFIDF, _KEYWORDS, _CONFIG

    xgb_path   = os.path.join(_MODELS_DIR, "xgboost_model.pkl")
    lr_path    = os.path.join(_MODELS_DIR, "logistic_regression_model.pkl")
    tfidf_path = os.path.join(_MODELS_DIR, "tfidf_vectorizer.pkl")
    config_path = os.path.join(_MODELS_DIR, "model_config.json")
    kw_path    = os.path.join(_STATIC_DIR,  "suspicious_keywords.json")

    missing = [p for p in [xgb_path, lr_path, tfidf_path] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"[predict] Missing model files: {missing}\n"
            "Please copy the .pkl files from Colab into backend/models/ first."
        )

    _MODELS["xgboost"]              = joblib.load(xgb_path)
    _MODELS["logistic_regression"]  = joblib.load(lr_path)
    _TFIDF                          = joblib.load(tfidf_path)

    # Load model config (thresholds, structural column names)
    if os.path.exists(config_path):
        with open(config_path) as f:
            _CONFIG.update(json.load(f))
        print(f"[predict] Config loaded — XGB threshold: {_CONFIG.get('xgb_threshold')}, "
              f"LR threshold: {_CONFIG.get('lr_threshold')}")

    if os.path.exists(kw_path):
        with open(kw_path) as f:
            _KEYWORDS = json.load(f).get("keywords", [])

    print(f"[predict] Models loaded from {_MODELS_DIR}")

_load_artefacts()


# Text cleaning — must mirror train_model.py exactly
try:
    from nltk.corpus import stopwords as _sw
    import nltk as _nltk
    _nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(_sw.words("english"))
except Exception:
    _STOP_WORDS = set()


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)        # strip HTML
    text = re.sub(r"[^a-z\s]", " ", text)        # letters only
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in _STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def _build_structural_features(raw_text: str, cleaned_text: str) -> np.ndarray:
    """
    Build the 9 structural features from raw job text.

    Note: has_salary, has_logo, has_questions, telecommuting are CSV metadata
    that don't exist in raw text input — we set them to 0.
    The text-derived features (title_len, desc_len, etc.) ARE computable.
    """
    text_len = len(raw_text)
    # Approximate: we don't have separate title/description/profile fields
    # from raw text, so we use the full text length as desc_len and set
    # profile_len to 0 (unknown) which will trigger no_profile=1
    return np.array([[
        0,                          # has_salary — unknown from raw text
        0,                          # has_logo — unknown from raw text
        0,                          # has_questions — unknown from raw text
        0,                          # telecommuting — unknown from raw text
        min(text_len, 200),         # title_len — approximate with capped text length
        text_len,                   # desc_len — use full text length
        0,                          # profile_len — unknown from raw text
        int(text_len < 200),        # short_desc — flag if text is very short
        1,                          # no_profile — no company profile available
    ]], dtype=float)


# Helpers
def _get_risk_level(confidence: float, prediction: str) -> str:
    if prediction == "Fake":
        if confidence >= 0.80:
            return "High"
        elif confidence >= 0.60:
            return "Medium"
        else:
            return "Low"
    return "Low"


def _extract_suspicious_keywords(cleaned_text: str, top_n: int = 5) -> list[str]:
    """Return global suspicious keywords that appear in this job posting."""
    found = [kw for kw in _KEYWORDS if kw in cleaned_text]
    return found[:top_n]


def _build_explanation(
    prediction: str, confidence: float, risk_level: str, keywords: list
) -> str:
    conf_pct = round(confidence * 100, 1)
    if prediction == "Fake":
        kw_str = ", ".join(f'"{k}"' for k in keywords) if keywords else "generic patterns"
        return (
            f"The model is {conf_pct}% confident this is a FAKE job posting "
            f"(Risk: {risk_level}). "
            f"Suspicious signals detected: {kw_str}. "
            "Common red flags include unrealistic pay, vague descriptions, "
            "and requests for personal/financial information."
        )
    return (
        f"The model is {conf_pct}% confident this is a LEGITIMATE job posting. "
        "The text contains typical patterns of genuine employment advertisements."
    )


# Endpoint
@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """
    Classify a job posting as Real or Fake.

    - **job_text**: The full text of the job posting.
    - **model_name**: `"xgboost"` (default) or `"logistic_regression"`.
    """
    model = _MODELS.get(body.model_name)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model_name}'. "
                   "Choose 'xgboost' or 'logistic_regression'."
        )

    # 1. Clean text
    cleaned = _clean_text(body.job_text)
    if not cleaned.strip():
        raise HTTPException(status_code=422, detail="Job text is too short or contains no meaningful words.")

    # 2. Vectorise text with TF-IDF
    X_tfidf = _TFIDF.transform([cleaned])

    # 3. Build structural features and combine with TF-IDF
    struct = _build_structural_features(body.job_text, cleaned)
    X_struct_sparse = sp.csr_matrix(struct)
    X = sp.hstack([X_tfidf, X_struct_sparse])

    # 4. Predict using probability + threshold
    proba = model.predict_proba(X)[0]          # [P(Real), P(Fake)]
    fake_prob = float(proba[1])

    # Use tuned threshold from config if available
    if body.model_name == "xgboost":
        threshold = _CONFIG.get("xgb_threshold", 0.5)
    else:
        threshold = _CONFIG.get("lr_threshold", 0.5)

    label = 1 if fake_prob >= threshold else 0
    confidence = fake_prob if label == 1 else (1.0 - fake_prob)

    prediction = "Fake" if label == 1 else "Real"
    risk_level = _get_risk_level(confidence, prediction)
    keywords   = _extract_suspicious_keywords(cleaned)
    explanation = _build_explanation(prediction, confidence, risk_level, keywords)

    # 5. Persist to SQLite
    save_prediction(
        job_text            = body.job_text,
        prediction          = prediction,
        confidence          = confidence,
        model_used          = body.model_name,
        risk_level          = risk_level,
        suspicious_keywords = keywords,
        explanation         = explanation,
    )

    return PredictResponse(
        prediction          = prediction,
        confidence          = confidence,
        model_used          = body.model_name,
        risk_level          = risk_level,
        suspicious_keywords = keywords,
        explanation         = explanation,
    )
