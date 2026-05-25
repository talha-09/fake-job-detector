"""
metrics.py — Read-only endpoints for model metrics, history, and stats.

GET /api/metrics  → model performance numbers from model_metrics.json
GET /api/history  → last 100 predictions from SQLite
GET /api/stats    → aggregate counts
"""

import os
import json
from fastapi import APIRouter, HTTPException

from app.schemas  import MetricsResponse, StatsResponse, PredictionRecord
from app.database import get_all_predictions, get_prediction_stats

router = APIRouter()

# Path to metrics JSON 
_HERE         = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR   = os.path.normpath(os.path.join(_HERE, "..", "..", "static"))
_METRICS_PATH = os.path.join(_STATIC_DIR, "model_metrics.json")


# Endpoints
@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Returns accuracy, precision, recall, F1 and confusion matrix
    for both trained models (read from model_metrics.json).
    """
    if not os.path.exists(_METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="model_metrics.json not found. "
                   "Make sure you copied it from Colab to backend/static/."
        )

    with open(_METRICS_PATH, "r") as f:
        data = json.load(f)

    return MetricsResponse(**data)


@router.get("/history", response_model=list[PredictionRecord])
async def get_history():
    """
    Returns the 100 most recent predictions, newest first.
    """
    records = get_all_predictions(limit=100)
    return records


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Returns aggregate counts:  total, fake, real, fake_percentage.
    """
    return StatsResponse(**get_prediction_stats())
