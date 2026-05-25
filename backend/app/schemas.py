"""
schemas.py — Pydantic models for request / response validation.
FastAPI uses these automatically to validate incoming JSON and
serialise outgoing JSON.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# Request schemas
class PredictRequest(BaseModel):
    job_text: str = Field(
        ...,
        min_length=20,
        max_length=10_000,
        description="The raw job posting text to analyse.",
        examples=["Software Engineer needed at Acme Corp. Must know Python..."],
    )
    model_name: str = Field(
        default="logistic_regression",
        description="Which model to use: 'naive_bayes' or 'logistic_regression'.",
        pattern="^(naive_bayes|logistic_regression)$",
    )


# Response schemas
class PredictResponse(BaseModel):
    prediction:          str         # "Fake" | "Real"
    confidence:          float       #  0.0   – 1.0
    model_used:          str
    risk_level:          str         # "High" | "Medium" | "Low"
    suspicious_keywords: List[str]
    explanation:         str


class ModelMetrics(BaseModel):
    accuracy:          float
    precision:         float
    recall:            float
    f1_score:          float
    confusion_matrix:  List[List[int]]


class DatasetInfo(BaseModel):
    total_rows:    int
    real_postings: int
    fake_postings: int
    fake_percent:  float


class MetricsResponse(BaseModel):
    naive_bayes:          ModelMetrics
    logistic_regression:  ModelMetrics
    dataset:              DatasetInfo


class PredictionRecord(BaseModel):
    id:                  int
    job_text:            str
    prediction:          str
    confidence:          float
    model_used:          str
    risk_level:          str
    suspicious_keywords: List[str]
    explanation:         str
    created_at:          str


class StatsResponse(BaseModel):
    total_predictions: int
    fake_count:        int
    real_count:        int
    fake_percentage:   float
