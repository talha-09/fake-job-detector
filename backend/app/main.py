"""
main.py — FastAPI application entry point.

Run with:
    cd backend
    .\\venv\\Scripts\\Activate
    uvicorn app.main:app --reload --port 8000

Swagger UI available at: http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database        import init_db
from app.routes.predict  import router as predict_router
from app.routes.metrics  import router as metrics_router

# Paths 
_HERE       = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.normpath(os.path.join(_HERE, "..", "static"))

os.makedirs(_STATIC_DIR, exist_ok=True)


# Lifespan — runs on startup & shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database on startup."""
    init_db()
    print("[main] FastAPI is ready. Visit http://localhost:8000/docs")
    yield
    print("[main] FastAPI shutting down.")


# App instance
app = FastAPI(
    title       = "Fake Job Posting Detection API",
    description = (
        "An explainable AI system that classifies job postings as Real or Fake "
        "using XGBoost and Logistic Regression models trained on the "
        "Kaggle Fake Job Postings dataset (17,880 records)."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


# CORS — allow React dev server and production URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # Vite dev server
        "http://localhost:3000",       # fallback
        "https://fake-job-detector.vercel.app",  # production (update after deploy)
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# Static files — serve PNGs and JSON from /static
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# Routers
app.include_router(predict_router, prefix="/api", tags=["Prediction"])
app.include_router(metrics_router, prefix="/api", tags=["Metrics"])


# Health check
@app.get("/", tags=["Health"])
async def root():
    return {
        "status":  "running",
        "message": "Fake Job Posting Detection API",
        "docs":    "http://localhost:8000/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
