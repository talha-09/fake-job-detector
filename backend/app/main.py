"""
main.py — FastAPI application entry point.

Run with:
    cd backend
    .\\venv\\Scripts\\Activate
    uvicorn app.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs  (disabled in production)
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
    print("[main] FastAPI is ready.")
    yield
    print("[main] FastAPI shutting down.")


# Environment — set ENVIRONMENT=production on Render to disable docs UI
_ENV = os.getenv("ENVIRONMENT", "development")
_IS_PROD = _ENV == "production"

# CORS — load allowed origins from env var (comma-separated) for security.
# In production set: ALLOWED_ORIGINS=https://your-app.vercel.app
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",  # dev fallback only
)
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]


# App instance — docs disabled in production to avoid exposing API surface
app = FastAPI(
    title       = "Fake Job Posting Detection API",
    description = (
        "An explainable AI system that classifies job postings as Real or Fake "
        "using XGBoost and Logistic Regression models trained on the "
        "Kaggle Fake Job Postings dataset (17,880 records)."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = None if _IS_PROD else "/docs",
    redoc_url   = None if _IS_PROD else "/redoc",
    openapi_url = None if _IS_PROD else "/openapi.json",
)


# CORS — allow React dev server and production URL

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _ALLOWED_ORIGINS,
    allow_credentials = False,          # No cookies/auth headers used — keep False
    allow_methods     = ["GET", "POST"], # Only what the API actually needs
    allow_headers     = ["Content-Type", "Accept"],
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
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
