"""
database.py — SQLite setup and helper functions.

Tables
------
predictions : stores every prediction made via /api/predict
"""

import sqlite3
import os
from datetime import datetime

# Path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "..", "email_detector.db")


# Connection helper
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row        # rows behave like dicts
    return conn


# Init — called once at FastAPI startup

def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                job_text            TEXT    NOT NULL,
                prediction          TEXT    NOT NULL,   -- 'Fake' or 'Real'
                confidence          REAL    NOT NULL,   -- 0.0 – 1.0
                model_used          TEXT    NOT NULL,   -- 'xgboost' | 'logistic_regression'
                risk_level          TEXT    NOT NULL,   -- 'High' | 'Medium' | 'Low'
                suspicious_keywords TEXT    NOT NULL,   -- JSON array stored as string
                explanation         TEXT    NOT NULL,
                created_at          TEXT    NOT NULL
            )
        """)
        conn.commit()
        print("[DB] Database initialised successfully.")
    finally:
        conn.close()


# Write
def save_prediction(
    job_text: str,
    prediction: str,
    confidence: float,
    model_used: str,
    risk_level: str,
    suspicious_keywords: list,
    explanation: str,
) -> int:
    """Insert a prediction record and return its new id."""
    import json as _json

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (job_text, prediction, confidence, model_used,
                 risk_level, suspicious_keywords, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_text[:2000],                      # cap at 2000 chars
                prediction,
                round(confidence, 4),
                model_used,
                risk_level,
                _json.dumps(suspicious_keywords),
                explanation,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()



# Read
 
def get_all_predictions(limit: int = 100) -> list[dict]:
    """Return the most recent `limit` predictions, newest first."""
    import json as _json

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["suspicious_keywords"] = _json.loads(d["suspicious_keywords"])
            result.append(d)
        return result
    finally:
        conn.close()


def get_prediction_stats() -> dict:
    """Return aggregate counts for the /api/stats endpoint."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        fake  = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 'Fake'"
        ).fetchone()[0]
        real  = total - fake
        return {
            "total_predictions": total,
            "fake_count":        fake,
            "real_count":        real,
            "fake_percentage":   round((fake / total * 100) if total else 0, 1),
        }
    finally:
        conn.close()
