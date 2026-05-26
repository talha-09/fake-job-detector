# JobGuard AI — Explainable Fake Job Posting Detection

JobGuard AI is a full-stack machine learning application that classifies job postings as legitimate or fraudulent. The system combines a React single-page application with a FastAPI backend, trained classifiers, and a local persistence layer so users can analyze postings, review model benchmarks, and audit past predictions.

The models are trained on the public [Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) dataset (17,880 labeled records). The minority class (fraudulent postings) represents approximately 4.8% of the corpus, which informs the training strategy and evaluation metrics documented below.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Repository Structure](#repository-structure)
6. [Prerequisites](#prerequisites)
7. [Installation](#installation)
8. [Model Training](#model-training)
9. [Running the Application](#running-the-application)
10. [API Reference](#api-reference)
11. [Configuration](#configuration)
12. [Frontend Application](#frontend-application)
13. [Machine Learning Pipeline](#machine-learning-pipeline)
14. [Model Performance](#model-performance)
15. [Testing](#testing)
16. [Limitations and Responsible Use](#limitations-and-responsible-use)

---

## Overview

Job seekers and recruiters increasingly encounter fraudulent job advertisements designed to harvest personal data, solicit payments, or route victims into scam workflows. This project addresses that problem with an explainable classification pipeline:

- **Input:** Raw job posting text (title, description, requirements, or full combined copy).
- **Processing:** Text normalization, stopword removal, and TF-IDF vectorization using the same preprocessing logic at training and inference time.
- **Inference:** One of two scikit-learn classifiers (Multinomial Naive Bayes or Logistic Regression) returns a label, confidence score, risk tier, matched suspicious keywords, and a human-readable explanation.
- **Persistence:** Each prediction is stored in a local SQLite database for history and aggregate statistics.

The default production-facing classifier is **Logistic Regression**, which achieves higher precision on the held-out test set. **Naive Bayes** remains available for comparison and for lexical-probability-style explanations via its feature log-odds.

---

## Key Features


| Capability               | Description                                                                   |
| ------------------------ | ----------------------------------------------------------------------------- |
| Dual-model inference     | Switch between Naive Bayes and Logistic Regression at request time            |
| Explainable output       | Risk level, suspicious keyword hits, and narrative explanation per prediction |
| Class imbalance handling | SMOTE applied on the training split before model fitting                      |
| Model benchmarks         | Accuracy, precision, recall, F1, and confusion matrices exposed via REST      |
| Prediction history       | Last 100 scans retrieved from SQLite with timestamps and metadata             |
| Interactive dashboard    | Recharts visualizations and static evaluation assets served from the API      |
| OpenAPI documentation    | Auto-generated Swagger UI at `/docs` when the backend is running              |


---

## System Architecture

```mermaid
flowchart TB
    subgraph Client["React Frontend (Vite)"]
        Home[Home]
        Analyze[Analyze]
        Dashboard[Dashboard]
        History[History]
    end

    subgraph API["FastAPI Backend"]
        Predict["POST /api/predict"]
        Metrics["GET /api/metrics"]
        Stats["GET /api/stats"]
        Hist["GET /api/history"]
        Static["/static/*"]
    end

    subgraph ML["Inference Artefacts"]
        TFIDF[tfidf_vectorizer.pkl]
        NB[naive_bayes_model.pkl]
        LR[logistic_regression_model.pkl]
        KW[suspicious_keywords.json]
    end

    subgraph Store["Persistence"]
        SQLite[(email_detector.db)]
        MetricsJSON[model_metrics.json]
    end

    Analyze --> Predict
    Dashboard --> Metrics
    Dashboard --> Static
    History --> Hist
  Predict --> TFIDF
    Predict --> NB
    Predict --> LR
    Predict --> KW
    Predict --> SQLite
    Metrics --> MetricsJSON
```



**Request lifecycle for classification**

1. The client sends `job_text` and optional `model_name` to `POST /api/predict`.
2. The backend cleans the text using the same rules as `backend/training/train_model.py`.
3. The persisted TF-IDF vectorizer transforms the cleaned string into a sparse feature vector.
4. The selected model predicts class `0` (Real) or `1` (Fake) and returns class probabilities.
5. The service derives risk level, matches global suspicious keywords, builds an explanation string, inserts a row into SQLite, and returns JSON to the client.

Models are loaded once at application startup (module import time in `predict.py`), not retrained per request.

---

## Technology Stack

### Backend


| Component     | Version / Library                                                   |
| ------------- | ------------------------------------------------------------------- |
| Runtime       | Python 3.10+ recommended                                            |
| Web framework | FastAPI, Uvicorn                                                    |
| ML            | scikit-learn, imbalanced-learn (SMOTE), NLTK, joblib, NumPy, pandas |
| Persistence   | SQLite3 (stdlib)                                                    |
| Validation    | Pydantic v2 (via FastAPI)                                           |


### Frontend


| Component    | Version / Library |
| ------------ | ----------------- |
| UI framework | React 19          |
| Build tool   | Vite 8            |
| Routing      | React Router 7    |
| HTTP client  | Axios             |
| Charts       | Recharts          |
| Icons        | Lucide React      |


---

## Repository Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point, CORS, static mount
│   │   ├── database.py          # SQLite schema and CRUD helpers
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routes/
│   │       ├── predict.py       # POST /api/predict
│   │       └── metrics.py       # GET /api/metrics, /history, /stats
│   ├── models/                  # Trained .pkl files (generated; not always in VCS)
│   ├── static/
│   │   ├── model_metrics.json   # Evaluation metrics for dashboard
│   │   └── suspicious_keywords.json
│   ├── training/
│   │   ├── train_model.py       # End-to-end training script
│   │   ├── evaluate.py          # Confusion matrix and chart generation
│   │   └── data/
│   │       └── fake_job_postings.csv   # Kaggle dataset (user-provided)
│   ├── requirements.txt
│   └── test_api.py              # Manual integration smoke test
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Router and layout
│   │   ├── components/          # Navbar
│   │   ├── pages/               # Home, Analyze, Dashboard, History
│   │   └── services/api.js      # Axios client and API helpers
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## Prerequisites

- **Python** 3.10 or newer with `pip` and virtual environment support
- **Node.js** 18 or newer with `npm`
- **Kaggle dataset:** Download `fake_job_postings.csv` and place it at `backend/training/data/fake_job_postings.csv` before training
- **Trained artefacts:** The API requires three pickle files under `backend/models/` unless you run the training script locally

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Fake Email Project"
```

### 2. Backend setup

```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

On first run, NLTK stopwords are downloaded automatically by the training and prediction modules.

### 3. Frontend setup

```bash
cd frontend
npm install
```

---

## Model Training

Training is performed offline and is not triggered by the web server.

1. Obtain the dataset from Kaggle and save it as:
  ```
   backend/training/data/fake_job_postings.csv
  ```
2. Run the training script from the `backend` directory:
  ```bash
   cd backend
   python training/train_model.py
  ```
3. Optional: generate evaluation charts (confusion matrices, comparison plots):
  ```bash
   python training/evaluate.py
  ```

**Training script outputs**


| Output                    | Path                                           |
| ------------------------- | ---------------------------------------------- |
| Naive Bayes model         | `backend/models/naive_bayes_model.pkl`         |
| Logistic Regression model | `backend/models/logistic_regression_model.pkl` |
| TF-IDF vectorizer         | `backend/models/tfidf_vectorizer.pkl`          |
| Metrics JSON              | `backend/static/model_metrics.json`            |
| Suspicious keywords       | `backend/static/suspicious_keywords.json`      |


If model files are missing, the API will fail at startup with an explicit error directing you to complete training or copy artefacts into `backend/models/`.

---

## Running the Application

Start the backend and frontend in separate terminals.

**Terminal 1 — API server**

```bash
cd backend
.\venv\Scripts\activate          # Windows
uvicorn app.main:app --reload --port 8000
```

- API root: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

**Terminal 2 — Frontend dev server**

```bash
cd frontend
npm run dev
```

- Application UI: `http://localhost:5173` (default Vite port)

Ensure `VITE_API_URL` is unset during local development so the frontend targets `http://localhost:8000` (see [Configuration](#configuration)).

---

## API Reference

All JSON API routes are prefixed with `/api`. Static assets are served at `/static/{filename}`.

### Health


| Method | Path      | Description                             |
| ------ | --------- | --------------------------------------- |
| `GET`  | `/`       | Service metadata and documentation link |
| `GET`  | `/health` | Liveness probe (`{"status": "ok"}`)     |


### Prediction

#### `POST /api/predict`

Classifies a job posting and persists the result.

**Request body**

```json
{
  "job_text": "Full text of the job posting to analyze...",
  "model_name": "logistic_regression"
}
```


| Field        | Type   | Required | Constraints                                                             |
| ------------ | ------ | -------- | ----------------------------------------------------------------------- |
| `job_text`   | string | Yes      | 20–10,000 characters                                                    |
| `model_name` | string | No       | `naive_bayes` or `logistic_regression` (default: `logistic_regression`) |


**Response body**

```json
{
  "prediction": "Fake",
  "confidence": 0.94,
  "model_used": "logistic_regression",
  "risk_level": "High",
  "suspicious_keywords": ["home", "earn", "immediately"],
  "explanation": "The model is 94.0% confident this is a FAKE job posting ..."
}
```


| Field                 | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `prediction`          | `"Real"` or `"Fake"`                                            |
| `confidence`          | Probability assigned to the predicted class (0.0–1.0)           |
| `risk_level`          | `High`, `Medium`, or `Low` (elevated only for Fake predictions) |
| `suspicious_keywords` | Subset of globally ranked keywords present in the cleaned text  |
| `explanation`         | Human-readable summary of the verdict                           |


**Risk level rules (Fake predictions only)**


| Confidence | Risk level |
| ---------- | ---------- |
| ≥ 0.80     | High       |
| ≥ 0.60     | Medium     |
| < 0.60     | Low        |


Real predictions always return `Low` risk.

**Error responses**


| Status | Condition                              |
| ------ | -------------------------------------- |
| `400`  | Invalid `model_name`                   |
| `422`  | Text too short or empty after cleaning |
| `500`  | Missing model artefacts at startup     |


### Metrics and history

#### `GET /api/metrics`

Returns held-out test metrics and dataset statistics from `backend/static/model_metrics.json`.

#### `GET /api/history`

Returns up to 100 most recent prediction records from SQLite, newest first.

#### `GET /api/stats`

Returns aggregate counts: total predictions, fake count, real count, and fake percentage across all stored scans.

---

## Configuration

### Backend CORS

Allowed origins are defined in `backend/app/main.py`:

- `http://localhost:5173` (Vite dev server)
- `http://localhost:3000` (fallback)
- `https://fake-job-detector.vercel.app` (production frontend placeholder)

Add or update origins when deploying to new hosts.

### Frontend API base URL

Create `frontend/.env` for production builds:

```env
VITE_API_URL=https://your-backend-host.example.com
```

If unset, the client defaults to `http://localhost:8000`.

### Database location

SQLite database file: `backend/email_detector.db` (created automatically on first API startup). Database files are listed in `.gitignore` and are not committed to version control.

---

## Frontend Application


| Route        | Page      | Purpose                                                          |
| ------------ | --------- | ---------------------------------------------------------------- |
| `/`          | Home      | Product overview, pipeline description, and benchmark highlights |
| `/analyze`   | Analyze   | Submit job text, select model, view classification result        |
| `/dashboard` | Dashboard | Model metrics, dataset info, and evaluation visualizations       |
| `/history`   | History   | Browse prior predictions stored by the API                       |


The HTTP layer is centralized in `frontend/src/services/api.js`, which wraps Axios with a 15-second timeout and consistent `/api` prefix handling.

**Production build**

```bash
cd frontend
npm run build
npm run preview   # optional local preview of production bundle
```

---

## Machine Learning Pipeline

### Data preparation

1. **Load** `fake_job_postings.csv` (17,880 rows).
2. **Combine** text fields: `title`, `company_profile`, `description`, `requirements`, `benefits`.
3. **Clean** text: lowercase, strip HTML, retain alphabetic tokens, remove English stopwords, drop tokens with length ≤ 2.
4. **Vectorize** with `TfidfVectorizer`:
  - `max_features=5000`
  - `ngram_range=(1, 2)`
  - `sublinear_tf=True`
  - `min_df=2`
5. **Split** 80/20 stratified train/test (`random_state=42`).
6. **Balance** training features with SMOTE (`random_state=42`).
7. **Train** Multinomial Naive Bayes (`alpha=0.5`) and Logistic Regression (`max_iter=1000`, `C=1.0`, `solver=lbfgs`).
8. **Evaluate** on the untouched test set; serialize models, metrics, and top discriminative keywords.

### Explainability

- **Suspicious keywords:** The fifty TF-IDF features with the largest Naive Bayes log-probability gap between Fake and Real classes are stored in `suspicious_keywords.json`. At inference time, any keyword appearing in the cleaned posting is surfaced in the API response (up to five matches).
- **Narrative explanation:** A template string summarizes confidence, risk, and matched terms without exposing raw model coefficients to the client.

Inference preprocessing in `predict.py` must remain identical to `train_model.py`; any divergence will degrade accuracy.

---

## Model Performance

Metrics below are from the committed `model_metrics.json` (held-out 20% test split). Reported values may change if you retrain with a different random seed or dataset version.

### Dataset


| Metric              | Value  |
| ------------------- | ------ |
| Total records       | 17,880 |
| Legitimate postings | 17,014 |
| Fraudulent postings | 866    |
| Fraud rate          | 4.84%  |


### Naive Bayes


| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 92.31% |
| Precision | 37.86% |
| Recall    | 91.91% |
| F1 score  | 53.63% |


High recall with lower precision is typical when the model aggressively flags minority-class patterns.

### Logistic Regression (recommended default)


| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 98.04% |
| Precision | 75.37% |
| Recall    | 88.44% |
| F1 score  | 81.38% |


For operational use cases where false positives carry a high cost (e.g., auto-rejecting legitimate employers), prefer Logistic Regression and treat outputs as advisory rather than authoritative.

---

## Testing

### Manual API smoke test

With the backend running on port 8000:

```bash
cd backend
python test_api.py
```

This script exercises `/health`, `/api/metrics`, `/api/predict` (fake and real samples), `/api/stats`, and `/api/history`.

### Frontend lint

```bash
cd frontend
npm run lint
```

---

## Limitations and Responsible Use

This system is a **decision-support tool**, not a substitute for human judgment, background checks, or platform moderation policies.

- **Distribution shift:** Models reflect language patterns in the Kaggle corpus; novel scam formats may not be detected.
- **False positives and negatives:** Even at 98% accuracy, errors occur especially for ambiguous or unusually worded legitimate postings.
- **English-centric preprocessing:** Non-English postings may be poorly represented after cleaning.
- **Keyword explainability:** Surfaced terms indicate statistical association with the training label, not proven intent or illegality.
- **Privacy:** Submitted job text is stored locally (truncated to 2,000 characters per record). Do not submit personally identifiable information unless your deployment policy explicitly allows it.

Use predictions to prioritize review, not to make irreversible employment or legal decisions without additional verification.

---

## Acknowledgments

- Dataset: Shivam Bansal, *Real or Fake: Fake JobPosting Prediction* (Kaggle)
- Built with FastAPI, scikit-learn, React, and Vite

For questions, issues, or contributions, open an issue or pull request in the project repository.