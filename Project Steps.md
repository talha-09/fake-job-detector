# Fake Job Posting Detection — Complete Project Steps

> **Project**: Explainable AI-Based Fake Job Posting Detection System
> **Tech Stack**: React (Vite) + FastAPI + Naive Bayes + SQLite
> **Dataset**: Kaggle Fake Job Postings (17,880 records)
> **Timeline**: 22 Days

---

## PREREQUISITES — Install These First

Before starting, make sure you have these installed on your laptop:

| Software         | Download Link                     | Verify Command     |
| ---------------- | --------------------------------- | ------------------ |
| **Python 3.10+** | https://www.python.org/downloads/ | `python --version` |
| **Node.js 18+**  | https://nodejs.org/ (LTS version) | `node --version`   |
| **npm**          | Comes with Node.js                | `npm --version`    |
| **Git**          | https://git-scm.com/downloads     | `git --version`    |
| **VS Code**      | https://code.visualstudio.com/    | —                  |

> ⚠️ During Python installation, CHECK "Add Python to PATH" checkbox!

---

## PHASE 0: PROJECT SETUP (Day 1)

### Step 1: Create the Project Folder

Open PowerShell/Terminal and run:

```powershell
mkdir "D:\devVSCode\AI Project\Fake Email Project"
cd "D:\devVSCode\AI Project\Fake Email Project"
```

### Step 2: Create the Folder Structure

```powershell
# Backend folders
mkdir backend
mkdir backend\app
mkdir backend\app\routes
mkdir backend\models
mkdir backend\static
mkdir backend\training
mkdir backend\training\data

# Frontend will be created by Vite in Step 5
```

### Step 3: Create .gitignore

Create file `.gitignore` in the root folder with this content:

```
# Python
backend/venv/
__pycache__/
*.pyc
.env

# Node
frontend/node_modules/
frontend/dist/

# IDE
.vscode/
.idea/

# Database
*.db
```

### Step 4: Setup Backend (Python)

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project\backend"

# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate

# You should see (venv) at the start of your terminal prompt
```

Create file `backend/requirements.txt` with this content:

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.25.0
nltk>=3.8.1
joblib>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
imbalanced-learn>=0.11.0
python-multipart>=0.0.6
```

Install all packages:

```powershell
pip install -r requirements.txt
```

Download NLTK data:

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

### Step 5: Setup Frontend (React)

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project"

# Create React app with Vite
npx -y create-vite@latest frontend -- --template react

cd frontend

# Install dependencies
npm install

# Install additional packages we need
npm install axios react-router-dom recharts
```

### Step 6: Create All Empty Backend Files

Create these files (they can be empty for now, we fill them in later phases):

```
backend/app/__init__.py           ← empty file
backend/app/main.py               ← empty file
backend/app/database.py           ← empty file
backend/app/schemas.py            ← empty file
backend/app/routes/__init__.py    ← empty file
backend/app/routes/predict.py     ← empty file
backend/app/routes/metrics.py     ← empty file
backend/training/train_model.py   ← empty file
backend/training/evaluate.py      ← empty file
```

PowerShell command to create them all at once:

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project"

# Create empty Python files
New-Item -ItemType File -Path backend\app\__init__.py -Force
New-Item -ItemType File -Path backend\app\main.py -Force
New-Item -ItemType File -Path backend\app\database.py -Force
New-Item -ItemType File -Path backend\app\schemas.py -Force
New-Item -ItemType File -Path backend\app\routes\__init__.py -Force
New-Item -ItemType File -Path backend\app\routes\predict.py -Force
New-Item -ItemType File -Path backend\app\routes\metrics.py -Force
New-Item -ItemType File -Path backend\training\train_model.py -Force
New-Item -ItemType File -Path backend\training\evaluate.py -Force
```

### Step 7: Download the Dataset

1. Go to: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
2. Click "Download" (you need a free Kaggle account)
3. Extract the zip file
4. Copy `fake_job_postings.csv` to: `backend/training/data/fake_job_postings.csv`

### Step 8: Verify Everything Works

```powershell
# Test Python (Terminal 1)
cd "D:\devVSCode\AI Project\Fake Email Project\backend"
.\venv\Scripts\Activate
python -c "import fastapi; import sklearn; import pandas; print('Backend OK!')"

# Test Frontend (Terminal 2)
cd "D:\devVSCode\AI Project\Fake Email Project\frontend"
npm run dev
# Should open at http://localhost:5173 — you'll see the Vite + React default page
# Press Ctrl+C to stop
```

### Phase 0 Complete — Your Folder Should Look Like:

```
Fake Email Project/
├── .gitignore
├── backend/
│   ├── venv/
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── predict.py
│   │       └── metrics.py
│   ├── models/              (empty — .pkl files go here after training)
│   ├── static/              (empty — chart PNGs go here after evaluation)
│   └── training/
│       ├── train_model.py
│       ├── evaluate.py
│       └── data/
│           └── fake_job_postings.csv
│
├── frontend/
│   ├── node_modules/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
```

---

## PHASE 1: ML MODEL TRAINING (Day 2–5)

### Step 9: Write the Training Script

Open `backend/training/train_model.py` and write code that does:

1. **Load** `fake_job_postings.csv` with pandas
2. **Combine text columns** (`title`, `company_profile`, `description`, `requirements`, `benefits`) into one `combined_text` column
3. **Clean text**: lowercase, remove HTML tags, remove punctuation, remove stopwords
4. **TF-IDF Vectorize** the cleaned text (max 5000 features)
5. **Train/Test Split**: 80% train, 20% test (use `stratify=y`)
6. **Handle class imbalance** with SMOTE (only ~4.8% are fake)
7. **Train Model 1**: Multinomial Naive Bayes
8. **Train Model 2**: Logistic Regression (for comparison)
9. **Print accuracy** for both models
10. **Save** both models + TF-IDF vectorizer as `.pkl` files to `backend/models/`

Run it:

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project\backend\training"
..\venv\Scripts\Activate
python train_model.py
```

Expected output: NB accuracy ~90-95%, LR accuracy ~95-98%

### Step 10: Write the Evaluation Script

Open `backend/training/evaluate.py` and generate these visualizations:

1. **Confusion Matrix** (both models) → save as PNG
2. **Accuracy Comparison Bar Chart** (NB vs LR) → save as PNG
3. **Classification Report Heatmap** → save as PNG
4. **Class Distribution Chart** (real vs fake counts) → save as PNG
5. **Top 20 Suspicious Keywords** bar chart → save as PNG
6. **Save metrics to JSON** file (`backend/static/model_metrics.json`)

All PNGs saved to: `backend/static/`

Run it:

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project\backend\training"
python evaluate.py
```

### Phase 1 Complete — You Should Have:

```
backend/models/
├── naive_bayes_model.pkl
├── logistic_regression_model.pkl
└── tfidf_vectorizer.pkl

backend/static/
├── nb_confusion_matrix.png
├── lr_confusion_matrix.png
├── accuracy_comparison.png
├── class_distribution.png
├── top_suspicious_keywords.png
└── model_metrics.json
```

---

## PHASE 2: BACKEND — FASTAPI (Day 6–9)

### Step 11: Database Setup

Open `backend/app/database.py` and create:

- SQLite connection to `email_detector.db`
- Table `predictions` with columns: `id`, `job_text`, `prediction`, `confidence`, `model_used`, `suspicious_keywords`, `created_at`
- Functions: `init_db()`, `save_prediction()`, `get_all_predictions()`, `get_prediction_stats()`

### Step 12: Pydantic Schemas

Open `backend/app/schemas.py` and create:

- `PredictRequest`: `job_text` (str), `model_name` (str, default "naive_bayes")
- `PredictResponse`: `prediction` (str), `confidence` (float), `model_used` (str), `risk_level` (str), `suspicious_keywords` (list), `explanation` (str)

### Step 13: Prediction Route

Open `backend/app/routes/predict.py` and create:

- Load the `.pkl` models and TF-IDF vectorizer at module level
- `POST /api/predict` endpoint that:
    1. Receives job text
    2. Cleans it (same function as training)
    3. Vectorizes with TF-IDF
    4. Predicts using selected model
    5. Extracts suspicious keywords (words with highest fraud probability)
    6. Generates explanation text
    7. Saves to database
    8. Returns result

### Step 14: Metrics Route

Open `backend/app/routes/metrics.py` and create:

- `GET /api/metrics` — returns accuracy/precision/recall/f1 from saved JSON
- `GET /api/history` — returns all past predictions from SQLite
- `GET /api/stats` — returns total predictions count, fake count, real count

### Step 15: FastAPI Main App

Open `backend/app/main.py` and create:

- FastAPI app instance
- CORS middleware (allow `http://localhost:5173`)
- Mount `/static` directory for chart images
- Include predict and metrics routers
- Call `init_db()` on startup

### Step 16: Test the Backend

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project\backend"
.\venv\Scripts\Activate
uvicorn app.main:app --reload --port 8000
```

Open in browser: `http://localhost:8000/docs` — this is the auto-generated Swagger UI

Test these endpoints:

- `POST /api/predict` with sample job text
- `GET /api/metrics`
- `GET /api/history`
- `GET /api/stats`
- Open `http://localhost:8000/static/nb_confusion_matrix.png` — should show chart image

---

## PHASE 3: FRONTEND — REACT (Day 10–16)

### Step 17: Create Frontend Folders

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project\frontend\src"
mkdir components
mkdir pages
mkdir services
```

### Step 18: API Service

Create `frontend/src/services/api.js`:

- Axios instance with `baseURL` pointing to `http://localhost:8000/api`
- Export functions: `predictJob()`, `getHistory()`, `getMetrics()`, `getStats()`

### Step 19: Navbar Component

Create `frontend/src/components/Navbar.jsx`:

- Links to: Home (`/`), Analyze (`/analyze`), Dashboard (`/dashboard`), History (`/history`)
- Active link highlighting

### Step 20: Setup Routing

Edit `frontend/src/App.jsx`:

- Import `BrowserRouter`, `Routes`, `Route` from react-router-dom
- 4 routes: `/`, `/analyze`, `/dashboard`, `/history`

### Step 21: Build the Analyze Page (MOST IMPORTANT)

Create `frontend/src/pages/Analyze.jsx`:

- Large textarea for pasting job description
- Dropdown to select model (Naive Bayes / Logistic Regression)
- "Analyze" button
- Result display showing:
    - FAKE (red) or LEGITIMATE (green) verdict
    - Confidence percentage bar
    - Risk level (High/Medium/Low)
    - Suspicious keywords as colored tags
    - Explanation text

### Step 22: Build the Dashboard Page

Create `frontend/src/pages/Dashboard.jsx`:

- Metric cards: Accuracy, Precision, Recall, F1 for each model
- Recharts BarChart comparing model accuracies
- Confusion matrix images loaded from backend `/static/`
- Recharts PieChart for class distribution

### Step 23: Build the History Page

Create `frontend/src/pages/History.jsx`:

- Table with columns: Job Text (truncated), Prediction, Confidence, Model, Date
- Color-coded badges for Fake (red) / Real (green)

### Step 24: Build the Home Page

Create `frontend/src/pages/Home.jsx`:

- Hero section with project title
- "How it works" section (3 steps: Paste → Analyze → Results)
- CTA button linking to `/analyze`

### Step 25: Styling

Edit `frontend/src/index.css` and `frontend/src/App.css`:

- Modern color scheme (Indigo primary, green=real, red=fake)
- Google Font (Inter or Outfit)
- Card layouts with shadows
- Responsive design

### Step 26: Test Frontend

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project\frontend"
npm run dev
```

Opens at `http://localhost:5173` — test all pages

---

## PHASE 4: INTEGRATION & TESTING (Day 17–18)

### Step 27: Run Both Together

Open **two terminals**:

```powershell
# Terminal 1 — Backend
cd "D:\devVSCode\AI Project\Fake Email Project\backend"
.\venv\Scripts\Activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd "D:\devVSCode\AI Project\Fake Email Project\frontend"
npm run dev
```

### Step 28: Test Full Flow

1. Go to `http://localhost:5173`
2. Navigate to Analyze page
3. Paste a real job posting from LinkedIn → should say "Legitimate"
4. Paste a fake one (e.g., "EARN $5000 DAILY WORK FROM HOME NO EXPERIENCE WIRE TRANSFER") → should say "Fake"
5. Check History page → should show your predictions
6. Check Dashboard → should show charts and metrics

### Step 29: Fix Issues

- CORS errors? Check `main.py` allows `http://localhost:5173`
- API not connecting? Make sure backend is running on port 8000
- Charts not loading? Check `/static/` mount path
- Add loading spinners and error handling

---

## PHASE 5: DEPLOYMENT (Day 19–21)

### Step 30: Push to GitHub

```powershell
cd "D:\devVSCode\AI Project\Fake Email Project"
git init
git add .
git commit -m "Initial commit - Fake Job Posting Detection"
```

Create a repo on GitHub, then:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/fake-job-detector.git
git push -u origin main
```

> ⚠️ Make sure your `.pkl` model files are NOT in `.gitignore` — they need to be deployed with the backend. Remove the `*.pkl` line from `.gitignore` before pushing.

### Step 31: Deploy Backend to Render

1. Go to https://render.com → Sign up (free)
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Settings:
    - **Name**: `fake-job-detector-api`
    - **Root Directory**: `backend`
    - **Runtime**: Python 3
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Click "Create Web Service"
6. Wait for build to complete (~5 minutes)
7. Note down your URL: `https://fake-job-detector-api.onrender.com`

### Step 32: Deploy Frontend to Vercel

1. Go to https://vercel.com → Sign up (free)
2. Click "Add New Project" → Import your GitHub repo
3. Settings:
    - **Root Directory**: `frontend`
    - **Framework Preset**: Vite
    - **Environment Variable**: `VITE_API_URL` = `https://fake-job-detector-api.onrender.com`
4. Click "Deploy"
5. Your live URL: `https://fake-job-detector.vercel.app`

### Step 33: Update Backend CORS

After getting your Vercel URL, update `backend/app/main.py` CORS to also allow your production URL:

```python
allow_origins=[
    "http://localhost:5173",
    "https://fake-job-detector.vercel.app"  # Add your actual Vercel URL
]
```

Push the change → Render will auto-redeploy.

### Step 34: Test Deployed App

- Open your Vercel URL on phone and laptop
- Test predictions, dashboard, history
- Share with a friend to test

---

## PHASE 6: FINAL PREP (Day 22)

### Step 35: Write README.md

Include: project overview, tech stack, how to run locally, screenshots, live demo link, team members.

### Step 36: Viva Prep — Know These Answers

| Question                         | Your Answer                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------- |
| Why Naive Bayes?                 | Great for text classification, based on Bayes' theorem, fast, handles high-dimensional TF-IDF data |
| What is TF-IDF?                  | Term Frequency–Inverse Document Frequency — converts text to numbers based on word importance      |
| What is SMOTE?                   | Synthetic Minority Oversampling — creates synthetic fake-job samples to balance the dataset        |
| Why class imbalance matters?     | Only 4.8% are fake — without SMOTE, model would just predict "real" always and get 95% accuracy    |
| What is the Explainable AI part? | We show which words triggered the detection and give a human-readable reason                       |
| What's your accuracy?            | NB: ~90-95%, LR: ~95-98%, both above the 80% requirement                                           |
| Why two models?                  | Comparison shows analytical depth                                                                  |
| Why FastAPI over Flask?          | Faster, auto-generates API docs, built-in validation with Pydantic                                 |

---

## DAILY COMMANDS CHEAT SHEET

```powershell
# Activate backend
cd "D:\devVSCode\AI Project\Fake Email Project\backend"
.\venv\Scripts\Activate
uvicorn app.main:app --reload --port 8000

# Start frontend
cd "D:\devVSCode\AI Project\Fake Email Project\frontend"
npm run dev

# Train model (run once)
cd "D:\devVSCode\AI Project\Fake Email Project\backend\training"
..\venv\Scripts\Activate
python train_model.py
python evaluate.py
```

| What               | URL                        |
| ------------------ | -------------------------- |
| Frontend           | http://localhost:5173      |
| Backend API        | http://localhost:8000      |
| API Docs (Swagger) | http://localhost:8000/docs |
