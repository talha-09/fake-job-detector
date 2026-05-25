/**
 * api.js — Centralized Axios HTTP client for communicating with the FastAPI backend.
 *
 * All frontend API calls go through this file so that base URL, headers, and
 * timeouts are configured in one place. In production, set the VITE_API_URL
 * environment variable to point to the deployed Render backend URL.
 */

import axios from "axios";

// Use environment variable if deployed, otherwise default to local dev server
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Create a reusable Axios instance with pre-configured defaults for all API calls
const api = axios.create({
  baseURL: `${API_BASE}/api`,
  headers: { "Content-Type": "application/json" },
  timeout: 15000, // 15-second timeout to handle slow model inference gracefully
});

/**
 * predictJob — Sends job posting text to the backend for classification.
 * The backend cleans the text, vectorizes it with TF-IDF, and runs the
 * selected model (Naive Bayes or Logistic Regression) to return a verdict.
 *
 * @param {string} jobText    — The raw job posting text to analyze
 * @param {string} modelName  — Which model to use: "naive_bayes" or "logistic_regression"
 * @returns {Promise<{prediction, confidence, model_used, risk_level, suspicious_keywords, explanation}>}
 */
export async function predictJob(jobText, modelName = "logistic_regression") {
  const { data } = await api.post("/predict", {
    job_text: jobText,
    model_name: modelName,
  });
  return data;
}

/**
 * getHistory — Fetches the 100 most recent prediction records from SQLite.
 * Each record includes the original text, prediction, confidence, model used,
 * risk level, suspicious keywords, explanation, and timestamp.
 */
export async function getHistory() {
  const { data } = await api.get("/history");
  return data;
}

/**
 * getMetrics — Retrieves model performance metrics (accuracy, precision, recall, F1)
 * and confusion matrices for both Naive Bayes and Logistic Regression models.
 * This data is read from the model_metrics.json file generated during training.
 */
export async function getMetrics() {
  const { data } = await api.get("/metrics");
  return data;
}

/**
 * getStats — Returns aggregate prediction statistics: total count of predictions,
 * how many were classified as fake vs real, and the fake percentage.
 */
export async function getStats() {
  const { data } = await api.get("/stats");
  return data;
}

/**
 * staticUrl — Builds a full URL to access static files (chart PNGs, JSON) hosted
 * by FastAPI's StaticFiles mount at /static/. Used by Dashboard to load confusion
 * matrix images and other evaluation visualizations.
 */
export function staticUrl(filename) {
  return `${API_BASE}/static/${filename}`;
}

export default api;
