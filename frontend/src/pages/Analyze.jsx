/**
 * Analyze.jsx — Core prediction interface for job posting classification.
 *
 * Provides a textarea for pasting job descriptions, a model selector dropdown,
 * and a submit button. After classification, displays a detailed result card
 * with verdict, confidence bar, risk level, flagged keywords, and explanation.
 * This page demonstrates the Explainable AI aspect of the system.
 */

import { useState } from "react";
import {
  Search, AlertTriangle, ShieldCheck, ShieldAlert,
  Loader2, ArrowLeft, Tag, FileText, TrendingUp,
  CircleAlert, CircleDot, CircleCheck,
} from "lucide-react";
import { predictJob } from "../services/api";
import "./Analyze.css";

export default function Analyze() {
  // The raw job posting text entered by the user in the textarea
  const [jobText, setJobText] = useState("");
  // Which ML model to use — defaults to LR since it has higher accuracy
  const [modelName, setModelName] = useState("logistic_regression");
  // Whether an API request is currently in flight
  const [loading, setLoading] = useState(false);
  // Error message to display if the API call fails or input is invalid
  const [error, setError] = useState(null);
  // The prediction result object returned from the backend
  const [result, setResult] = useState(null);

  /**
   * handleSubmit — Validates input length, calls the prediction API,
   * and stores the result or error in state for rendering.
   */
  async function handleSubmit(e) {
    e.preventDefault();
    // Backend requires at least 20 characters for meaningful text analysis
    if (jobText.trim().length < 20) {
      setError("Please enter at least 20 characters of job posting text.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Send text to POST /api/predict and get back the classification result
      const data = await predictJob(jobText, modelName);
      setResult(data);
    } catch (err) {
      // Extract the most useful error message from the Axios error object
      const msg =
        err.response?.data?.detail || err.message || "An unexpected error occurred.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  /**
   * handleReset — Clears the result and input so the user can analyze another posting.
   */
  function handleReset() {
    setResult(null);
    setError(null);
    setJobText("");
  }

  // Convenience flag to determine color scheme (red for fake, green for real)
  const isFake = result?.prediction === "Fake";

  return (
    <div className="page">
      <div className="container">
        {/* Page header */}
        <div className="analyze-header animate-fade-in-up">
          <h1>Analyze Job Posting</h1>
          <p>Submit a job description to classify it as legitimate or fraudulent</p>
        </div>

        {/* ── Input Form — shown only when there's no result or loading state ── */}
        {!result && !loading && (
          <form onSubmit={handleSubmit}>
            <div className="glass-card analyze-input-card animate-fade-in-up">
              {/* Textarea where the user pastes the full job posting text */}
              <textarea
                id="job-text-input"
                className="analyze-textarea"
                placeholder="Paste the full job posting text here...

Example: We are looking for a Software Engineer to join our growing team. Requirements include 3+ years of Python experience, knowledge of REST APIs..."
                value={jobText}
                onChange={(e) => setJobText(e.target.value)}
                maxLength={10000}
              />

              <div className="analyze-controls">
                {/* Model selector — lets users choose between NB and LR models */}
                <div className="analyze-model-select">
                  <label htmlFor="model-select">Model:</label>
                  <select
                    id="model-select"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                  >
                    <option value="logistic_regression">
                      Logistic Regression (Recommended)
                    </option>
                    <option value="naive_bayes">Naive Bayes</option>
                  </select>
                </div>

                {/* Character counter — helps users gauge input length */}
                <span className="analyze-char-count">
                  {jobText.length.toLocaleString()} / 10,000
                </span>

                {/* Submit button — disabled until minimum 20 characters are entered */}
                <button
                  type="submit"
                  className="btn btn-primary analyze-submit"
                  id="analyze-btn"
                  disabled={jobText.trim().length < 20}
                >
                  <Search size={16} />
                  Analyze
                </button>
              </div>
            </div>
          </form>
        )}

        {/* Error banner — shown if validation fails or the API returns an error */}
        {error && (
          <div className="analyze-error" role="alert">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        {/* Loading state — visible while waiting for the backend response */}
        {loading && (
          <div className="analyze-loading glass-card">
            <Loader2 size={36} className="analyze-spinner-icon" />
            <p>
              Classifying with{" "}
              {modelName === "logistic_regression"
                ? "Logistic Regression"
                : "Naive Bayes"}
              …
            </p>
          </div>
        )}

        {/* ── Result Card — displayed after a successful prediction ─────── */}
        {result && (
          <>
            <div className="analyze-result" id="analyze-result">
              {/* Verdict header — prominent label with colored background */}
              <div className={`analyze-verdict ${isFake ? "fake" : "real"}`}>
                <div className="analyze-verdict-icon-wrap">
                  {isFake
                    ? <ShieldAlert size={40} strokeWidth={1.5} />
                    : <ShieldCheck size={40} strokeWidth={1.5} />
                  }
                </div>
                <div className="analyze-verdict-label">
                  {isFake ? "Fraudulent Posting Detected" : "Legitimate Posting"}
                </div>
                <div className="analyze-verdict-model">
                  Classified using{" "}
                  {result.model_used === "logistic_regression"
                    ? "Logistic Regression"
                    : "Naive Bayes"}
                </div>
              </div>

              {/* Detailed breakdown — confidence, risk, keywords, explanation */}
              <div className="analyze-details">
                {/* Confidence score — animated progress bar showing model certainty */}
                <div className="analyze-detail-row">
                  <div className="analyze-detail-label">
                    <TrendingUp size={14} />
                    Confidence
                  </div>
                  <div className="analyze-detail-value">
                    <div className="analyze-confidence-bar-track">
                      <div
                        className={`analyze-confidence-bar-fill ${isFake ? "fake" : "real"}`}
                        style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                      />
                    </div>
                    <div className="analyze-confidence-text">
                      <strong>{(result.confidence * 100).toFixed(1)}%</strong>{" "}
                      confidence
                    </div>
                  </div>
                </div>

                {/* Risk level — High/Medium/Low indicator based on confidence */}
                <div className="analyze-detail-row">
                  <div className="analyze-detail-label">
                    <CircleAlert size={14} />
                    Risk Level
                  </div>
                  <div className="analyze-detail-value">
                    <span className={`analyze-risk ${result.risk_level.toLowerCase()}`}>
                      {result.risk_level === "High" && <CircleAlert size={12} />}
                      {result.risk_level === "Medium" && <CircleDot size={12} />}
                      {result.risk_level === "Low" && <CircleCheck size={12} />}
                      {result.risk_level} Risk
                    </span>
                  </div>
                </div>

                {/* Flagged keywords — terms matching known fraud patterns */}
                <div className="analyze-detail-row">
                  <div className="analyze-detail-label">
                    <Tag size={14} />
                    Flagged Keywords
                  </div>
                  <div className="analyze-detail-value">
                    {result.suspicious_keywords.length > 0 ? (
                      <div className="analyze-keywords">
                        {result.suspicious_keywords.map((kw, i) => (
                          <span key={i} className="analyze-keyword-tag">
                            {kw}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="analyze-no-keywords">
                        No suspicious keywords identified
                      </span>
                    )}
                  </div>
                </div>

                {/* AI explanation — human-readable reasoning behind the decision */}
                <div className="analyze-detail-row">
                  <div className="analyze-detail-label">
                    <FileText size={14} />
                    Explanation
                  </div>
                  <div className="analyze-detail-value">
                    <div className="analyze-explanation">
                      {result.explanation}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Button to clear results and submit another posting */}
            <div className="analyze-again">
              <button
                className="btn btn-outline"
                onClick={handleReset}
                id="analyze-again-btn"
              >
                <ArrowLeft size={16} />
                Analyze Another
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
