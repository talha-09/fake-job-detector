/**
 * Analyze.jsx — Mathematical vector analysis form.
 *
 * Implements a Technical Brutalist scan workspace and outputs results styled
 * as an official "Audit Memorandum". Features crisp monospace diagnostic metrics
 * and sharp-edged visual indicators.
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
  const [jobText, setJobText] = useState("");
  const [modelName, setModelName] = useState("logistic_regression");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (jobText.trim().length < 20) {
      setError("Input underlength. Please enter at least 20 characters of job text.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictJob(jobText, modelName);
      setResult(data);
    } catch (err) {
      const msg =
        err.response?.data?.detail || err.message || "An unexpected system error occurred.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setResult(null);
    setError(null);
    setJobText("");
  }

  const isFake = result?.prediction === "Fake";

  return (
    <div className="page">
      <div className="container">
        {/* Editorial Section Header */}
        <div className="analyze-header animate-fade-in-up">
          <div className="mono-label">VERIFICATION WORKSPACE</div>
          <h1>Audit Job Posting</h1>
          <p>We parse lexical patterns to verify legitimacy against 17,880 historic job postings.</p>
        </div>

        {/* Input Form — shown only when no result is loaded */}
        {!result && !loading && (
          <form onSubmit={handleSubmit} className="animate-fade-in-up">
            <div className="glass-card analyze-input-card">
              <div className="textarea-terminal-header">
                <span className="terminal-dot red" />
                <span className="terminal-dot yellow" />
                <span className="terminal-dot green" />
                <span className="terminal-filename">job_description.txt</span>
              </div>
              
              <textarea
                id="job-text-input"
                className="analyze-textarea"
                placeholder="Paste the full job description or email correspondence here... (minimum 20 characters)"
                value={jobText}
                onChange={(e) => setJobText(e.target.value)}
                maxLength={10000}
              />

              <div className="analyze-controls">
                {/* Asymmetric dropdown */}
                <div className="analyze-model-select">
                  <label htmlFor="model-select" className="mono-label">Classifier Model:</label>
                  <select
                    id="model-select"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                  >
                    <option value="logistic_regression">
                      Logistic Regression [98.0% Accuracy]
                    </option>
                    <option value="naive_bayes">
                      Naive Bayes [90.7% Accuracy]
                    </option>
                  </select>
                </div>

                <span className="analyze-char-count mono-label">
                  {jobText.length.toLocaleString()} / 10,000 bytes
                </span>

                <button
                  type="submit"
                  className="btn btn-primary analyze-submit"
                  id="analyze-btn"
                  disabled={jobText.trim().length < 20}
                >
                  <Search size={13} />
                  Analyze Text
                </button>
              </div>
            </div>
          </form>
        )}

        {/* Error Flag Banner */}
        {error && (
          <div className="analyze-error mono-label" role="alert">
            <AlertTriangle size={14} />
            Error: {error}
          </div>
        )}

        {/* Technical Loading Status */}
        {loading && (
          <div className="analyze-loading glass-card">
            <Loader2 size={24} className="analyze-spinner-icon" />
            <p className="mono-label animate-pulse">
              Tokenising text via TF-IDF vectoriser... Running{" "}
              {modelName === "logistic_regression"
                ? "logistic_regression_model"
                : "naive_bayes_model"}
              ...
            </p>
          </div>
        )}

        {/* Technical Brutalist Audit Memorandum Report */}
        {result && (
          <div className="result-container animate-fade-in-up">
            <div className={`analyze-result ${isFake ? "scam-audit" : "safe-audit"}`}>
              
              {/* Memorandum Header Block */}
              <div className="memo-header-block">
                <div className="memo-brand-stamp">
                  <div className="stamp-title">JOBGUARD SCANNER</div>
                  <div className="stamp-id mono-label">Id: {Math.floor(Math.random() * 900000 + 100000)}</div>
                </div>

                <div className="verdict-banner">
                  <div className="verdict-icon-wrap">
                    {isFake
                      ? <ShieldAlert size={28} />
                      : <ShieldCheck size={28} />
                    }
                  </div>
                  <div className="verdict-text-group">
                    <div className="mono-label">AUDIT MEMORANDUM VERDICT</div>
                    <div className="verdict-large-title">
                      {isFake ? "FLAGGED: HIGH RISK OF SCAM" : "VERIFIED: LOW RISK / LEGITIMATE"}
                    </div>
                  </div>
                </div>

                <div className="metadata-grid">
                  <div className="meta-item">
                    <span className="meta-key mono-label">Scan Timestamp:</span>
                    <span className="meta-val mono-label">{new Date().toISOString().slice(0, 19).replace('T', ' ')} UTC</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-key mono-label">Audit Engine:</span>
                    <span className="meta-val mono-label">
                      {result.model_used === "logistic_regression"
                        ? "LOGISTIC_REGRESSION_V1"
                        : "NAIVE_BAYES_V1"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Statistical Ledger Breakdown */}
              <div className="analyze-details">
                
                {/* Row 1: Model Certainty */}
                <div className="analyze-detail-row">
                  <div className="analyze-detail-label mono-label">
                    <TrendingUp size={12} />
                    Confidence
                  </div>
                  <div className="analyze-detail-value">
                    <div className="analyze-confidence-bar-track">
                      <div
                        className={`analyze-confidence-bar-fill ${isFake ? "fake" : "real"}`}
                        style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                      />
                    </div>
                    <div className="analyze-confidence-text mono-label">
                      Classifier probability coefficient: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                    </div>
                  </div>
                </div>

                {/* Row 2: Risk Class */}
                <div className="analyze-detail-row">
                  <div className="mono-label analyze-detail-label">
                    <CircleAlert size={12} />
                    Risk Level
                  </div>
                  <div className="analyze-detail-value">
                    <span className={`analyze-risk ${result.risk_level.toLowerCase()}`}>
                      {result.risk_level === "High" && <CircleAlert size={12} />}
                      {result.risk_level === "Medium" && <CircleDot size={12} />}
                      {result.risk_level === "Low" && <CircleCheck size={12} />}
                      [{result.risk_level} Risk Class]
                    </span>
                  </div>
                </div>

                {/* Row 3: Flagged Lexical Markers */}
                <div className="analyze-detail-row">
                  <div className="mono-label analyze-detail-label">
                    <Tag size={12} />
                    Lexical Keys
                  </div>
                  <div className="analyze-detail-value">
                    {result.suspicious_keywords.length > 0 ? (
                      <div className="analyze-keywords">
                        {result.suspicious_keywords.map((kw, i) => (
                          <span key={i} className="analyze-keyword-tag mono-label">
                            [{kw}]
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="analyze-no-keywords mono-label text-dim">
                        No suspicious keywords identified
                      </span>
                    )}
                  </div>
                </div>

                {/* Row 4: Narrative Reasoning */}
                <div className="analyze-detail-row">
                  <div className="mono-label analyze-detail-label">
                    <FileText size={12} />
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

            {/* Form reset button */}
            <div className="analyze-again">
              <button
                className="btn btn-outline"
                onClick={handleReset}
                id="analyze-again-btn"
              >
                <ArrowLeft size={13} />
                Scan Another Posting
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
