/**
 * History.jsx — Prediction History Page
 *
 * Fetches all past predictions from GET /api/history and GET /api/stats,
 * then renders a summary statistics bar and a styled table with color-coded
 * badges, confidence scores, model identifiers, and timestamps.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Search, Loader2, ShieldCheck, ShieldAlert,
  CircleAlert, CircleDot, CircleCheck,
} from "lucide-react";
import { getHistory, getStats } from "../services/api";
import "./History.css";

export default function History() {
  // Holds the array of past prediction records from the backend
  const [records, setRecords] = useState([]);
  // Holds aggregate stats: total predictions, fake count, real count
  const [stats, setStats] = useState(null);
  // Tracks whether the API calls are still in flight
  const [loading, setLoading] = useState(true);

  // Fetch history records and stats once on component mount
  useEffect(() => {
    Promise.all([getHistory(), getStats()])
      .then(([historyData, statsData]) => {
        setRecords(historyData);
        setStats(statsData);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  // Display a spinner while waiting for the API response
  if (loading) {
    return (
      <div className="page">
        <div className="container history-loading">
          <Loader2 size={36} className="history-spinner-icon" />
          <p>Loading prediction history…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="container">
        {/* Page heading */}
        <div className="history-header animate-fade-in-up">
          <h1>Prediction History</h1>
          <p>Review all past job posting analyses</p>
        </div>

        {/* Stats summary bar — aggregate counts above the table */}
        {stats && (
          <div className="history-stats stagger">
            <div className="glass-card history-stat animate-fade-in-up">
              <div className="history-stat-value" style={{ color: "var(--accent-light)" }}>
                {stats.total_predictions}
              </div>
              <div className="history-stat-label">Total Analyses</div>
            </div>
            <div className="glass-card history-stat animate-fade-in-up">
              <div className="history-stat-value" style={{ color: "var(--green)" }}>
                {stats.real_count}
              </div>
              <div className="history-stat-label">Legitimate</div>
            </div>
            <div className="glass-card history-stat animate-fade-in-up">
              <div className="history-stat-value" style={{ color: "var(--red)" }}>
                {stats.fake_count}
              </div>
              <div className="history-stat-label">Fraudulent</div>
            </div>
          </div>
        )}

        {/* Main content: prediction table or empty state */}
        {records.length === 0 ? (
          /* Empty state — no predictions have been recorded yet */
          <div className="glass-card history-empty animate-fade-in-up">
            <Search size={36} className="history-empty-icon" />
            <h3>No Predictions Yet</h3>
            <p>Submit a job posting for analysis to see results here.</p>
            <Link to="/analyze" className="btn btn-primary">
              <Search size={16} />
              Analyze a Posting
            </Link>
          </div>
        ) : (
          /* Prediction history table — one row per past analysis */
          <div className="glass-card history-table-wrapper animate-fade-in-up">
            <table className="history-table" id="history-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Job Text</th>
                  <th>Result</th>
                  <th>Confidence</th>
                  <th>Risk</th>
                  <th>Model</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {records.map((rec, i) => (
                  <tr key={rec.id}>
                    {/* Row index */}
                    <td style={{ color: "var(--text-muted)" }}>{i + 1}</td>

                    {/* Truncated job text — hover for full content */}
                    <td className="history-text-cell" title={rec.job_text}>
                      {rec.job_text.length > 80
                        ? rec.job_text.slice(0, 80) + "…"
                        : rec.job_text}
                    </td>

                    {/* Classification badge — color-coded for instant recognition */}
                    <td>
                      <span
                        className={`badge ${
                          rec.prediction === "Fake" ? "badge-fake" : "badge-real"
                        }`}
                      >
                        {rec.prediction === "Fake"
                          ? <ShieldAlert size={12} />
                          : <ShieldCheck size={12} />
                        }
                        {rec.prediction === "Fake" ? "Fake" : "Real"}
                      </span>
                    </td>

                    {/* Confidence percentage */}
                    <td className="history-confidence">
                      {(rec.confidence * 100).toFixed(1)}%
                    </td>

                    {/* Risk level badge */}
                    <td>
                      <span
                        className={`badge ${
                          rec.risk_level === "High"
                            ? "badge-fake"
                            : rec.risk_level === "Medium"
                            ? "badge-warning"
                            : "badge-real"
                        }`}
                      >
                        {rec.risk_level}
                      </span>
                    </td>

                    {/* Model identifier */}
                    <td>
                      <span className="history-model">
                        {rec.model_used === "logistic_regression" ? "LR" : "NB"}
                      </span>
                    </td>

                    {/* Timestamp */}
                    <td className="history-date">
                      {new Date(rec.created_at).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
