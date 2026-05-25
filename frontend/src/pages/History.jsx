/**
 * History.jsx — Prediction History Page
 *
 * Implements a Technical Brutalist ledger sheet detailing past classification records
 * with flat tabular grids, monospace technical keys, and sharp status badges.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Search, Loader2, ShieldCheck, ShieldAlert,
} from "lucide-react";
import { getHistory, getStats } from "../services/api";
import "./History.css";

export default function History() {
  const [records, setRecords] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getHistory(), getStats()])
      .then(([historyData, statsData]) => {
        setRecords(historyData);
        setStats(statsData);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page">
        <div className="container history-loading">
          <Loader2 size={24} className="history-spinner-icon" />
          <p className="mono-label animate-pulse">Querying local DB logs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="container">
        {/* Editorial Section Header */}
        <div className="history-header animate-fade-in-up">
          <div className="mono-label">DATABASE TRANSACTION HISTORY</div>
          <h1>Audit Ledger</h1>
          <p>Historical database of processed job listings and structural scam scans.</p>
        </div>

        {/* Minimal Stats Row */}
        {stats && (
          <div className="history-stats stagger">
            <div className="glass-card history-stat accent-card">
              <div className="history-stat-value">
                {stats.total_predictions}
              </div>
              <div className="mono-label stat-desc">Total Audits</div>
            </div>
            <div className="glass-card history-stat">
              <div className="history-stat-value" style={{ color: "var(--green)" }}>
                {stats.real_count}
              </div>
              <div className="mono-label stat-desc">Verified Legit</div>
            </div>
            <div className="glass-card history-stat">
              <div className="history-stat-value" style={{ color: "var(--red)" }}>
                {stats.fake_count}
              </div>
              <div className="mono-label stat-desc">Flagged Fraud</div>
            </div>
          </div>
        )}

        {/* Main Content Ledger */}
        {records.length === 0 ? (
          <div className="glass-card history-empty animate-fade-in-up">
            <Search size={28} className="history-empty-icon" />
            <h3>Ledger Empty</h3>
            <p className="mono-label">No previous scans have been recorded in the database.</p>
            <Link to="/analyze" className="btn btn-primary" style={{ marginTop: 16 }}>
              <Search size={13} />
              Scan a Posting
            </Link>
          </div>
        ) : (
          <div className="glass-card history-table-wrapper animate-fade-in-up">
            <table className="history-table" id="history-table">
              <thead>
                <tr>
                  <th className="mono-label">#</th>
                  <th className="mono-label">Job Text Scan</th>
                  <th className="mono-label">Verdict</th>
                  <th className="mono-label">Confidence</th>
                  <th className="mono-label">Risk Class</th>
                  <th className="mono-label">Classifier</th>
                  <th className="mono-label">Date Recorded</th>
                </tr>
              </thead>
              <tbody>
                {records.map((rec, i) => (
                  <tr key={rec.id}>
                    {/* Index */}
                    <td className="history-index mono-label">{i + 1}</td>

                    {/* Truncated job description */}
                    <td className="history-text-cell" title={rec.job_text}>
                      {rec.job_text.length > 70
                        ? rec.job_text.slice(0, 70) + "..."
                        : rec.job_text}
                    </td>

                    {/* Classification square badge */}
                    <td>
                      <span
                        className={`badge ${
                          rec.prediction === "Fake" ? "badge-fake" : "badge-real"
                        }`}
                      >
                        {rec.prediction === "Fake"
                          ? <ShieldAlert size={10} />
                          : <ShieldCheck size={10} />
                        }
                        {rec.prediction === "Fake" ? "Scam" : "Legit"}
                      </span>
                    </td>

                    {/* Tabular numeric confidence */}
                    <td className="history-confidence mono-label">
                      {(rec.confidence * 100).toFixed(1)}%
                    </td>

                    {/* Risk Badge */}
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

                    {/* Engine Identifier */}
                    <td className="history-model mono-label">
                      {rec.model_used === "logistic_regression" ? "LR_V1" : "NB_V1"}
                    </td>

                    {/* Monospace simple timestamp */}
                    <td className="history-date mono-label">
                      {new Date(rec.created_at).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "2-digit",
                        day: "2-digit",
                        hour: "2-digit",
                        minute: "2-digit",
                        hour12: false
                      }).replace(',', '')}
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
