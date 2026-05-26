/**
 * Dashboard.jsx — Model Performance and Visualization Page
 *
 * Implements flat, technical brutalist charts using customized Recharts instances,
 * styled KPI ledgers, and raw training dataset parameters.
 * Compares XGBoost (primary) vs Logistic Regression (secondary).
 */

import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from "recharts";
import {
  Target, CheckCircle2, Search, Scale, LayoutGrid,
  TrendingUp, Database, ShieldCheck, AlertTriangle, Loader2,
} from "lucide-react";
import { getMetrics, staticUrl } from "../services/api";
import "./Dashboard.css";

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch((err) => setError(err.response?.data?.detail || err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page">
        <div className="container dashboard-loading">
          <Loader2 size={24} className="dashboard-spinner-icon" />
          <p className="mono-label animate-pulse">Loading benchmark metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <div className="container dashboard-error">
          <AlertTriangle size={24} />
          <h2>Benchmark Loading Error</h2>
          <p className="mono-label">{error}</p>
        </div>
      </div>
    );
  }

  const xgb = metrics.xgboost;
  const lr = metrics.logistic_regression;
  const ds = metrics.dataset;

  const chartData = [
    { metric: "Accuracy",  xgb: +(xgb.accuracy  * 100).toFixed(1), lr: +(lr.accuracy  * 100).toFixed(1) },
    { metric: "Precision", xgb: +(xgb.precision * 100).toFixed(1), lr: +(lr.precision * 100).toFixed(1) },
    { metric: "Recall",    xgb: +(xgb.recall    * 100).toFixed(1), lr: +(lr.recall    * 100).toFixed(1) },
    { metric: "F1 Score",  xgb: +(xgb.f1_score  * 100).toFixed(1), lr: +(lr.f1_score  * 100).toFixed(1) },
  ];

  function handleImageLoad(e) {
    e.target.classList.add("loaded");
  }

  return (
    <div className="page">
      <div className="container">
        {/* Editorial Page Header */}
        <div className="dashboard-header animate-fade-in-up">
          <div className="mono-label">MODEL EVALUATION</div>
          <h1>System Benchmarks</h1>
          <p>Performance metrics and visual matrix audits for XGBoost and Logistic Regression.</p>
        </div>

        {/* Start Asymmetric KPI Ledger — XGBoost metrics */}
        <div className="dashboard-metrics stagger">
          <div className="glass-card dashboard-metric-card accent-card">
            <Target size={14} className="dashboard-metric-icon-svg" />
            <div className="dashboard-metric-value">
              {(xgb.accuracy * 100).toFixed(1)}%
            </div>
            <div className="mono-label stat-desc">XGBoost Accuracy</div>
          </div>

          <div className="glass-card dashboard-metric-card">
            <CheckCircle2 size={14} className="dashboard-metric-icon-svg" />
            <div className="dashboard-metric-value">
              {(xgb.precision * 100).toFixed(1)}%
            </div>
            <div className="mono-label stat-desc">XGBoost Precision</div>
          </div>

          <div className="glass-card dashboard-metric-card">
            <Search size={14} className="dashboard-metric-icon-svg" />
            <div className="dashboard-metric-value">
              {(xgb.recall * 100).toFixed(1)}%
            </div>
            <div className="mono-label stat-desc">XGBoost Recall</div>
          </div>

          <div className="glass-card dashboard-metric-card">
            <Scale size={14} className="dashboard-metric-icon-svg" />
            <div className="dashboard-metric-value">
              {(xgb.f1_score * 100).toFixed(1)}%
            </div>
            <div className="mono-label stat-desc">XGBoost F1 Score</div>
          </div>
        </div>

        {/* Asymmetrical Grid: Comparison Table + Customized Recharts */}
        <div className="dashboard-grid">
          {/* Comparison Table */}
          <div className="glass-card animate-fade-in-up">
            <div className="dashboard-card-title mono-label">
              <LayoutGrid size={13} />
              Evaluation Ledger
            </div>
            <table className="dashboard-table">
              <thead>
                <tr>
                  <th className="mono-label">Metric Parameter</th>
                  <th className="mono-label">XGBoost</th>
                  <th className="mono-label">Logistic Regression</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Accuracy",  xgb.accuracy,  lr.accuracy],
                  ["Precision", xgb.precision, lr.precision],
                  ["Recall",    xgb.recall,    lr.recall],
                  ["F1 Score",  xgb.f1_score,  lr.f1_score],
                ].map(([label, xgbVal, lrVal]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td className={`value-cell ${xgbVal >= lrVal ? "highlight-xgb" : ""}`}>
                      {(xgbVal * 100).toFixed(1)}%
                    </td>
                    <td className={`value-cell ${lrVal > xgbVal ? "highlight-lr" : ""}`}>
                      {(lrVal * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Grouped Bar Chart Override */}
          <div className="glass-card animate-fade-in-up">
            <div className="dashboard-card-title mono-label">
              <TrendingUp size={13} />
              Statistical Bar Graph
            </div>
            <div className="dashboard-chart-container">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} barGap={6}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="0" />
                  <XAxis
                    dataKey="metric"
                    tick={{ fill: "#aba69a", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    axisLine={{ stroke: "var(--border)" }}
                    tickLine={{ stroke: "var(--border)" }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: "#aba69a", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    axisLine={{ stroke: "var(--border)" }}
                    tickLine={{ stroke: "var(--border)" }}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(201, 168, 76, 0.03)" }}
                    contentStyle={{
                      background: "var(--bg-secondary)",
                      border: "1px solid var(--border)",
                      borderRadius: "0px",
                      color: "var(--text-primary)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 10,
                    }}
                    formatter={(v) => `${v}%`}
                  />
                  <Legend 
                    wrapperStyle={{ 
                      fontSize: 10, 
                      fontFamily: "var(--font-mono)", 
                      color: "var(--text-secondary)",
                      paddingTop: 12
                    }} 
                  />
                  <Bar dataKey="xgb" name="XGBoost" radius={0}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill="#00BCD4" />
                    ))}
                  </Bar>
                  <Bar dataKey="lr" name="Logistic Regression" radius={0}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill="#c5a65c" />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Backend Confusion Matrices — Combined image */}
        <div className="dashboard-confusion-grid stagger">
          <div className="glass-card dashboard-confusion-card animate-fade-in-up" style={{ gridColumn: "1 / -1" }}>
            <div className="dashboard-card-title mono-label">
              CONFUSION MATRICES — LOGISTIC REGRESSION vs XGBOOST
            </div>
            <img
              src={staticUrl("confusion_matrices.png")}
              alt="Confusion matrices for both models"
              className="dashboard-confusion-img"
              onLoad={handleImageLoad}
            />
          </div>
        </div>

        {/* Additional Charts: PR Curves, ROC Curves, Structural Signals */}
        <div className="dashboard-confusion-grid stagger">
          <div className="glass-card dashboard-confusion-card animate-fade-in-up">
            <div className="dashboard-card-title mono-label">
              PRECISION-RECALL CURVES
            </div>
            <img
              src={staticUrl("pr_curves.png")}
              alt="Precision-Recall curves"
              className="dashboard-confusion-img"
              onLoad={handleImageLoad}
            />
          </div>

          <div className="glass-card dashboard-confusion-card animate-fade-in-up">
            <div className="dashboard-card-title mono-label">
              ROC CURVES
            </div>
            <img
              src={staticUrl("roc_curves.png")}
              alt="ROC curves"
              className="dashboard-confusion-img"
              onLoad={handleImageLoad}
            />
          </div>
        </div>

        {/* Suspicious Keywords + Structural Signals */}
        <div className="dashboard-confusion-grid stagger">
          <div className="glass-card dashboard-confusion-card animate-fade-in-up">
            <div className="dashboard-card-title mono-label">
              TOP SUSPICIOUS KEYWORDS
            </div>
            <img
              src={staticUrl("suspicious_keywords.png")}
              alt="Top suspicious keywords"
              className="dashboard-confusion-img"
              onLoad={handleImageLoad}
            />
          </div>

          <div className="glass-card dashboard-confusion-card animate-fade-in-up">
            <div className="dashboard-card-title mono-label">
              STRUCTURAL FEATURE SIGNALS
            </div>
            <img
              src={staticUrl("structural_signals.png")}
              alt="Structural feature signal strengths"
              className="dashboard-confusion-img"
              onLoad={handleImageLoad}
            />
          </div>
        </div>

        {/* Dataset balance ledger sheet */}
        <div className="glass-card dashboard-dataset animate-fade-in-up">
          <div className="dashboard-card-title mono-label" style={{ width: "100%", justifyContent: "center" }}>
            <Database size={13} />
            Training Corpus Baseline Log
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value">{ds.total_rows.toLocaleString()}</div>
            <div className="mono-label text-dim">Total Corpus Records</div>
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value" style={{ color: "var(--green)" }}>
              {ds.real_postings.toLocaleString()}
            </div>
            <div className="mono-label text-dim">Legitimate Rows</div>
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value" style={{ color: "var(--red)" }}>
              {ds.fake_postings.toLocaleString()}
            </div>
            <div className="mono-label text-dim">Fraudulent Rows</div>
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value" style={{ color: "var(--accent)" }}>
              {ds.fake_percent}%
            </div>
            <div className="mono-label text-dim">Native Fraud Ratio</div>
          </div>
        </div>
      </div>
    </div>
  );
}
