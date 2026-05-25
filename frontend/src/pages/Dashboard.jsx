/**
 * Dashboard.jsx — Model Performance and Visualization Page
 *
 * Fetches accuracy/precision/recall/f1 from GET /api/metrics,
 * renders KPI cards, a Recharts grouped bar chart, confusion matrix
 * images from the backend /static/ directory, and dataset statistics.
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
  // State: metrics data from backend, loading flag, and error message
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch model performance metrics once on page mount
  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch((err) => setError(err.response?.data?.detail || err.message))
      .finally(() => setLoading(false));
  }, []);

  // Show a spinner while the API call is in progress
  if (loading) {
    return (
      <div className="page">
        <div className="container dashboard-loading">
          <Loader2 size={36} className="dashboard-spinner-icon" />
          <p>Loading dashboard metrics…</p>
        </div>
      </div>
    );
  }

  // Show an error message if the metrics fetch failed
  if (error) {
    return (
      <div className="page">
        <div className="container dashboard-error">
          <AlertTriangle size={32} />
          <h2>Failed to Load Metrics</h2>
          <p>{error}</p>
          <p style={{ marginTop: 12, color: "var(--text-muted)", fontSize: "0.85rem" }}>
            Ensure the backend is running on port 8000 and model_metrics.json exists in backend/static/
          </p>
        </div>
      </div>
    );
  }

  // Destructure the two model metric objects and dataset info
  const nb = metrics.naive_bayes;
  const lr = metrics.logistic_regression;
  const ds = metrics.dataset;

  // Build chart data — one bar group per metric, comparing NB vs LR
  const chartData = [
    { metric: "Accuracy",  nb: +(nb.accuracy  * 100).toFixed(1), lr: +(lr.accuracy  * 100).toFixed(1) },
    { metric: "Precision", nb: +(nb.precision * 100).toFixed(1), lr: +(lr.precision * 100).toFixed(1) },
    { metric: "Recall",    nb: +(nb.recall    * 100).toFixed(1), lr: +(lr.recall    * 100).toFixed(1) },
    { metric: "F1 Score",  nb: +(nb.f1_score  * 100).toFixed(1), lr: +(lr.f1_score  * 100).toFixed(1) },
  ];

  /**
   * handleImageLoad — Fades in confusion matrix images after they finish loading
   * to prevent a jarring flash of unstyled content.
   */
  function handleImageLoad(e) {
    e.target.classList.add("loaded");
  }

  return (
    <div className="page">
      <div className="container">
        {/* Page header */}
        <div className="dashboard-header animate-fade-in-up">
          <h1>Model Dashboard</h1>
          <p>Performance metrics and visualizations for both classification models</p>
        </div>

        {/* ── KPI Metric Cards — quick overview of best-model performance ── */}
        <div className="dashboard-metrics stagger">
          <div className="glass-card dashboard-metric-card accent animate-fade-in-up">
            <Target size={22} className="dashboard-metric-icon-svg accent" />
            <div className="dashboard-metric-value accent">
              {(lr.accuracy * 100).toFixed(1)}%
            </div>
            <div className="dashboard-metric-label">LR Accuracy</div>
          </div>

          <div className="glass-card dashboard-metric-card green animate-fade-in-up">
            <CheckCircle2 size={22} className="dashboard-metric-icon-svg green" />
            <div className="dashboard-metric-value green">
              {(lr.precision * 100).toFixed(1)}%
            </div>
            <div className="dashboard-metric-label">LR Precision</div>
          </div>

          <div className="glass-card dashboard-metric-card cyan animate-fade-in-up">
            <Search size={22} className="dashboard-metric-icon-svg cyan" />
            <div className="dashboard-metric-value cyan">
              {(lr.recall * 100).toFixed(1)}%
            </div>
            <div className="dashboard-metric-label">LR Recall</div>
          </div>

          <div className="glass-card dashboard-metric-card yellow animate-fade-in-up">
            <Scale size={22} className="dashboard-metric-icon-svg yellow" />
            <div className="dashboard-metric-value yellow">
              {(lr.f1_score * 100).toFixed(1)}%
            </div>
            <div className="dashboard-metric-label">LR F1 Score</div>
          </div>
        </div>

        {/* ── Two-column: comparison table + bar chart ──────────────────── */}
        <div className="dashboard-grid">
          {/* Left: side-by-side performance comparison table */}
          <div className="glass-card animate-fade-in-up">
            <div className="dashboard-card-title">
              <LayoutGrid size={16} />
              Model Comparison
            </div>
            <table className="dashboard-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Naive Bayes</th>
                  <th>Logistic Regression</th>
                </tr>
              </thead>
              <tbody>
                {/* Each row compares one metric across both models */}
                {[
                  ["Accuracy",  nb.accuracy,  lr.accuracy],
                  ["Precision", nb.precision, lr.precision],
                  ["Recall",    nb.recall,    lr.recall],
                  ["F1 Score",  nb.f1_score,  lr.f1_score],
                ].map(([label, nbVal, lrVal]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td className={`value-cell ${nbVal > lrVal ? "highlight" : ""}`}>
                      {(nbVal * 100).toFixed(1)}%
                    </td>
                    <td className={`value-cell ${lrVal >= nbVal ? "highlight" : ""}`}>
                      {(lrVal * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Right: Recharts grouped bar chart for visual comparison */}
          <div className="glass-card animate-fade-in-up">
            <div className="dashboard-card-title">
              <TrendingUp size={16} />
              Performance Overview
            </div>
            <div className="dashboard-chart-container">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="metric"
                    tick={{ fill: "#8b8da3", fontSize: 12 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: "#8b8da3", fontSize: 12 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#1e1f2e",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 8,
                      color: "#e4e5f1",
                      fontSize: 13,
                    }}
                    formatter={(v) => `${v}%`}
                  />
                  <Legend wrapperStyle={{ fontSize: 12, color: "#8b8da3" }} />
                  <Bar dataKey="nb" name="Naive Bayes" radius={[4,4,0,0]}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill="#6366f1" />
                    ))}
                  </Bar>
                  <Bar dataKey="lr" name="Logistic Regression" radius={[4,4,0,0]}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill="#a78bfa" />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* ── Confusion matrix images generated by evaluate.py ──────────── */}
        <div className="dashboard-confusion-grid stagger">
          <div className="glass-card dashboard-confusion-card animate-fade-in-up">
            <div className="dashboard-card-title">
              Naive Bayes — Confusion Matrix
            </div>
            <img
              src={staticUrl("nb_confusion_matrix.png")}
              alt="Naive Bayes confusion matrix"
              className="dashboard-confusion-img"
              onLoad={handleImageLoad}
            />
          </div>

          <div className="glass-card dashboard-confusion-card animate-fade-in-up">
            <div className="dashboard-card-title">
              Logistic Regression — Confusion Matrix
            </div>
            <img
              src={staticUrl("lr_confusion_matrix.png")}
              alt="Logistic Regression confusion matrix"
              className="dashboard-confusion-img"
              onLoad={handleImageLoad}
            />
          </div>
        </div>

        {/* ── Dataset statistics — training data overview ───────────────── */}
        <div className="glass-card dashboard-dataset animate-fade-in-up">
          <div className="dashboard-card-title" style={{ width: "100%", justifyContent: "center" }}>
            <Database size={16} />
            Training Dataset
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value">{ds.total_rows.toLocaleString()}</div>
            <div className="dashboard-dataset-label">Total Records</div>
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value" style={{ color: "var(--green)" }}>
              {ds.real_postings.toLocaleString()}
            </div>
            <div className="dashboard-dataset-label">Legitimate</div>
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value" style={{ color: "var(--red)" }}>
              {ds.fake_postings.toLocaleString()}
            </div>
            <div className="dashboard-dataset-label">Fraudulent</div>
          </div>
          <div className="dashboard-dataset-item">
            <div className="dashboard-dataset-value" style={{ color: "var(--yellow)" }}>
              {ds.fake_percent}%
            </div>
            <div className="dashboard-dataset-label">Fraud Rate</div>
          </div>
        </div>
      </div>
    </div>
  );
}
