/**
 * Home.jsx — Landing interface introducing the JobGuard Structural Fraud Scanner.
 *
 * Implements a Technical Brutalist + High-End Editorial layout with asymmetric grids,
 * stark typography contrast, direct copy, and clean flat data blocks.
 */

import { Link } from "react-router-dom";
import {
  Search, BarChart3, Database, ShieldCheck, Brain,
  ScaleIcon, GitCompareArrows, Clock, ClipboardPaste,
  Cpu, CheckCircle2,
} from "lucide-react";
import "./Home.css";

export default function Home() {
  return (
    <div className="page">
      {/* Asymmetrical Hero Segment */}
      <section className="home-hero container">
        <div className="hero-grid">
          {/* Left Column: Direct Copy & Call to Actions */}
          <div className="hero-content">
            <div className="mono-label hero-status hero-anim-status">
              <span className="status-dot" />
              STATUS: SCANNER ACTIVE
            </div>
            
            <h1 className="hero-title hero-anim-title">
              We audit job listings for <em>structural signs</em> of fraud.
            </h1>
            
            <p className="hero-subtitle hero-anim-subtitle">
              A mathematical scanner that identifies employment scams. We vectorise text 
              lexicon patterns and evaluate them using classification models trained on 17,880 
              historic records. Transparent, explainable, and fast.
            </p>

            <div className="hero-actions hero-anim-actions">
              <Link to="/analyze" className="btn btn-primary" id="cta-analyze">
                <Search size={14} />
                Scan Posting
              </Link>
              <Link to="/dashboard" className="btn btn-outline" id="cta-dashboard">
                <BarChart3 size={14} />
                Model Benchmarks
              </Link>
            </div>
          </div>

          {/* Right Column: Giant Asymmetric Stat Card */}
          <div className="hero-sidebar hero-anim-card">
            <div className="glass-card hero-highlight-card accent-card">
              <div className="mono-label">PRIMARY CLASSIFIER</div>
              <div className="giant-metric-title">98.6%</div>
              <div className="mono-label text-dim">XGBoost Accuracy</div>
              <div className="divider-line" />
              <p className="stat-card-text">
                Gradient boosted trees with structural feature analysis and class imbalance handling via scale_pos_weight.
              </p>
            </div>
          </div>
        </div>

        {/* Primary Audit Parameters Row */}
        <div className="home-stats">
          <div className="glass-card home-stat">
            <Database size={16} className="home-stat-icon" />
            <div className="home-stat-value">17,880</div>
            <div className="mono-label stat-desc">Audit Records</div>
          </div>
          <div className="glass-card home-stat">
            <ShieldCheck size={16} className="home-stat-icon" />
            <div className="home-stat-value">98.6%</div>
            <div className="mono-label stat-desc">XGBoost Accuracy</div>
          </div>
          <div className="glass-card home-stat">
            <Cpu size={16} className="home-stat-icon" />
            <div className="home-stat-value">2</div>
            <div className="mono-label stat-desc">Parallel Models</div>
          </div>
          <div className="glass-card home-stat">
            <ScaleIcon size={16} className="home-stat-icon" />
            <div className="home-stat-value">4.8%</div>
            <div className="mono-label stat-desc">Historic Scam Rate</div>
          </div>
        </div>
      </section>

      {/* 3-Step Scan Pipeline */}
      <section className="home-how container">
        <div className="section-header-asymmetric">
          <div className="mono-label">Pipeline Workflow</div>
          <h2>Three stages of <em>verification</em></h2>
        </div>

        <div className="home-steps">
          {/* Step 1 */}
          <div className="glass-card home-step">
            <div className="step-header">
              <div className="mono-label">Stage 01</div>
              <div className="home-step-icon-wrap"><ClipboardPaste size={20} /></div>
            </div>
            <h3>Paste Posting Text</h3>
            <p>
              Supply the text of the job description or suspicious email correspondence 
              into our direct analyzer workspace.
            </p>
          </div>

          {/* Step 2 */}
          <div className="glass-card home-step">
            <div className="step-header">
              <div className="mono-label">Stage 02</div>
              <div className="home-step-icon-wrap"><Cpu size={20} /></div>
            </div>
            <h3>Lexical Vectorization</h3>
            <p>
              The text is parsed, stripped of noise, mapped to numerical vectors using TF-IDF, 
              and audited against baseline parameters.
            </p>
          </div>

          {/* Step 3 */}
          <div className="glass-card home-step">
            <div className="step-header">
              <div className="mono-label">Stage 03</div>
              <div className="home-step-icon-wrap"><CheckCircle2 size={20} /></div>
            </div>
            <h3>Risk Assessment</h3>
            <p>
              Review a structured report detailing the final prediction score, statistical confidence, 
              and high-risk phrase alerts.
            </p>
          </div>
        </div>
      </section>

      {/* Engine Architecture Grid */}
      <section className="home-features container">
        <div className="section-header-asymmetric">
          <div className="mono-label">Engine Architecture</div>
          <h2>Core system parameters</h2>
        </div>

        <div className="home-features-grid">
          {/* Feature 1 */}
          <div className="glass-card home-feature">
            <div className="home-feature-icon">
              <Brain size={18} strokeWidth={2} />
            </div>
            <div>
              <h3>Explainable AI Metrics</h3>
              <p>
                We do not output black-box predictions. Every classification reveals the lexical 
                features and specific terms that triggered the model.
              </p>
            </div>
          </div>

          {/* Feature 2 */}
          <div className="glass-card home-feature">
            <div className="home-feature-icon">
              <ScaleIcon size={18} strokeWidth={2} />
            </div>
            <div>
              <h3>Synthetic Oversampling</h3>
              <p>
                Because fraud only accounts for 4.8% of typical job listings, we utilize SMOTE 
                synthetic modeling to balance and train classifiers without bias.
              </p>
            </div>
          </div>

          {/* Feature 3 */}
          <div className="glass-card home-feature">
            <div className="home-feature-icon">
              <GitCompareArrows size={18} strokeWidth={2} />
            </div>
            <div>
              <h3>Dual Model Validation</h3>
              <p>
                We cross-examine data across two models: XGBoost (gradient boosted trees, 
                optimized for precision) and Logistic Regression (optimized for recall).
              </p>
            </div>
          </div>

          {/* Feature 4 */}
          <div className="glass-card home-feature">
            <div className="home-feature-icon">
              <Clock size={18} strokeWidth={2} />
            </div>
            <div>
              <h3>Monochrome History Logs</h3>
              <p>
                All past scanner outcomes are recorded sequentially in an internal local 
                SQLite log ledger to track persistent scam sources.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
