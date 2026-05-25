/**
 * Home.jsx — Landing page introducing the Fake Job Posting Detection system.
 *
 * Displays a hero section with CTA buttons, project statistics, a three-step
 * workflow explanation, and a feature grid highlighting key capabilities.
 * All sections use staggered fade-in animations for a polished entrance.
 */

import { Link } from "react-router-dom";
import {
  Search, BarChart3, Database, ShieldCheck, Brain,
  ScaleIcon, GitCompareArrows, Clock, ClipboardPaste,
  Cpu, CheckCircle2, ArrowRight, Zap,
} from "lucide-react";
import "./Home.css";

export default function Home() {
  return (
    <div className="page">
      {/* Hero Section — primary headline with CTA buttons */}
      <section className="home-hero container">
        {/* Status badge indicating the system is operational */}
        <div className="home-hero-badge">
          <span className="home-hero-badge-dot" />
          AI-Powered Detection Engine
        </div>

        {/* Main headline — gradient text highlights the core value proposition */}
        <h1>
          Detect <span className="gradient-text">Fake Job Postings</span>
          <br />Before They Cause Harm
        </h1>

        {/* Subtitle explaining the technology and dataset behind the system */}
        <p className="home-hero-subtitle">
          An explainable AI system leveraging Naive Bayes and Logistic Regression
          models, trained on 17,880 real-world job postings from Kaggle, to identify
          fraudulent listings with high accuracy.
        </p>

        {/* Primary and secondary call-to-action buttons */}
        <div className="home-hero-actions">
          <Link to="/analyze" className="btn btn-primary" id="cta-analyze">
            <Search size={16} />
            Start Analysis
          </Link>
          <Link to="/dashboard" className="btn btn-outline" id="cta-dashboard">
            <BarChart3 size={16} />
            View Dashboard
          </Link>
        </div>

        {/* Quick statistics — establishes credibility with concrete numbers */}
        <div className="home-stats stagger">
          <div className="glass-card home-stat animate-fade-in-up">
            <Database size={20} className="home-stat-icon accent" />
            <div className="home-stat-value accent">17,880</div>
            <div className="home-stat-label">Training Records</div>
          </div>
          <div className="glass-card home-stat animate-fade-in-up">
            <ShieldCheck size={20} className="home-stat-icon green" />
            <div className="home-stat-value green">98.0%</div>
            <div className="home-stat-label">LR Accuracy</div>
          </div>
          <div className="glass-card home-stat animate-fade-in-up">
            <Cpu size={20} className="home-stat-icon cyan" />
            <div className="home-stat-value cyan">2</div>
            <div className="home-stat-label">ML Models</div>
          </div>
          <div className="glass-card home-stat animate-fade-in-up">
            <Zap size={20} className="home-stat-icon" />
            <div className="home-stat-value">4.84%</div>
            <div className="home-stat-label">Fraud Rate</div>
          </div>
        </div>
      </section>

      {/* ── Workflow — three-step process explaining how the system works ── */}
      <section className="home-how container">
        <div className="home-section-title">
          <h2>How It Works</h2>
          <p>Three steps to verify any job posting</p>
        </div>

        <div className="home-steps stagger">
          {/* Step 1: User submits the job description for review */}
          <div className="glass-card home-step animate-fade-in-up">
            <div className="home-step-icon-wrap">
              <ClipboardPaste size={28} strokeWidth={1.5} />
            </div>
            <div className="home-step-number">01</div>
            <h3>Submit Job Posting</h3>
            <p>
              Paste the job description from any source — LinkedIn, Indeed,
              email, or company website — into the analysis form.
            </p>
          </div>

          {/* Step 2: ML pipeline processes text through TF-IDF and classification */}
          <div className="glass-card home-step animate-fade-in-up">
            <div className="home-step-icon-wrap">
              <Cpu size={28} strokeWidth={1.5} />
            </div>
            <div className="home-step-number">02</div>
            <h3>Model Classification</h3>
            <p>
              The text is cleaned, vectorized using TF-IDF, and classified
              by the selected model — Naive Bayes or Logistic Regression.
            </p>
          </div>

          {/* Step 3: User receives a detailed, explainable verdict */}
          <div className="glass-card home-step animate-fade-in-up">
            <div className="home-step-icon-wrap">
              <CheckCircle2 size={28} strokeWidth={1.5} />
            </div>
            <div className="home-step-number">03</div>
            <h3>Review Results</h3>
            <p>
              Receive a clear verdict with confidence score, risk assessment,
              flagged keywords, and a human-readable explanation.
            </p>
          </div>
        </div>
      </section>

      {/* Key Features — highlights the system's differentiators */}
      <section className="home-features container">
        <div className="home-section-title">
          <h2>Key Capabilities</h2>
          <p>Built for transparency, accuracy, and usability</p>
        </div>

        <div className="home-features-grid stagger">
          {/* Explainable AI — the core differentiator from black-box classifiers */}
          <div className="glass-card home-feature animate-fade-in-up">
            <div className="home-feature-icon">
              <Brain size={22} strokeWidth={1.5} />
            </div>
            <div>
              <h3>Explainable AI</h3>
              <p>
                Transparent classification — see which words triggered the
                detection and understand the reasoning behind every verdict.
              </p>
            </div>
          </div>

          {/* SMOTE — addresses the 95:5 class imbalance in the dataset */}
          <div className="glass-card home-feature animate-fade-in-up">
            <div className="home-feature-icon">
              <ScaleIcon size={22} strokeWidth={1.5} />
            </div>
            <div>
              <h3>SMOTE Balancing</h3>
              <p>
                Synthetic Minority Oversampling handles the 95:5 class
                imbalance for reliable detection of rare fraud cases.
              </p>
            </div>
          </div>

          {/* Model comparison — demonstrates analytical depth */}
          <div className="glass-card home-feature animate-fade-in-up">
            <div className="home-feature-icon">
              <GitCompareArrows size={22} strokeWidth={1.5} />
            </div>
            <div>
              <h3>Model Comparison</h3>
              <p>
                Compare Naive Bayes against Logistic Regression with
                detailed metrics, confusion matrices, and performance charts.
              </p>
            </div>
          </div>

          {/* History tracking — persists every prediction for auditing */}
          <div className="glass-card home-feature animate-fade-in-up">
            <div className="home-feature-icon">
              <Clock size={22} strokeWidth={1.5} />
            </div>
            <div>
              <h3>Prediction History</h3>
              <p>
                Every analysis is persisted to a database for review,
                pattern tracking, and aggregate statistics monitoring.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
