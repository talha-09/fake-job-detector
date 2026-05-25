/**
 * App.jsx — Root application component with React Router navigation.
 *
 * Sets up BrowserRouter with 4 routes (Home, Analyze, Dashboard, History),
 * includes the sticky Navbar on all pages, and renders a footer.
 * This file replaces the default Vite boilerplate with our app structure.
 */

import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Analyze from "./pages/Analyze";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        {/* Sticky navigation bar — always visible on all pages */}
        <Navbar />

        {/* Main content area — renders the matched route's page component */}
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>

        {/* Footer — static copyright text at the bottom of every page */}
        <footer className="app-footer">
          © 2026 JobGuard AI — Explainable Fake Job Posting Detection
        </footer>
      </div>
    </BrowserRouter>
  );
}
