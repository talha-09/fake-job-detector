/**
 * Navbar.jsx — Fixed navigation bar with responsive mobile menu.
 *
 * Uses React Router's useLocation to highlight the active page link.
 * On viewports below 768px, links collapse into a toggleable dropdown.
 * Icons are rendered via Lucide React for a clean, professional look.
 */

import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Home, Search, BarChart3, ClipboardList, Shield, Menu, X } from "lucide-react";
import "./Navbar.css";

// Navigation items — each maps a route path to a label and icon component
const NAV_ITEMS = [
  { to: "/",          label: "Home",      Icon: Home },
  { to: "/analyze",   label: "Analyze",   Icon: Search },
  { to: "/dashboard", label: "Dashboard", Icon: BarChart3 },
  { to: "/history",   label: "History",   Icon: ClipboardList },
];

export default function Navbar() {
  // Get current route path to determine which link should be highlighted
  const { pathname } = useLocation();
  // Toggle state for the mobile hamburger menu (open/closed)
  const [open, setOpen] = useState(false);

  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-inner">
        {/* Brand logo — clicking navigates to the home page */}
        <Link to="/" className="navbar-brand" onClick={() => setOpen(false)}>
          <div className="navbar-brand-icon">
            <Shield size={15} strokeWidth={2.5} />
          </div>
          <div className="navbar-brand-text">
            <span>JobGuard</span>
          </div>
        </Link>

        {/* Hamburger button — only visible on mobile, toggles the nav menu */}
        <button
          className="navbar-toggle"
          onClick={() => setOpen((o) => !o)}
          aria-label="Toggle navigation"
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>

        {/* Navigation links — inline on desktop, dropdown on mobile */}
        <ul className={`navbar-links ${open ? "open" : ""}`}>
          {NAV_ITEMS.map(({ to, label, Icon }) => (
            <li key={to}>
              <Link
                to={to}
                className={`navbar-link ${pathname === to ? "active" : ""}`}
                onClick={() => setOpen(false)}
                id={`nav-${label.toLowerCase()}`}
              >
                <Icon size={16} strokeWidth={2} />
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
