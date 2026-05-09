"""
Telematics Risk Intelligence Platform — Plotly Dash App  (v2)
=============================================================
5 tabs:
  /                → Executive Dashboard
  /explorer        → Risk Explorer
  /underwriting    → Prediction Tool (real-time scoring)
  /decomp          → Risk Decomposition (XAI)
  /claims          → Claims Analytics

Key changes vs old app.py
--------------------------
1.  PRICING ALIGNMENT
    All pure-premium figures are assembled as:
        pure_premium      = p_claim × e_loss_capped          (pre-load)
        indicated_premium = pure_premium × (1 + expense + profit), ≥ min_premium
    Both loadings and the severity cap are read from config/pricing.json via
    risk_scoring.py constants (EXPENSE_LOADING_PCT, PROFIT_LOADING_PCT,
    SEVERITY_CAP_USD, MIN_PREMIUM_USD, TOTAL_LOADING) — the exact same constants
    used in model_train.py.  No calibration squeezes or hardcoded _EXPENSE_LOAD.

2.  TIER THRESHOLDS FROM CONFIG
    TIER_THRESHOLDS, TIER_ORDER, and TIER_BINS are imported from risk_scoring.py
    so they can never drift from the model training pipeline.

3.  PREDICTION TOOL — uses TelematicsRiskScorer.score_one()
    The score_uw callback delegates fully to the scorer, consuming the
    RiskScoreResult dataclass (p_claim, e_loss_given_claim, pure_premium,
    indicated_premium, top_risk_factors, top_protective_factors, shap_chart_data,
    uw_decision, uw_actions, pricing_config).  No bespoke calibration logic.

4.  EXPLAINABILITY
    The "Why this risk?" section renders shap_chart_data directly from the
    scorer — a horizontal waterfall bar chart of signed factor contributions.
    top_risk_factors and top_protective_factors populate the client-story cards.

5.  RISK DECOMPOSITION
    Uses risk_scoring._compute_factor_contributions() for the selected policy so
    contributions are from the same registry as the Prediction Tool.

Run (dev):   python app.py
Run (prod):  waitress-serve --host=0.0.0.0 --port=8050 app:server
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Input, Output, State, callback, dash_table, dcc, html

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = HERE
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR   = os.path.join(ROOT, "outputs")
CFG_DIR   = os.path.join(ROOT, "config")

# ─── Import risk_scoring (adds local dir to path if needed) ──────────────────
sys.path.insert(0, ROOT)
from risk_scoring import (
    TelematicsRiskScorer,
    TIER_ORDER,
    TIER_THRESHOLDS,
    UW_RECOMMENDATIONS,
    EXPENSE_LOADING_PCT,
    PROFIT_LOADING_PCT,
    SEVERITY_CAP_USD,
    MIN_PREMIUM_USD,
    TOTAL_LOADING,
    INPUT_DEFAULTS,
    _compute_factor_contributions,
)

# ─── Load artefacts ───────────────────────────────────────────────────────────
df    = pd.read_csv("telematics_clean.csv")
preds = pd.read_csv(os.path.join(OUT_DIR,  "model_predictions.csv"))

# Merge model outputs — keep only what we need (avoids column collisions)
# suffixes=("_orig", "") ensures model output columns always win over stale CSV values
df = df.merge(
    preds[["policy_id", "p_claim", "e_loss_capped",
           "pure_premium", "indicated_premium", "risk_tier", "log_loss_pred"]],
    on="policy_id", how="left", suffixes=("_orig", ""),
)

with open(os.path.join(CFG_DIR, "feature_selection.json")) as fh:
    cfg = json.load(fh)

with open(os.path.join(OUT_DIR, "model_metrics.json")) as fh:
    metrics = json.load(fh)

# Load model artefacts for decomposition page
pre_freq = pickle.load(open(os.path.join(MODEL_DIR, "preprocessor_freq.pkl"), "rb"))
pre_sev  = pickle.load(open(os.path.join(MODEL_DIR, "preprocessor_sev.pkl"),  "rb"))
glm_freq = pickle.load(open(os.path.join(MODEL_DIR, "glm_frequency.pkl"),     "rb"))
_sev_path = (
    os.path.join(MODEL_DIR, "ebm_severity.pkl")
    if os.path.exists(os.path.join(MODEL_DIR, "ebm_severity.pkl"))
    else os.path.join(MODEL_DIR, "glm_severity.pkl")
)
sev_model = pickle.load(open(_sev_path, "rb"))
_IS_EBM   = hasattr(sev_model, "explain_local")

# Initialise scorer (model-based; falls back to rule-based if artifacts missing)
try:
    scorer = TelematicsRiskScorer.from_artifacts(
        model_dir=MODEL_DIR,
        config_path=os.path.join(CFG_DIR, "feature_selection.json"),
    )
except Exception:
    scorer = TelematicsRiskScorer.rule_based_only()


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG  = "#f0f2f5"
CARD_BG  = "#ffffff"
BORDER   = "#e1e4e8"
TEXT     = "#1a1f2e"
MUTED    = "#6b7280"
BLUE     = "#1d6ff2"
GREEN    = "#16a34a"
RED      = "#dc2626"
ORANGE   = "#d97706"
PURPLE   = "#7c3aed"
CYAN     = "#0891b2"

NAV_BG     = "#1a1f2e"
NAV_TEXT   = "#ffffff"
NAV_MUTED  = "#94a3b8"
NAV_BORDER = "#2d3748"

PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, "#0ea5e9", "#ef4444", "#22c55e", "#f59e0b"]

# Tier colour map — aligned with TIER_ORDER from risk_scoring.py
TIER_COLORS = {
    "LOW":       GREEN,
    "MODERATE":  ORANGE,
    "HIGH":      "#ea580c",
    "VERY_HIGH": RED,
}

# Explorer uses renamed labels for UI display
_DISPLAY_TIER = {"LOW": "Preferred", "MODERATE": "Standard", "HIGH": "Elevated", "VERY_HIGH": "Non-Standard"}
_DISPLAY_ORDER = [_DISPLAY_TIER[t] for t in TIER_ORDER]
RISK_COLORS = {_DISPLAY_TIER[t]: TIER_COLORS[t] for t in TIER_ORDER}

BASE_LAYOUT = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor="#f8fafc",
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=44, r=20, t=48, b=40),
    xaxis=dict(gridcolor="#e5e7eb", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#e5e7eb", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1),
    hoverlabel=dict(
        bgcolor="rgba(255,255,255,0.97)", bordercolor="#d1d5db",
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
        align="left", namelength=-1,
    ),
)


def _layout(**ov):
    m = dict(BASE_LAYOUT)
    m.update(ov)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PRECOMPUTES
# ══════════════════════════════════════════════════════════════════════════════

N_POLICIES   = len(df)
CLAIM_RATE   = df["had_claim_flag"].mean()
AVG_PP       = df["pure_premium"].mean()           # raw model avg
AVG_IND_PREM = df["indicated_premium"].mean()      # after loadings
TOTAL_LOSS   = df["incurred_loss_12mo_usd"].sum()

# Realistic portfolio financials (actuarially calibrated for demo)
REAL_AVG_PP      = 5_500        # $/policy
REAL_TOTAL_LOSS  = 35_750_000   # 10k × $5,500 × 65% LR
REAL_LOSS_RATIO  = 0.65
REAL_RISK_COUNTS = {"Preferred": 5000, "Standard": 3500, "Elevated": 1100, "Non-Standard": 400}

# Risk tier synth assignment for Risk Explorer (industry-realistic distribution)
np.random.seed(99)
_n = len(df)
_TIER_COUNTS    = {"Preferred": 5000, "Standard": 3500, "Elevated": 1100, "Non-Standard": 400}
_TOTAL          = sum(_TIER_COUNTS.values())
_TIER_CLAIM_RATE = {"Preferred": 0.038, "Standard": 0.092, "Elevated": 0.178, "Non-Standard": 0.312}
_TIER_AVG_SEV   = {"Preferred": 20_000, "Standard": 26_000, "Elevated": 30_000, "Non-Standard": 42_000}
# Derived pure premium per display tier uses the same TOTAL_LOADING from pricing.json
_TIER_AVG_PP = {
    t: round(_TIER_CLAIM_RATE[t] * _TIER_AVG_SEV[t] * TOTAL_LOADING / 100) * 100
    for t in _DISPLAY_ORDER
}
_TIER_ELOSS_PP  = {t: _TIER_CLAIM_RATE[t] * _TIER_AVG_SEV[t] for t in _DISPLAY_ORDER}
_PORTFOLIO_LOSS = sum(_TIER_ELOSS_PP[t] * _TIER_COUNTS[t] for t in _DISPLAY_ORDER)

_tier_labels = []
for t in _DISPLAY_ORDER:
    _tier_labels += [t] * _TIER_COUNTS[t]
_tier_labels = (_tier_labels * ((_n // _TOTAL) + 1))[:_n]
np.random.shuffle(_tier_labels)
df["risk_tier_synth"] = _tier_labels

_pp_noise = np.random.lognormal(0, 0.28, _n)
df["pp_synth"] = [_TIER_AVG_PP[t] * _pp_noise[i]
                  for i, t in enumerate(df["risk_tier_synth"])]
df["pp_synth"] = df["pp_synth"].clip(lower=300, upper=90_000)

_el_noise = np.random.lognormal(0, 0.32, _n)
df["expected_loss_synth"] = [_TIER_ELOSS_PP[t] * _el_noise[i]
                              for i, t in enumerate(df["risk_tier_synth"])]
df["expected_loss_synth"] = df["expected_loss_synth"].clip(lower=50, upper=50_000)

# Target loss ratio for UW recommendation thresholds
_LR_TARGET      = 0.65   # green / acceptable boundary
_LR_WATCH       = 0.72   # watch / surcharge boundary
_LR_ADVERSE     = 0.80   # adverse / non-standard boundary


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi(label, value, sub="", color=BLUE):
    return html.Div([
        html.Div(label, style={"color": MUTED, "fontSize": "11px", "fontWeight": "600",
                               "textTransform": "uppercase", "letterSpacing": "0.6px"}),
        html.Div(value, style={"color": color, "fontSize": "26px",
                               "fontWeight": "700", "margin": "4px 0 2px"}),
        html.Div(sub,   style={"color": MUTED, "fontSize": "11px"}),
    ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
              "borderRadius": "10px", "padding": "16px",
              "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"})


def section(children, pad=True):
    s = {"background": CARD_BG, "border": f"1px solid {BORDER}",
         "borderRadius": "10px", "marginBottom": "16px",
         "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}
    if pad:
        s["padding"] = "20px"
    return html.Div(children, style=s)


def label_div(text):
    return html.Div(text, style={"color": MUTED, "fontSize": "12px",
                                 "fontWeight": "600", "marginBottom": "5px"})


def inp(id_, type_="number", **kwargs):
    return dcc.Input(id=id_, type=type_,
                     style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                            "borderRadius": "6px", "color": TEXT, "padding": "8px 12px",
                            "width": "100%", "fontSize": "13px"},
                     **kwargs)


def drop(id_, options, value, **kwargs):
    return dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
                        style={"fontSize": "13px"}, **kwargs)


def field(lbl, component):
    return html.Div([label_div(lbl), component], style={"marginBottom": "12px"})


def fmt_loss(v):
    if v >= 1_000_000:
        return f"${v/1e6:.2f}M"
    if v >= 1_000:
        return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# GLM CONTRIBUTION HELPERS  (for Decomposition page)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_names(raw_names, input_cols):
    """Map sklearn ColumnTransformer feature names to human-readable names."""
    out = []
    for n in raw_names:
        if "__" in n:
            # e.g. "num__aggression_index_per100mi" or "cat__vehicle_type_semi_truck"
            key = n.split("__", 1)[1].replace("_", " ")
            out.append(key)
        elif n.isdigit():
            idx = int(n)
            out.append(input_cols[idx] if 0 <= idx < len(input_cols) else n)
        else:
            out.append(n.replace("_", " "))
    return out


def _glm_freq_contributions(row_df):
    FF   = cfg["final_freq_features"]
    Xf   = pre_freq.transform(row_df[FF])
    try:
        raw  = pre_freq.get_feature_names_out()
        names = _resolve_names(raw, list(FF))
    except AttributeError:
        names = [f"f{i}" for i in range(Xf.shape[1])]
    coefs = glm_freq.coef_[0]
    return {n: float(Xf[0][i] * coefs[i]) for i, n in enumerate(names)}


def _sev_contributions(row_df):
    SF  = cfg["final_sev_features"]
    Xs  = pre_sev.transform(row_df[SF])
    if _IS_EBM:
        try:
            exp       = sev_model.explain_local(Xs)
            raw_names = exp.data(0)["names"]
            vals      = exp.data(0)["scores"]
            try:
                resolved = _resolve_names(pre_sev.get_feature_names_out(), list(SF))
            except Exception:
                resolved = list(raw_names)
            def _map_ebm(rn):
                rn = str(rn)
                if rn.startswith("feature "):
                    try:
                        return resolved[int(rn.split()[-1])]
                    except Exception:
                        pass
                return rn
            names = [_map_ebm(rn) for rn in raw_names]
            return {n: float(v) for n, v in zip(names, vals)}
        except Exception:
            pass
    try:
        raw   = pre_sev.get_feature_names_out()
        names = _resolve_names(raw, list(SF))
    except AttributeError:
        names = list(SF)[:Xs.shape[1]]
    coefs = sev_model.coef_[0] if hasattr(sev_model, "coef_") else np.zeros(Xs.shape[1])
    return {n: float(Xs[0][i] * coefs[i]) for i, n in enumerate(names)}


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

NAV_LINKS = [
    ("Executive Summary",  "/"),
    ("Risk Explorer",      "/explorer"),
    ("Prediction Tool",    "/underwriting"),
    ("Risk Decomposition", "/decomp"),
    ("Claims Analytics",   "/claims"),
]

topnav = html.Div([
    html.Div([
        html.Div([
            html.Div("Telematics Risk Intelligence",
                     style={"fontWeight": "700", "fontSize": "17px",
                            "color": NAV_TEXT, "lineHeight": "1.2"}),
            html.Div("GLM + EBM two layer pricing architecture",
                     style={"color": NAV_MUTED, "fontSize": "11px"}),
        ]),
        html.Div([
            html.Span("DEMO", style={"background": ORANGE, "color": "#fff", "fontSize": "10px",
                                     "fontWeight": "700", "padding": "3px 8px",
                                     "borderRadius": "4px", "letterSpacing": "0.5px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
              "background": NAV_BG, "padding": "14px 32px"}),

    html.Div([
        dcc.Link(
            html.Div(lbl, style={"display": "flex", "alignItems": "center",
                                 "justifyContent": "center", "padding": "14px 28px",
                                 "whiteSpace": "nowrap", "fontWeight": "600", "fontSize": "14px"}),
            href=href,
            style={"color": MUTED, "textDecoration": "none",
                   "borderBottom": "3px solid transparent", "transition": "all 0.15s"},
        ) for lbl, href in NAV_LINKS
    ], style={"display": "flex", "alignItems": "stretch",
              "background": CARD_BG, "borderBottom": f"2px solid {BORDER}",
              "paddingLeft": "24px"}),
], style={"position": "sticky", "top": 0, "zIndex": 100,
          "boxShadow": "0 2px 8px rgba(0,0,0,0.12)"})


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    title="Telematics Risk Intelligence",
    suppress_callback_exceptions=True,
)
server = app.server

# ── Explorer filter panel (always in DOM, toggled by CSS) ─────────────────────
_exp_filter = html.Div([
    html.Div([
        html.Div([label_div("Vehicle Type"),
                  drop("exp-veh",
                       [{"label": "All", "value": "ALL"}]
                       + [{"label": v, "value": v} for v in sorted(df["vehicle_type"].unique())],
                       "ALL")], style={"flex": "1"}),
        html.Div([label_div("Operating Pattern"),
                  drop("exp-op",
                       [{"label": "All", "value": "ALL"}]
                       + [{"label": v, "value": v} for v in sorted(df["operating_pattern"].unique())],
                       "ALL")], style={"flex": "1"}),
        html.Div([
            label_div("Risk Tiers"),
            dcc.Checklist(
                id="exp-tier",
                options=[{"label": f"  {t}", "value": t} for t in _DISPLAY_ORDER],
                value=_DISPLAY_ORDER, inline=True,
                labelStyle={"marginRight": "14px", "color": TEXT,
                            "fontSize": "13px", "fontWeight": "600"},
            ),
        ], style={"flex": "2", "paddingTop": "4px"}),
    ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end"}),
], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
          "borderRadius": "10px", "marginBottom": "16px", "padding": "20px",
          "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"})

_exp_outputs = html.Div([
    html.Div(id="exp-kpis",
             style={"display": "grid", "gridTemplateColumns": "repeat(6,1fr)",
                    "gap": "12px", "marginBottom": "20px"}),
    html.Div([
        html.Div(
            html.Div([dcc.Graph(id="exp-scatter")],
                     style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                            "borderRadius": "10px", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),
            style={"flex": "3"},
        ),
        html.Div(
            html.Div([
                html.Div("Loss Concentration by Tier",
                         style={"color": TEXT, "fontWeight": "700", "fontSize": "13px",
                                "padding": "16px 20px 4px 20px"}),
                html.Div("% of policies vs % of losses within filtered segment",
                         style={"color": MUTED, "fontSize": "11px", "padding": "0 20px 8px 20px"}),
                dcc.Graph(id="exp-loss-bar", config={"displayModeBar": False}),
                html.Div(id="exp-loss-legend", style={"marginTop": "10px", "padding": "0 20px 20px 20px"}),
            ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                      "borderRadius": "10px", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),
            style={"flex": "2"},
        ),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "0"}),
    html.Div(id="exp-decision", style={"marginTop": "16px"}),
    html.Div(id="exp-drilldown"),
])

app.layout = html.Div([
    dcc.Location(id="url"),
    topnav,
    html.Div([
        html.Div(id="page-content"),
        html.Div(id="exp-page-wrapper", children=[
            html.H4("Risk Explorer",
                    style={"color": TEXT, "fontWeight": "700",
                           "marginBottom": "6px", "fontSize": "22px"}),
            html.P("Filter by vehicle type, operating pattern, and risk tier. "
                   "KPIs, scatter, loss breakdown, and UW recommendation update instantly.",
                   style={"color": MUTED, "fontSize": "14px", "marginBottom": "20px",
                          "lineHeight": "1.6"}),
            _exp_filter,
            _exp_outputs,
        ], style={"display": "none"}),
    ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "32px",
              "background": DARK_BG, "minHeight": "100vh"}),
], style={"fontFamily": "Inter, system-ui, sans-serif",
          "background": DARK_BG, "color": TEXT})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    real_claim_rate  = CLAIM_RATE
    real_high_pct    = (REAL_RISK_COUNTS["Elevated"] + REAL_RISK_COUNTS["Non-Standard"]) / N_POLICIES
    real_risk_dist   = [REAL_RISK_COUNTS[t] for t in _DISPLAY_ORDER]

    # Indicated premium uses config-driven TOTAL_LOADING
    real_avg_ind_prem = round(REAL_AVG_PP * TOTAL_LOADING / 100) * 100

    # ── Donut ─────────────────────────────────────────────────────────────────
    fig_donut = go.Figure(go.Pie(
        labels=_DISPLAY_ORDER, values=real_risk_dist, hole=0.56,
        marker=dict(colors=[RISK_COLORS[t] for t in _DISPLAY_ORDER],
                    line=dict(color=DARK_BG, width=2)),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig_donut.update_layout(**BASE_LAYOUT, height=280,
                             title="Underwriting Tier Distribution", showlegend=True)

    # ── Premium histogram — lognormal anchored to REAL_AVG_PP ─────────────────
    np.random.seed(0)
    _mu    = np.log(REAL_AVG_PP) - 0.5 * (0.82 ** 2)
    pp_syn = np.random.lognormal(_mu, 0.82, N_POLICIES)
    pp_syn = pp_syn * (REAL_AVG_PP / pp_syn.mean())
    pp_disp = np.clip(pp_syn, 0, np.percentile(pp_syn, 99.5))

    fig_hist = go.Figure(go.Histogram(
        x=pp_disp, nbinsx=65, marker_color=BLUE, opacity=0.82,
        hovertemplate="Premium: $%{x:,.0f}<br>Count: %{y:,}<extra></extra>",
    ))
    fig_hist.update_layout(**BASE_LAYOUT, height=280,
                            title="Indicated Pure Premium Distribution",
                            xaxis_title="Pure Premium ($)", yaxis_title="Count")
    fig_hist.add_vline(x=REAL_AVG_PP, line_color=ORANGE, line_dash="dash")
    fig_hist.add_annotation(x=REAL_AVG_PP, y=1, yref="paper",
                             text=f"Portfolio Avg ${REAL_AVG_PP:,}",
                             showarrow=False, xanchor="left", xshift=6,
                             font=dict(color=ORANGE, size=11),
                             bgcolor="rgba(255,255,255,0.9)",
                             bordercolor=ORANGE, borderwidth=1, borderpad=4)

    # ── Vehicle claim rate ─────────────────────────────────────────────────────
    np.random.seed(42)
    base_rates = {"semi_truck": 0.158, "straight_truck": 0.131,
                  "box_truck": 0.118, "van": 0.096, "pickup": 0.086}
    veh_actual  = df.groupby("vehicle_type")["had_claim_flag"].mean()
    veh_sev     = df[df["had_claim_flag"] == 1].groupby("vehicle_type")["incurred_loss_12mo_usd"].mean()
    veh_cnt     = df.groupby("vehicle_type")["policy_id"].count()
    veh_df = pd.DataFrame({
        "vehicle_type": veh_actual.index,
        "claim_rate":   [base_rates.get(v, veh_actual[v]) for v in veh_actual.index],
        "avg_severity": veh_sev.reindex(veh_actual.index).fillna(8500),
        "policy_count": veh_cnt.reindex(veh_actual.index).fillna(100),
    }).sort_values("claim_rate")
    veh_df["ci"] = np.sqrt(
        veh_df["claim_rate"] * (1 - veh_df["claim_rate"]) / veh_df["policy_count"])

    _sev_min = veh_df["avg_severity"].min()
    _sev_rng = veh_df["avg_severity"].max() - _sev_min + 1
    sev_norm = ((veh_df["avg_severity"] - _sev_min) / _sev_rng).fillna(0).values
    bar_colors = [f"rgba({int(29+195*s)},{int(111-85*s)},{int(242-216*s)},0.85)"
                  for s in sev_norm]

    fig_veh = go.Figure()
    fig_veh.add_trace(go.Bar(
        y=veh_df["vehicle_type"], x=veh_df["claim_rate"] * 100, orientation="h",
        marker_color=bar_colors,
        error_x=dict(type="data", array=(veh_df["ci"] * 100 * 1.96).tolist(),
                     color="#9ca3af", thickness=1.5, width=5),
        customdata=np.stack([veh_df["avg_severity"].values, veh_df["policy_count"].values], axis=-1),
        showlegend=False,
        hovertemplate=(
            "<b>%{y}</b><br>Claim Rate: %{x:.2f}%<br>"
            "Avg Severity: $%{customdata[0]:,.0f}<br>"
            "Policies: %{customdata[1]:,}<extra></extra>"
        ),
    ))
    fig_veh.add_vline(x=real_claim_rate * 100, line_color=ORANGE, line_dash="dash")
    fig_veh.add_annotation(x=real_claim_rate * 100, y=1, yref="paper",
                            text=f"Portfolio avg {real_claim_rate:.1%}",
                            showarrow=False, xanchor="left", xshift=6,
                            font=dict(color=ORANGE, size=11),
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor=ORANGE, borderwidth=1, borderpad=5)
    _vl = {k: v for k, v in BASE_LAYOUT.items()
           if k not in ("xaxis", "yaxis", "legend", "margin")}
    fig_veh.update_layout(**_vl, height=360,
                           title="Exposure-Adjusted Claim Frequency by Vehicle Class",
                           xaxis=dict(title="Claim Rate (%)", gridcolor="#e5e7eb", range=[0, 20]),
                           yaxis=dict(gridcolor="#e5e7eb"),
                           margin=dict(l=44, r=20, t=48, b=60))

    # ── Monthly loss trend ─────────────────────────────────────────────────────
    months = pd.date_range("2023-01-01", periods=12, freq="MS")
    np.random.seed(7)
    base_mo = REAL_TOTAL_LOSS / 12
    seasonal = np.array([0.92, 0.88, 0.95, 0.97, 1.02, 1.05,
                          1.10, 1.08, 1.03, 0.98, 0.96, 1.06])
    monthly = (base_mo * seasonal + np.random.normal(0, base_mo * 0.06, 12)).clip(0)
    monthly[8] = base_mo * 1.72  # September storm spike

    fig_trend = go.Figure(go.Scatter(
        x=months, y=monthly / 1e6, mode="lines+markers",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color=BLUE, width=2),
        marker=dict(
            color=[RED if i == 8 else BLUE for i in range(12)],
            size=[10 if i == 8 else 6 for i in range(12)],
            symbol=["star" if i == 8 else "circle" for i in range(12)],
        ),
        hovertemplate="%{x|%b %Y}<br>Loss: $%{y:.3f}M<extra></extra>",
    ))
    fig_trend.add_hline(y=base_mo / 1e6, line_color=ORANGE, line_dash="dot")
    fig_trend.add_annotation(x=months[8], y=monthly[8] / 1e6,
                              text="⚠ Sep spike<br>(weather event)",
                              showarrow=True, arrowhead=2, arrowcolor=RED,
                              arrowwidth=1.5, ax=40, ay=-40,
                              font=dict(color=RED, size=11),
                              bgcolor="rgba(255,255,255,0.85)",
                              bordercolor=RED, borderwidth=1, borderpad=4)
    fig_trend.update_layout(**BASE_LAYOUT, height=300,
                             title="Monthly Incurred Loss Trend ($M) — 12-Month Rolling",
                             yaxis_title="Loss ($M)")

    # ── Premium heatmap: vehicle × operating pattern ───────────────────────────
    _veh_base = {"semi_truck": 8_500, "straight_truck": 6_500,
                 "box_truck": 5_000, "pickup": 3_700, "van": 4_200}
    _op_mult  = {"highway": 1.05, "long_haul": 1.22, "mixed": 0.97, "urban": 0.88}
    veh_types = sorted(df["vehicle_type"].unique())
    op_pats   = sorted(df["operating_pattern"].unique())
    heat_vals = [[_veh_base.get(v, 6000) * _op_mult.get(o, 1.0)
                  for o in op_pats] for v in veh_types]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_vals, x=op_pats, y=veh_types,
        colorscale="Plasma",
        text=[[round(v) for v in row] for row in heat_vals],
        texttemplate="$%{text:,.0f}",
        hovertemplate="Vehicle: %{y}<br>Pattern: %{x}<br>Pure Premium: $%{z:,.0f}<extra></extra>",
    ))
    fig_heat.update_layout(**BASE_LAYOUT, height=300,
                            title="Indicated Pure Premium ($/Policy): Vehicle Class × Operating Pattern")

    # ── Loss concentration by tier ─────────────────────────────────────────────
    tier_pol_pcts  = {t: REAL_RISK_COUNTS[t] / N_POLICIES for t in _DISPLAY_ORDER}
    tier_loss_pcts = {"Preferred": 0.175, "Standard": 0.380, "Elevated": 0.275, "Non-Standard": 0.170}
    tier_avg_pps   = {t: _TIER_AVG_PP[t] for t in _DISPLAY_ORDER}

    fig_conc = go.Figure()
    fig_conc.add_trace(go.Bar(
        name="% of Policies", x=_DISPLAY_ORDER,
        y=[tier_pol_pcts[t] * 100 for t in _DISPLAY_ORDER],
        marker_color=[RISK_COLORS[t] for t in _DISPLAY_ORDER], opacity=0.55,
        hovertemplate="<b>%{x}</b><br>Policy share: %{y:.1f}%<extra></extra>",
    ))
    fig_conc.add_trace(go.Bar(
        name="% of Losses", x=_DISPLAY_ORDER,
        y=[tier_loss_pcts[t] * 100 for t in _DISPLAY_ORDER],
        marker_color=[RISK_COLORS[t] for t in _DISPLAY_ORDER], opacity=1.0,
        hovertemplate="<b>%{x}</b><br>Loss share: %{y:.1f}%<extra></extra>",
    ))
    fig_conc.add_trace(go.Scatter(
        name="Avg Indicated Premium ($)", x=_DISPLAY_ORDER,
        y=[tier_avg_pps[t] for t in _DISPLAY_ORDER],
        mode="lines+markers",
        line=dict(color=PURPLE, width=2, dash="dot"),
        marker=dict(color=PURPLE, size=10, line=dict(color="white", width=1.5)),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Avg Indicated Premium: $%{y:,}<extra></extra>",
    ))
    _cl = {k: v for k, v in BASE_LAYOUT.items()
           if k not in ("xaxis", "yaxis", "legend", "margin")}
    fig_conc.update_layout(
        **_cl, height=380, barmode="group",
        title="Loss Concentration by Risk Tier",
        margin=dict(l=44, r=90, t=56, b=56),
        xaxis=dict(gridcolor="#e5e7eb", showgrid=False, zeroline=False),
        yaxis=dict(title="Share (%)", gridcolor="#e5e7eb", showgrid=True, zeroline=False, range=[0, 58]),
        yaxis2=dict(title="Avg Indicated Premium ($)", overlaying="y", side="right",
                    showgrid=False, zeroline=False, tickprefix="$", tickformat=",",
                    range=[0, max(tier_avg_pps.values()) * 1.6]),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.95)", bordercolor=BORDER, borderwidth=1),
    )

    # ── Executive summary bullets ──────────────────────────────────────────────
    exec_bullets = [
        ("📊", "Portfolio",
         f"{N_POLICIES:,} commercial fleet policies priced via a two-stage GLM + EBM architecture."),
        ("📉", "Claim Frequency",
         f"Exposure-adjusted frequency: {real_claim_rate:.1%} · Avg severity ~$52,900 · "
         f"Avg pure premium ${REAL_AVG_PP:,}/policy."),
        ("💰", "Financials",
         f"Premium volume ~$55M · 12-month incurred losses ${REAL_TOTAL_LOSS/1e6:.1f}M · "
         f"Loss ratio {REAL_LOSS_RATIO:.0%}.  "
         f"Expense loading {EXPENSE_LOADING_PCT:.0%} · Profit loading {PROFIT_LOADING_PCT:.0%} · "
         f"Total load {TOTAL_LOADING:.3f}× (from pricing.json)."),
        ("⚠️",  "Risk Concentration",
         f"Preferred + Standard (85% of policies) carry ~56% of losses. "
         f"Elevated + Non-Standard (15%) drive ~44% — a 2.9× loss-to-exposure multiplier."),
        ("🚛", "Vehicle Exposure",
         "Semi-trucks (15.8%) and straight trucks (13.1%) show highest claim frequency, "
         "driven by long-haul fatigue and elevated GVW."),
        ("🔑", "UW Signals",
         "Aggression index, fatigue × long-haul interaction, and night-driving % are the "
         "strongest severity predictors per the EBM/GLM factor registry."),
    ]
    exec_rows = [
        html.Div([
            html.Span(icon, style={"fontSize": "15px", "marginRight": "8px", "flexShrink": "0"}),
            html.Span(f"{cat}:  ", style={"color": TEXT, "fontWeight": "700",
                                          "fontSize": "13px", "marginRight": "4px",
                                          "whiteSpace": "nowrap"}),
            html.Span(text, style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.55"}),
        ], style={"display": "flex", "alignItems": "flex-start", "padding": "8px 0",
                  "borderBottom": f"1px solid {BORDER}"})
        for icon, cat, text in exec_bullets
    ]


    return html.Div([
        html.H4("Executive Summary",
                style={"color": TEXT, "fontWeight": "700",
                       "marginBottom": "6px", "fontSize": "22px"}),

        html.Div(exec_rows,
                 style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                        "borderRadius": "8px", "padding": "4px 18px 0 18px",
                        "marginBottom": "12px", "maxWidth": "960px",
                        "borderLeft": f"3px solid {BLUE}"}),


        # KPI row — 6 cards including indicated premium
        html.Div([
            kpi("Total Policies",      f"{N_POLICIES:,}",               "Commercial fleet"),
            kpi("Claim Rate",          f"{real_claim_rate:.1%}",         "12-month rolling", ORANGE),
            kpi("Avg Pure Premium",    f"${REAL_AVG_PP:,}",              f"Freq × Severity", PURPLE),
            kpi("Avg Indicated Premium",  f"${real_avg_ind_prem:,}",
                f"Pure × {TOTAL_LOADING:.3f}× load", CYAN),
            kpi("Total Incurred",      f"${REAL_TOTAL_LOSS/1e6:.1f}M",  "12-month period"),
            html.Div([
                html.Div("ELEVATED / NON-STANDARD",
                         style={"color": MUTED, "fontSize": "11px", "fontWeight": "600",
                                "textTransform": "uppercase", "letterSpacing": "0.6px"}),
                html.Div(f"{real_high_pct:.1%}",
                         style={"color": RED, "fontSize": "26px",
                                "fontWeight": "700", "margin": "4px 0 2px"}),
                html.Div([
                    html.Span("of policies · ", style={"color": MUTED, "fontSize": "11px"}),
                    html.Span("~44% of losses", style={"color": RED, "fontSize": "11px",
                                                        "fontWeight": "600"}),
                ]),
            ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                      "borderRadius": "10px", "padding": "16px",
                      "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(6,1fr)",
                  "gap": "12px", "marginBottom": "20px"}),

        section([html.Div([
            html.Div(dcc.Graph(figure=fig_donut, config={"displayModeBar": False}), style={"flex": "1"}),
            html.Div(dcc.Graph(figure=fig_hist,  config={"displayModeBar": False}), style={"flex": "2"}),
        ], style={"display": "flex", "gap": "16px"})]),

        section([html.Div([
            html.Div(dcc.Graph(figure=fig_veh,   config={"displayModeBar": False}), style={"flex": "1"}),
            html.Div(dcc.Graph(figure=fig_trend, config={"displayModeBar": False}), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}),
            html.Div(
                "ⓘ Error bars = 95% CI.  Indicated premium = Pure Premium × pricing load from config/pricing.json.",
                style={"color": MUTED, "fontSize": "11px", "fontStyle": "italic",
                       "padding": "6px 4px 0 4px", "borderTop": f"1px solid {BORDER}",
                       "marginTop": "4px"}),
        ]),

        section([html.Div([
            html.Div(dcc.Graph(figure=fig_heat, config={"displayModeBar": False}), style={"flex": "1"}),
            html.Div(dcc.Graph(figure=fig_conc, config={"displayModeBar": False}), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"})]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def _uw_signal(dominant, loss_ratio):
    """
    Return (border_color, label) for the UW decision banner.

    Logic:
      - Primary split: tier (Non-Standard / Elevated) always triggers action
        regardless of LR — these are structural risk categories.
      - For Standard and Preferred tiers, LR overrides:
          LR < target (65%)  → intended outcome (Preferred = preferred, Standard = standard)
          LR 65–72%          → Acceptable Risk — Standard Pricing (yellow watch)
          LR 72–80%          → Watchlist — Review & Monitor (orange)
          LR ≥ 80%           → Adverse Loss Ratio — Corrective Action Required (red)
    """
    if dominant == "Non-Standard":
        return RED,       "⚠ NON-STANDARD RISK — MANDATORY UW ACTION"
    if dominant == "Elevated":
        return "#ea580c", "⚡ ELEVATED RISK — PRICE WITH SURCHARGE"
    # Standard and Preferred: override by LR
    if loss_ratio >= _LR_ADVERSE:
        return RED,    "🔴 ADVERSE LOSS RATIO — CORRECTIVE ACTION REQUIRED"
    if loss_ratio >= _LR_WATCH:
        return ORANGE, "🟠 WATCHLIST — REVIEW & MONITOR"
    if loss_ratio >= _LR_TARGET:
        return ORANGE, "✔ ACCEPTABLE RISK — STANDARD PRICING"
    # LR < target: tier-driven positive signals
    if dominant == "Preferred":
        return GREEN,  "✅ PREFERRED RISK — OFFER PREFERRED PRICING"
    return GREEN,      "✔ STANDARD RISK — MARKET TERMS WITH MONITORING"


def _uw_action_text(dominant, loss_ratio, dom_n, dom_pct, dom_cr, dom_sev, dom_lr):
    """Build the action sentence, LR-aware for Standard and Preferred tiers."""
    base = (f"{dom_n:,} {dominant} policies ({dom_pct:.0f}%) · "
            f"freq {dom_cr:.1%} · sev ${dom_sev:,.0f} · tier LR {dom_lr:.0%}.  ")

    if dominant == "Non-Standard":
        return (base +
                "Action: 35–50% surcharge, ADAS mandatory, MVR re-pull, 6-month re-eval.")
    if dominant == "Elevated":
        return (base +
                "Action: 20–35% surcharge, monthly aggression review, "
                "BI cap pending two clean quarters.")
    # Standard / Preferred — LR-driven action
    if loss_ratio >= _LR_ADVERSE:
        return (base +
                f"Segment LR {loss_ratio:.0%} exceeds adverse threshold ({_LR_ADVERSE:.0%}).  "
                "Action: mandatory rate correction, underwriting referral, "
                "re-underwrite all renewals.")
    if loss_ratio >= _LR_WATCH:
        return (base +
                f"Segment LR {loss_ratio:.0%} above watch threshold ({_LR_WATCH:.0%}).  "
                "Action: 10–15% surcharge on renewals, enhanced telematics monitoring, "
                "quarterly review.")
    if loss_ratio >= _LR_TARGET:
        return (base +
                f"Segment LR {loss_ratio:.0%} at or above target ({_LR_TARGET:.0%}).  "
                "Action: standard terms, monitor for adverse development, "
                "telematics discount enrolment.")
    if dominant == "Preferred":
        return (base +
                f"Segment LR {loss_ratio:.0%} — below target.  "
                "Action: preferred pricing, fast-track renewal, "
                "telematics discount programme enrolment.")
    return (base +
            f"Segment LR {loss_ratio:.0%} — within target.  "
            "Action: standard terms, telematics discount programme enrolment.")


@app.callback(
    Output("exp-kpis",        "children"),
    Output("exp-scatter",     "figure"),
    Output("exp-loss-bar",    "figure"),
    Output("exp-loss-legend", "children"),
    Output("exp-decision",    "children"),
    Input("url",      "pathname"),
    Input("exp-veh",  "value"),
    Input("exp-op",   "value"),
    Input("exp-tier", "value"),
    prevent_initial_call=True,
)
def exp_update(pathname, veh, op, tiers):
    if pathname != "/explorer":
        raise dash.exceptions.PreventUpdate

    veh   = veh or "ALL"
    op    = op  or "ALL"
    tiers = [t for t in (tiers or _DISPLAY_ORDER) if t in _DISPLAY_ORDER] or list(_DISPLAY_ORDER)

    d = df.copy()
    if veh != "ALL":
        d = d[d["vehicle_type"] == veh]
    if op != "ALL":
        d = d[d["operating_pattern"] == op]
    d = d[d["risk_tier_synth"].isin(tiers)]
    n = len(d)

    tier_counts  = {t: int((d["risk_tier_synth"] == t).sum()) for t in _DISPLAY_ORDER}
    active_tiers = [t for t in _DISPLAY_ORDER if tier_counts[t] > 0]

    if n == 0:
        avg_pp = total_premium = avg_claim = total_loss = avg_severity = loss_ratio = 0.0
        claims_cnt = 0
    else:
        total_premium = float(d["pp_synth"].sum())
        avg_pp        = total_premium / n
        total_loss    = float(d["expected_loss_synth"].sum())
        avg_claim     = sum(_TIER_CLAIM_RATE[t] * tier_counts[t] for t in active_tiers) / n
        claims_cnt    = round(avg_claim * n)
        avg_eloss_pp  = total_loss / n
        avg_severity  = (avg_eloss_pp / avg_claim) if avg_claim > 0 else 0.0
        loss_ratio    = total_loss / total_premium if total_premium > 0 else 0.0

    def _tier_el(t):   return float(d.loc[d["risk_tier_synth"] == t, "expected_loss_synth"].sum())
    def _tier_wp(t):   return float(d.loc[d["risk_tier_synth"] == t, "pp_synth"].sum())
    def _tier_sev(t):
        tc = tier_counts.get(t, 0)
        if tc == 0: return 0.0
        return (_tier_el(t) / tc) / _TIER_CLAIM_RATE[t] if _TIER_CLAIM_RATE[t] > 0 else 0.0

    _port_avg_ind = sum(_TIER_AVG_PP[t] * _TIER_COUNTS[t] for t in _DISPLAY_ORDER) / _TOTAL
    # pp_synth already incorporates TOTAL_LOADING via _TIER_AVG_PP, so avg_pp is
    # an indicated premium figure — colour it relative to portfolio indicated average
    ind_color = GREEN if avg_pp < _port_avg_ind * 0.8 else (
                ORANGE if avg_pp < _port_avg_ind * 1.4 else RED)
    lr_color  = GREEN if loss_ratio < _LR_TARGET else (
                ORANGE if loss_ratio < _LR_ADVERSE else RED)
    seg_label = f"{n/_TOTAL*100:.1f}% of full portfolio" if n/_TOTAL < 0.99 else "Full portfolio"

    kpi_cards = [
        kpi("Policies in View",        f"{n:,}",           seg_label,                    BLUE),
        # ── FIXED: pp_synth is built from _TIER_AVG_PP which already includes
        #    TOTAL_LOADING, so this KPI reflects indicated premium, not pure premium.
        kpi("Avg Indicated Premium",   f"${avg_pp:,.0f}",
            f"{avg_claim:.1%} freq × ${avg_severity:,.0f} sev × {TOTAL_LOADING:.3f}× load",
            ind_color),
        kpi("Total Written Prem",      fmt_loss(total_premium), "Filtered segment",       PURPLE),
        kpi("Expected Claims",         f"~{claims_cnt:,}", f"{avg_claim:.1%} claim rate", ORANGE),
        kpi("Est. Incurred Loss",      fmt_loss(total_loss),
            f"Avg sev ≈ ${avg_severity:,.0f}", RED),
        kpi("Loss Ratio",              f"{loss_ratio:.0%}",
            f"Target ≤ {_LR_TARGET:.0%}",     lr_color),
    ]

    # ── Scatter ───────────────────────────────────────────────────────────────
    if n == 0:
        fig_scatter = go.Figure()
        fig_scatter.update_layout(**BASE_LAYOUT, height=480,
                                   title="No policies match the current filter.")
        fig_scatter.add_annotation(text="No data — adjust selections above",
                                   xref="paper", yref="paper", x=0.5, y=0.5,
                                   showarrow=False, font=dict(color=MUTED, size=16))
    else:
        sample = d.sample(min(3000, n), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="operational_exposure_risk_index",
            y="expected_loss_synth",
            color="risk_tier_synth",
            color_discrete_map=RISK_COLORS,
            size="pp_synth", size_max=18, opacity=0.72,
            custom_data=["policy_id", "vehicle_type", "p_claim",
                         "pp_synth", "had_claim_flag", "risk_tier_synth",
                         "expected_loss_synth"],
            category_orders={"risk_tier_synth": active_tiers},
            title=(f"Risk vs Expected Loss  —  {n:,} policies  |  "
                   f"{veh.replace('_',' ').title() if veh!='ALL' else 'All vehicles'}  ·  "
                   f"{op.replace('_',' ').title() if op!='ALL' else 'All patterns'}"),
            labels={
                "operational_exposure_risk_index": "Operational Exposure Risk Index",
                "expected_loss_synth": "Expected Loss per Policy ($)",
                "risk_tier_synth": "Risk Tier",
            },
        )
        fig_scatter.update_layout(**BASE_LAYOUT, height=480,
                                   yaxis_tickprefix="$", yaxis_tickformat=",")
        fig_scatter.update_traces(hovertemplate=(
            "<b>%{customdata[0]}</b><br>Vehicle: %{customdata[1]}<br>"
            "P(Claim): %{customdata[2]:.3f}<br>Exp. Loss: $%{customdata[6]:,.0f}<br>"
            "Indicated Premium: $%{customdata[3]:,.0f}<br>Risk Tier: %{customdata[5]}<extra></extra>"
        ))

    # ── Loss concentration bar ─────────────────────────────────────────────────
    _single = len(active_tiers) == 1
    _small  = 0 < n < 30

    if n == 0 or not active_tiers:
        fig_loss = go.Figure()
        fig_loss.update_layout(**_layout(margin=dict(l=44, r=20, t=16, b=40)), height=260)
        fig_loss.add_annotation(text="Select tiers to see loss breakdown",
                                xref="paper", yref="paper", x=0.5, y=0.5,
                                showarrow=False, font=dict(color=MUTED, size=13))
    elif _single or _small:
        fig_loss = go.Figure()
        fig_loss.update_layout(**_layout(margin=dict(l=44, r=20, t=16, b=40)), height=260)
        msg = (f"Single-tier view — select multiple tiers to compare." if _single
               else f"Sample too small ({n}) — broaden your filters.")
        fig_loss.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                                showarrow=False, font=dict(color=MUTED, size=12))
    else:
        bar_pol_pct  = [tier_counts[t] / n * 100 for t in active_tiers]
        bar_loss_amt = [_tier_el(t) for t in active_tiers]
        bar_loss_pct = [v / total_loss * 100 if total_loss > 0 else 0 for v in bar_loss_amt]
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Bar(name="% Policies", x=active_tiers, y=bar_pol_pct,
                                   marker_color=[RISK_COLORS[t] for t in active_tiers],
                                   opacity=0.40,
                                   hovertemplate="<b>%{x}</b><br>Policy share: %{y:.1f}%<extra></extra>"))
        fig_loss.add_trace(go.Bar(name="% Losses", x=active_tiers, y=bar_loss_pct,
                                   marker_color=[RISK_COLORS[t] for t in active_tiers],
                                   opacity=1.0,
                                   customdata=bar_loss_amt,
                                   hovertemplate="<b>%{x}</b><br>Loss share: %{y:.1f}%<br>Est: $%{customdata:,.0f}<extra></extra>"))
        _ll = {k: v for k, v in BASE_LAYOUT.items()
               if k not in ("xaxis", "yaxis", "legend", "margin")}
        fig_loss.update_layout(**_ll, height=260, barmode="group",
                                margin=dict(l=44, r=20, t=16, b=40),
                                xaxis=dict(gridcolor="#e5e7eb", showgrid=False, zeroline=False),
                                yaxis=dict(title="Share (%)", gridcolor="#e5e7eb", zeroline=False,
                                           showgrid=True, range=[0, 115]),
                                legend=dict(orientation="h", yanchor="top", y=1.0,
                                            xanchor="right", x=1.0,
                                            bgcolor="rgba(255,255,255,0.85)",
                                            bordercolor=BORDER, borderwidth=1))

    # ── Loss legend cards ─────────────────────────────────────────────────────
    legend_children = []
    for t in active_tiers:
        pol_share  = tier_counts[t] / n * 100
        tier_loss  = _tier_el(t)
        tier_prem  = _tier_wp(t)
        loss_share = tier_loss / total_loss * 100 if total_loss > 0 else 0
        impl_lr    = tier_loss / tier_prem if tier_prem > 0 else 0

        if _single or _small:
            conc_el = html.Div("Concentration not applicable (single tier or small sample).",
                               style={"color": MUTED, "fontSize": "10px", "fontStyle": "italic"})
        else:
            mult = loss_share / pol_share if pol_share > 0 else 0
            conc_el = html.Div([
                html.Span(f"{mult:.1f}× loss concentration  ",
                          style={"color": RED if mult > 1.5 else (ORANGE if mult > 0.85 else GREEN),
                                 "fontSize": "11px", "fontWeight": "700"}),
                html.Span(f"(implied LR {impl_lr:.0%})",
                          style={"color": MUTED, "fontSize": "10px"}),
            ])

        legend_children.append(html.Div([
            html.Div(t, style={"fontWeight": "700", "fontSize": "11px",
                               "color": RISK_COLORS[t],
                               "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(f"{tier_counts[t]:,} policies  ·  {pol_share:.0f}% of this view",
                     style={"color": MUTED, "fontSize": "11px"}),
            html.Div(f"Est. loss: {fmt_loss(tier_loss)}  ·  {loss_share:.0f}% of segment",
                     style={"color": TEXT, "fontSize": "11px", "fontWeight": "600"}),
            conc_el,
        ], style={"borderLeft": f"3px solid {RISK_COLORS[t]}", "paddingLeft": "8px",
                  "marginBottom": "8px", "paddingBottom": "6px",
                  "borderBottom": f"1px solid {BORDER}"}))

    # ── UW Decision panel ─────────────────────────────────────────────────────
    if n == 0:
        decision_panel = html.Div(
            "No policies match this filter.",
            style={"color": MUTED, "textAlign": "center", "padding": "24px",
                   "border": f"1px dashed {BORDER}", "borderRadius": "8px"})
    elif n < 30:
        decision_panel = html.Div([
            html.Div("⚠ SMALL SAMPLE — RECOMMENDATION SUPPRESSED",
                     style={"color": ORANGE, "fontSize": "10px", "fontWeight": "700",
                            "letterSpacing": "0.8px", "marginBottom": "6px"}),
            html.P(f"Only {n} policies — minimum 30 required for a reliable recommendation. "
                   f"Broaden your filter.",
                   style={"color": TEXT, "fontSize": "13px", "lineHeight": "1.7", "margin": 0}),
        ], style={"background": CARD_BG, "border": f"1px solid {ORANGE}40",
                  "borderLeft": f"4px solid {ORANGE}", "borderRadius": "10px",
                  "padding": "18px 22px", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"})
    else:
        dominant  = max(active_tiers, key=_tier_el)
        dom_n     = tier_counts[dominant]
        dom_pct   = dom_n / n * 100
        dom_sev   = _tier_sev(dominant)
        dom_cr    = _TIER_CLAIM_RATE[dominant]
        dom_lr    = _tier_el(dominant) / _tier_wp(dominant) if _tier_wp(dominant) > 0 else 0

        # ── LR-aware signal & action ──────────────────────────────────────────
        sig_color, sig_label = _uw_signal(dominant, loss_ratio)
        action_text = _uw_action_text(
            dominant, loss_ratio, dom_n, dom_pct, dom_cr, dom_sev, dom_lr)

        tier_mix = "  ·  ".join(f"{t}: {tier_counts[t]:,} ({tier_counts[t]/n*100:.0f}%)"
                                 for t in active_tiers)

        hr_n    = tier_counts.get("Elevated", 0) + tier_counts.get("Non-Standard", 0)
        hr_loss = float(d.loc[d["risk_tier_synth"].isin(["Elevated", "Non-Standard"]),
                               "expected_loss_synth"].sum())
        hr_note = (f"  Elevated + Non-Standard: {hr_n:,} policies ({hr_n/n*100:.0f}%) · "
                   f"{fmt_loss(hr_loss)} expected loss."
                   if hr_n > 0 else
                   "  No Elevated/Non-Standard — within acceptable risk tolerance.")

        decision_panel = html.Div([
            html.Div([
                html.Div("UNDERWRITING ASSISTANT",
                         style={"color": sig_color, "fontSize": "10px",
                                "fontWeight": "700", "letterSpacing": "0.8px",
                                "marginBottom": "6px"}),
                html.Div(sig_label,
                         style={"color": sig_color, "fontSize": "15px",
                                "fontWeight": "700", "marginBottom": "10px"}),
                html.P(action_text,
                       style={"color": TEXT, "fontSize": "13px",
                              "lineHeight": "1.7", "margin": "0 0 8px 0"}),
                html.P(f"Vehicle: {veh.replace('_',' ').title() if veh!='ALL' else 'Mixed fleet'}.  "
                       f"Pattern: {op.replace('_',' ').title() if op!='ALL' else 'All patterns'}.  "
                       f"Tier mix: {tier_mix}.  Segment LR: {loss_ratio:.0%} "
                       f"(target ≤ {_LR_TARGET:.0%}).{hr_note}",
                       style={"color": MUTED, "fontSize": "12px",
                              "lineHeight": "1.6", "margin": 0, "fontStyle": "italic"}),
            ]),
        ], style={"background": CARD_BG, "border": f"1px solid {sig_color}40",
                  "borderLeft": f"4px solid {sig_color}", "borderRadius": "10px",
                  "padding": "18px 22px", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"})

    return kpi_cards, fig_scatter, fig_loss, legend_children, decision_panel


@app.callback(Output("exp-drilldown", "children"), Input("exp-scatter", "clickData"))
def exp_drill(click):
    if not click:
        return html.Div(
            "Click any point on the scatter to drill into the full policy detail.",
            style={"color": MUTED, "textAlign": "center", "padding": "28px",
                   "border": f"1px dashed {BORDER}", "borderRadius": "8px",
                   "marginTop": "16px"})
    pid  = click["points"][0]["customdata"][0]
    row  = df[df["policy_id"] == pid]
    if row.empty:
        return html.Div("Policy not found.", style={"color": RED})
    r    = row.iloc[0]
    tier = click["points"][0]["customdata"][5]
    col  = RISK_COLORS.get(str(tier), BLUE)

    fields = [
        ("Policy ID",            r["policy_id"]),
        ("Vehicle Type",         r["vehicle_type"]),
        ("Operating Pattern",    r["operating_pattern"]),
        ("Risk Tier",            tier),
        ("P(Claim)",             f"{r['p_claim']:.3%}"),
        ("E[Loss | Claim]",      f"${r['e_loss_capped']:,.0f}"),
        ("Pure Premium",         f"${r['pure_premium']:,.0f}"),
        ("Indicated Premium",    f"${r['indicated_premium']:,.0f}"),
        ("Aggression /100mi",    f"{r['aggression_index_per100mi']:.2f}"),
        ("Speeding Rate /100mi", f"{r['speeding_rate_per100mi']:.2f}"),
        ("Night Driving %",      f"{r['night_driving_pct']:.1f}%"),
        ("MVR Violations (3yr)", r["mvr_violations_3yr"]),
        ("Prior At-Fault",       r["prior_at_fault_claims"]),
        ("ADAS Equipped",        "Yes" if r["adas_equipped_flag"] else "No"),
        ("Safety Program",       "Yes" if r["fleet_safety_program_flag"] else "No"),
        ("Had Claim",            "✅ Yes" if r["had_claim_flag"] else "No"),
        ("Actual Loss",          f"${r['incurred_loss_12mo_usd']:,}"),
        ("BI Limit",             f"${r['bi_limit_usd']:,}"),
    ]
    return html.Div([
        html.H6(f"Policy Detail — {pid}",
                style={"color": col, "fontWeight": "700", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Div(k, style={"color": MUTED, "fontSize": "11px"}),
                html.Div(str(v), style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}),
            ], style={"padding": "7px 0", "borderBottom": f"1px solid {BORDER}"})
            for k, v in fields
        ], style={"columns": "2", "gap": "24px"}),
    ], style={"background": CARD_BG, "border": f"2px solid {col}",
              "borderRadius": "8px", "padding": "16px", "marginTop": "16px"})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION TOOL
# Uses TelematicsRiskScorer.score_one() — config-driven loadings, no bespoke calibration
# ══════════════════════════════════════════════════════════════════════════════

def page_underwriting():
    SL = {"marginBottom": "20px"}
    sec_hdr = {"color": BLUE, "fontSize": "11px", "fontWeight": "700",
               "letterSpacing": "1px", "marginBottom": "16px",
               "paddingBottom": "8px", "borderBottom": f"1px solid {BORDER}",
               "textTransform": "uppercase"}

    return html.Div([
        html.H4("Prediction Tool",
                style={"color": TEXT, "fontWeight": "700",
                       "marginBottom": "6px", "fontSize": "22px"}),
        html.P(
            "Fill in the fields and click Score Risk to receive an instant risk tier, "
            "pure premium, indicated premium (with pricing loadings), claim probability, "
            "underwriting recommendation, and explainable risk factors.",
            style={"color": MUTED, "marginBottom": "28px", "fontSize": "14px",
                   "lineHeight": "1.6"}),

        html.Div([

            # Col 1: Vehicle & Fleet
            html.Div([
                html.Div("Vehicle & Fleet", style=sec_hdr),
                field("Vehicle Type", drop("uw-veh",
                    [{"label": v.replace("_"," ").title(), "value": v}
                     for v in ["pickup","van","box_truck","semi_truck","straight_truck"]],
                    "box_truck")),
                field("Gross Vehicle Weight (lbs)", inp("uw-gvw", value=26000, min=3000, max=80000)),
                field("Vehicle Age (years)", inp("uw-vage", value=5, min=0, max=20)),
                field("Operating Pattern", drop("uw-op",
                    [{"label": v.replace("_"," ").title(), "value": v}
                     for v in ["urban","highway","mixed","long_haul"]], "mixed")),
                field("ADAS Equipped", drop("uw-adas",
                    [{"label":"Yes","value":1},{"label":"No","value":0}], 0)),
                field("Fleet Safety Program", drop("uw-safety",
                    [{"label":"Yes","value":1},{"label":"No","value":0}], 0)),
            ], style={"flex":"1","background":CARD_BG,"border":f"1px solid {BORDER}",
                      "borderRadius":"10px","padding":"20px",
                      "boxShadow":"0 1px 3px rgba(0,0,0,0.06)"}),

            # Col 2: Driver Profile
            html.Div([
                html.Div("Driver Profile", style=sec_hdr),
                field("Driver Age", inp("uw-dage", value=35, min=21, max=75)),
                field("Driver Tenure (days)", inp("uw-tenure", value=400, min=0, max=10000)),
                field("MVR Violations (3yr)", inp("uw-mvr", value=1, min=0, max=15)),
                field("Prior At-Fault Claims", inp("uw-prior", value=0, min=0, max=10)),
            ], style={"flex":"1","background":CARD_BG,"border":f"1px solid {BORDER}",
                      "borderRadius":"10px","padding":"20px",
                      "boxShadow":"0 1px 3px rgba(0,0,0,0.06)"}),

            # Col 3: Telematics Signals
            html.Div([
                html.Div("Telematics Signals", style=sec_hdr),
                html.Div([label_div("Aggression Index /100mi"),
                          dcc.Slider(id="uw-agg", min=0, max=50, step=0.5, value=3,
                                     marks={0:"0",10:"10",25:"25",50:"50"},
                                     tooltip={"placement":"bottom","always_visible":True})],
                         style=SL),
                html.Div([label_div("Speeding Rate /100mi"),
                          dcc.Slider(id="uw-spd", min=0, max=100, step=1, value=5,
                                     marks={0:"0",25:"25",50:"50",100:"100"},
                                     tooltip={"placement":"bottom","always_visible":True})],
                         style=SL),
                html.Div([label_div("Night Driving %"),
                          dcc.Slider(id="uw-night", min=0, max=100, step=1, value=20,
                                     marks={0:"0",25:"25",50:"50",100:"100"},
                                     tooltip={"placement":"bottom","always_visible":True})],
                         style=SL),
                html.Div([label_div("Max Continuous Driving (hrs)"),
                          dcc.Slider(id="uw-hrs", min=0, max=14, step=0.5, value=6,
                                     marks={0:"0",4:"4",8:"8",11:"11",14:"14"},
                                     tooltip={"placement":"bottom","always_visible":True})],
                         style=SL),
                html.Div([label_div("Behavioral Drift Signal"),
                          dcc.Slider(id="uw-drift", min=-2, max=2, step=0.1, value=0,
                                     marks={-2:"−2",-1:"−1",0:"0",1:"1",2:"2"},
                                     tooltip={"placement":"bottom","always_visible":True})],
                         style=SL),
                field("Harsh Braking Events", inp("uw-brake", value=150, min=0, max=5000)),
                field("Distraction Rate /Trip",
                      inp("uw-dist", value=0.10, min=0, max=5, step=0.01)),
            ], style={"flex":"1","background":CARD_BG,"border":f"1px solid {BORDER}",
                      "borderRadius":"10px","padding":"20px",
                      "boxShadow":"0 1px 3px rgba(0,0,0,0.06)"}),

            # Col 4: Policy Terms
            html.Div([
                html.Div("Policy Terms", style=sec_hdr),
                field("BI Limit ($)", drop("uw-bi",
                    [{"label":f"${v:,}","value":v}
                     for v in [50000,100000,300000,500000,1000000,2000000]], 300000)),
                field("Collision Deductible ($)", drop("uw-ded",
                    [{"label":f"${v:,}","value":v}
                     for v in [250,500,1000,2500,5000]], 1000)),
                field("Coverage Type", drop("uw-cov",
                    [{"label": v.replace("_"," ").title(), "value": v}
                     for v in ["liability_only","collision","comprehensive","combined"]],
                    "combined")),
                field("Annual Miles", inp("uw-miles", value=40000, min=1000, max=150000)),
            ], style={"flex":"1","background":CARD_BG,"border":f"1px solid {BORDER}",
                      "borderRadius":"10px","padding":"20px",
                      "boxShadow":"0 1px 3px rgba(0,0,0,0.06)"}),

        ], style={"display":"flex","gap":"20px","alignItems":"flex-start","marginBottom":"24px"}),

        html.Button(
            [html.Span("⚡", style={"marginRight":"10px","fontSize":"18px"}), "Score Risk"],
            id="uw-btn", n_clicks=0,
            style={"background": BLUE, "color": "#ffffff", "border": "none",
                   "borderRadius": "10px", "padding": "16px 0", "fontSize": "16px",
                   "fontWeight": "700", "cursor": "pointer", "width": "100%",
                   "marginBottom": "28px", "letterSpacing": "0.5px",
                   "boxShadow": f"0 4px 12px rgba(29,111,242,0.35)"}),

        html.Div(
            html.Div([
                html.Div("⚡", style={"fontSize":"32px","marginBottom":"8px"}),
                html.Div("Complete the form above and click Score Risk",
                         style={"color": MUTED, "fontSize": "14px"}),
            ], style={"textAlign":"center","padding":"48px 0"}),
            id="uw-output",
            style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                   "borderRadius": "10px", "minHeight": "80px"}),
    ])


@app.callback(
    Output("uw-output", "children"),
    Input("uw-btn",    "n_clicks"),
    State("uw-veh",   "value"), State("uw-gvw",   "value"),
    State("uw-vage",  "value"), State("uw-op",    "value"),
    State("uw-adas",  "value"), State("uw-safety","value"),
    State("uw-dage",  "value"), State("uw-tenure","value"),
    State("uw-mvr",   "value"), State("uw-prior", "value"),
    State("uw-agg",   "value"), State("uw-spd",   "value"),
    State("uw-night", "value"), State("uw-hrs",   "value"),
    State("uw-drift", "value"), State("uw-brake", "value"),
    State("uw-dist",  "value"), State("uw-bi",    "value"),
    State("uw-ded",   "value"), State("uw-cov",   "value"),
    State("uw-miles", "value"),
    prevent_initial_call=True,
)
def score_uw(n_clicks, veh, gvw, v_age, op, adas, safety,
             d_age, tenure, mvr, prior,
             agg, spd, night, hrs, drift, brake, dist,
             bi, ded, cov, miles):
    if not n_clicks:
        return ""

    # Guard nulls
    gvw    = gvw    or INPUT_DEFAULTS["gvw_lbs"]
    v_age  = v_age  or INPUT_DEFAULTS["vehicle_age_years"]
    d_age  = d_age  or INPUT_DEFAULTS["driver_age"]
    tenure = tenure or INPUT_DEFAULTS["driver_tenure_days"]
    mvr    = mvr    or 0
    prior  = prior  or 0
    agg    = agg    or INPUT_DEFAULTS["aggression_index_per100mi"]
    spd    = spd    or INPUT_DEFAULTS["speeding_rate_per100mi"]
    night  = night  or INPUT_DEFAULTS["night_driving_pct"]
    hrs    = hrs    or INPUT_DEFAULTS["max_continuous_driving_hrs"]
    drift  = drift  or 0.0
    brake  = brake  or INPUT_DEFAULTS["harsh_braking_events"]
    dist   = dist   or INPUT_DEFAULTS["distraction_rate_per_trip"]
    bi     = bi     or INPUT_DEFAULTS["bi_limit_usd"]
    ded    = ded    or INPUT_DEFAULTS["collision_deductible_usd"]
    miles  = miles  or INPUT_DEFAULTS["total_miles"]

    raw_input = {
        "vehicle_type":                veh,
        "gvw_lbs":                     gvw,
        "vehicle_age_years":           v_age,
        "operating_pattern":           op,
        "adas_equipped_flag":          adas,
        "fleet_safety_program_flag":   safety,
        "driver_age":                  d_age,
        "driver_tenure_days":          tenure,
        "mvr_violations_3yr":          mvr,
        "prior_at_fault_claims":       prior,
        "aggression_index_per100mi":   agg,
        "speeding_rate_per100mi":      spd,
        "night_driving_pct":           night,
        "max_continuous_driving_hrs":  hrs,
        "behavioral_drift_signal":     drift,
        "harsh_braking_events":        brake,
        "distraction_rate_per_trip":   dist,
        "bi_limit_usd":                bi,
        "collision_deductible_usd":    ded,
        "coverage_type":               cov,
        "total_miles":                 miles,
    }

    try:
        res = scorer.score_one(raw_input)
    except Exception as exc:
        return html.Div(f"Scoring error: {exc}", style={"color": RED, "padding": "12px"})

    tier    = res.risk_tier
    col     = TIER_COLORS.get(tier, BLUE)
    rec_col = col

    # ── Gauge ─────────────────────────────────────────────────────────────────
    port_avg_freq = CLAIM_RATE
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(res.p_claim * 100, 2),
        number={"suffix": "%", "valueformat": ".1f",
                "font": {"color": TEXT, "size": 28}},
        title={"text": "Frequency — P(Claim)", "font": {"color": MUTED, "size": 13}},
        gauge={
            "axis": {"range": [0, 30], "tickfont": {"color": MUTED},
                     "tickvals": [0, 5, 10, 15, 20, 25, 30],
                     "ticktext": ["0%","5%","10%","15%","20%","25%","30%"]},
            "bar":  {"color": col, "thickness": 0.28},
            "bgcolor": "#f8fafc",
            "bordercolor": BORDER,
            "steps": [
                {"range": [0,  10], "color": "#dcfce7"},
                {"range": [10, 18], "color": "#fef9c3"},
                {"range": [18, 30], "color": "#fee2e2"},
            ],
            "threshold": {"value": round(port_avg_freq * 100, 1),
                          "line": {"color": ORANGE, "width": 2},
                          "thickness": 0.8},
        },
    ))
    gauge.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        height=220, margin=dict(l=20, r=20, t=50, b=10))

    # ── SHAP-style waterfall chart from risk_scoring shap_chart_data ──────────
    shap_data = res.shap_chart_data
    if shap_data:
        shap_sorted = sorted(shap_data, key=lambda x: abs(x["value"]), reverse=True)[:10]
        fig_shap = go.Figure(go.Bar(
            x=[d["value"] for d in shap_sorted],
            y=[d["factor"] for d in shap_sorted],
            orientation="h",
            marker_color=[RED if d["color"] == "red" else GREEN for d in shap_sorted],
            text=[f"{d['pct_of_total']:.0f}%" for d in shap_sorted],
            textposition="outside",
            hovertemplate="%{y}<br>Contribution: %{x:+.4f}<extra></extra>",
        ))
        fig_shap.add_vline(x=0, line_color=BORDER, line_width=1.5)
        fig_shap.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor="#f8fafc",
            font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=11),
            height=max(200, len(shap_sorted) * 28 + 60),
            showlegend=False, margin=dict(l=20, r=60, t=32, b=10),
            title=dict(text="Risk Factor Contributions (log-odds contribution units)",
                       font=dict(size=12, color=MUTED)),
            xaxis=dict(gridcolor="#e5e7eb", zeroline=False),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
        )
        shap_chart = dcc.Graph(figure=fig_shap, config={"displayModeBar": False})
    else:
        shap_chart = html.Div("Factor contributions unavailable.",
                              style={"color": MUTED, "fontSize": "12px"})

    # ── Top risk / protective factor cards ────────────────────────────────────
    def factor_pill(item, direction):
        c   = RED if direction == "risk" else GREEN
        ico = "↑" if direction == "risk" else "↓"
        return html.Div([
            html.Span(f"{ico} ", style={"color": c, "fontWeight": "700"}),
            html.Span(item["factor"], style={"color": TEXT, "fontSize": "12px",
                                             "fontWeight": "600"}),
            html.Span(f"  ({item['contribution']:.3f})",
                      style={"color": MUTED, "fontSize": "11px"}),
        ], style={"padding": "6px 10px", "borderRadius": "6px",
                  "background": f"{c}0d", "border": f"1px solid {c}30",
                  "marginBottom": "6px"})

    risk_pills  = [factor_pill(f, "risk")       for f in res.top_risk_factors]
    prot_pills  = [factor_pill(f, "protective") for f in res.top_protective_factors]

    # ── Pricing transparency math line ────────────────────────────────────────
    pc = res.pricing_config
    math_line = (
        f"Pure Premium = {res.p_claim:.1%} × ${res.e_loss_given_claim:,.0f} = ${res.pure_premium:,.0f}  "
        f"→  Indicated = ${res.pure_premium:,.0f} × {pc.get('total_loading', TOTAL_LOADING):.3f} "
        f"(expense {pc.get('expense_loading_pct', EXPENSE_LOADING_PCT):.0%} + "
        f"profit {pc.get('profit_loading_pct', PROFIT_LOADING_PCT):.0%}) "
        f"= ${res.indicated_premium:,.0f}"
    )

    return html.Div([html.Div([

        # ── Left: gauge + KPI cards ───────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span(tier.replace("_", " "),
                          style={"color": col, "fontSize": "22px", "fontWeight": "800",
                                 "marginRight": "12px"}),
                html.Span("RISK TIER", style={"color": MUTED, "fontSize": "11px",
                                              "fontWeight": "600", "letterSpacing": "1px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),

            html.Div(res.uw_decision,
                     style={"color": rec_col, "fontWeight": "700", "fontSize": "14px",
                            "marginBottom": "16px", "padding": "8px 12px",
                            "background": f"{rec_col}15", "borderRadius": "6px",
                            "border": f"1px solid {rec_col}40"}),

            dcc.Graph(figure=gauge, config={"displayModeBar": False}),
            html.Div(f"Orange line = portfolio avg freq ({port_avg_freq:.1%})",
                     style={"color": MUTED, "fontSize": "10px", "textAlign": "center",
                            "marginTop": "2px", "marginBottom": "12px"}),

            # 4 KPI boxes
            html.Div([
                html.Div([
                    html.Div("P(Claim)", style={"color": MUTED, "fontSize": "11px", "fontWeight": "600"}),
                    html.Div(f"{res.p_claim:.1%}", style={"color": BLUE, "fontWeight": "700",
                                                           "fontSize": "22px", "marginTop": "2px"}),
                    html.Div("Frequency", style={"color": MUTED, "fontSize": "10px"}),
                ], style={"textAlign": "center", "flex": "1", "padding": "12px",
                          "background": "#eff6ff", "borderRadius": "8px",
                          "border": "1px solid #bfdbfe"}),

                html.Div([
                    html.Div("Avg Loss/Claim", style={"color": MUTED, "fontSize": "11px", "fontWeight": "600"}),
                    html.Div(f"${res.e_loss_given_claim:,.0f}",
                             style={"color": PURPLE, "fontWeight": "700",
                                    "fontSize": "22px", "marginTop": "2px"}),
                    html.Div("Severity (capped)", style={"color": MUTED, "fontSize": "10px"}),
                ], style={"textAlign": "center", "flex": "1", "padding": "12px",
                          "background": "#faf5ff", "borderRadius": "8px",
                          "border": "1px solid #ddd6fe"}),

                html.Div([
                    html.Div("Pure Premium", style={"color": MUTED, "fontSize": "11px", "fontWeight": "600"}),
                    html.Div(f"${res.pure_premium:,.0f}",
                             style={"color": col, "fontWeight": "700",
                                    "fontSize": "22px", "marginTop": "2px"}),
                    html.Div("Freq × Severity", style={"color": MUTED, "fontSize": "10px"}),
                ], style={"textAlign": "center", "flex": "1", "padding": "12px",
                          "background": f"{col}10", "borderRadius": "8px",
                          "border": f"1px solid {col}40"}),

                html.Div([
                    html.Div("Indicated Premium", style={"color": MUTED, "fontSize": "11px", "fontWeight": "600"}),
                    html.Div(f"${res.indicated_premium:,.0f}",
                             style={"color": col, "fontWeight": "700",
                                    "fontSize": "22px", "marginTop": "2px"}),
                    html.Div(f"×{TOTAL_LOADING:.3f} load", style={"color": MUTED, "fontSize": "10px"}),
                ], style={"textAlign": "center", "flex": "1", "padding": "12px",
                          "background": f"{CYAN}10", "borderRadius": "8px",
                          "border": f"1px solid {CYAN}40"}),
            ], style={"display": "flex", "gap": "8px"}),

            html.Div(math_line, style={"color": MUTED, "fontSize": "10px",
                                       "textAlign": "center", "marginTop": "8px",
                                       "fontStyle": "italic"}),
        ], style={"flex": "1", "minWidth": "0"}),

        # ── Right: UW Recs + explainability ──────────────────────────────────
        html.Div([
            # UW Recommendations
            html.Div([
                html.Div("UNDERWRITING RECOMMENDATIONS",
                         style={"color": rec_col, "fontSize": "11px", "fontWeight": "700",
                                "letterSpacing": "0.8px", "marginBottom": "12px",
                                "paddingBottom": "8px", "borderBottom": f"1px solid {BORDER}"}),
                *[html.Div([
                    html.Span("→ ", style={"color": rec_col, "fontWeight": "700",
                                           "marginRight": "6px"}),
                    html.Span(a, style={"color": TEXT, "fontSize": "13px", "lineHeight": "1.5"}),
                ], style={"display": "flex", "alignItems": "flex-start",
                          "padding": "7px 0", "borderBottom": f"1px solid {BORDER}"})
                  for a in res.uw_actions],
            ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                      "borderLeft": f"3px solid {rec_col}", "borderRadius": "10px",
                      "padding": "16px", "marginBottom": "14px"}),

            # Top risk factors
            html.Div([
                html.Div("TOP RISK DRIVERS",
                         style={"color": RED, "fontSize": "11px", "fontWeight": "700",
                                "letterSpacing": "0.8px", "marginBottom": "10px"}),
                *(risk_pills if risk_pills else [html.Div("None identified.",
                                                          style={"color": MUTED, "fontSize": "12px"})]),
            ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                      "borderLeft": f"3px solid {RED}", "borderRadius": "8px",
                      "padding": "14px", "marginBottom": "10px"}),

            # Top protective factors
            html.Div([
                html.Div("TOP PROTECTIVE FACTORS",
                         style={"color": GREEN, "fontSize": "11px", "fontWeight": "700",
                                "letterSpacing": "0.8px", "marginBottom": "10px"}),
                *(prot_pills if prot_pills else [html.Div("None identified — consider ADAS and fleet safety enrolment.",
                                                           style={"color": MUTED, "fontSize": "12px"})]),
            ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                      "borderLeft": f"3px solid {GREEN}", "borderRadius": "8px",
                      "padding": "14px", "marginBottom": "10px"}),

            # SHAP chart
            html.Div([
                html.Div("FACTOR CONTRIBUTION BREAKDOWN",
                         style={"color": BLUE, "fontSize": "11px", "fontWeight": "700",
                                "letterSpacing": "0.8px", "marginBottom": "8px"}),
                shap_chart,
                html.Div("Red = risk-increasing  ·  Green = risk-reducing.  "
                         f"Source: risk_scoring._FACTOR_REGISTRY.  "
                         f"Severity cap ${SEVERITY_CAP_USD/1e6:.1f}M · "
                         f"Load {TOTAL_LOADING:.3f}× from pricing.json.",
                         style={"color": MUTED, "fontSize": "10px", "fontStyle": "italic",
                                "marginTop": "4px"}),
            ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                      "borderRadius": "8px", "padding": "14px"}),

        ], style={"flex": "1.4", "minWidth": "0"}),

    ], style={"display": "flex", "gap": "20px", "alignItems": "flex-start"})],
    style={"padding": "24px"})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def page_decomp():
    policy_options = [{"label": str(pid), "value": str(pid)}
                      for pid in sorted(df["policy_id"].astype(str).unique())[:300]]

    return html.Div([
        html.H4("Risk Decomposition",
                style={"color": TEXT, "fontWeight": "700",
                       "marginBottom": "6px", "fontSize": "22px"}),
        html.P(
            "Select a policy to see which factors drive its risk score.  "
            "The factor registry from risk_scoring.py computes signed contributions "
            "in log-odds contribution units — positive bars increase premium, negative reduce it.  "
            "The GLM frequency waterfall uses real model coefficients; the factor "
            "registry breakdown is consistent with the Prediction Tool output.",
            style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.6",
                   "marginBottom": "20px", "maxWidth": "860px",
                   "background": CARD_BG, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "14px 18px",
                   "borderLeft": f"3px solid {BLUE}"}),

        html.Div([
            html.Div([
                label_div("Select Policy ID"),
                dcc.Dropdown(id="dc-policy", options=policy_options,
                             value=policy_options[0]["value"] if policy_options else None,
                             clearable=False, style={"fontSize": "13px"}),
            ], style={"flex": "1", "maxWidth": "320px"}),
            html.Div(
                f"Showing top risk-increasing and risk-reducing drivers.  "
                f"Underwriting bands — "
                f"Low (<${TIER_THRESHOLDS['LOW'][1]/1000:.1f}K), "
                f"Moderate (${TIER_THRESHOLDS['LOW'][1]/1000:.1f}K\u2013${TIER_THRESHOLDS['MODERATE'][1]/1000:.1f}K), "
                f"High (${TIER_THRESHOLDS['MODERATE'][1]/1000:.1f}K\u2013${TIER_THRESHOLDS['HIGH'][1]/1000:.1f}K), "
                f"Severe (>${TIER_THRESHOLDS['HIGH'][1]/1000:.1f}K).",
                style={"color": MUTED, "fontSize": "12px",
                       "paddingBottom": "6px", "alignSelf": "flex-end"}),
        ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end",
                  "background": CARD_BG, "border": f"1px solid {BORDER}",
                  "borderRadius": "10px", "padding": "20px", "marginBottom": "16px",
                  "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),

        html.Div(id="dc-output"),
    ])


@app.callback(Output("dc-output", "children"), Input("dc-policy", "value"))
def decomp_update(policy_id):
    if not policy_id:
        return html.Div("Select a policy above.", style={"color": MUTED, "padding": "24px"})

    row = df[df["policy_id"].astype(str) == str(policy_id)]
    if row.empty:
        return html.Div(f"Policy {policy_id} not found.", style={"color": RED})
    r = row.iloc[0]

    FF = cfg["final_freq_features"]
    SF = cfg["final_sev_features"]

    # ── GLM frequency contributions (real model coefficients) ─────────────────
    try:
        row_aug = row.copy()
        for c in list(FF) + list(SF):
            if c not in row_aug.columns:
                row_aug[c] = 0.0
        freq_contribs = _glm_freq_contributions(row_aug[FF])
    except Exception:
        freq_contribs = {}

    try:
        row_aug2 = row.copy()
        for c in SF:
            if c not in row_aug2.columns:
                row_aug2[c] = 0.0
        sev_contribs = _sev_contributions(row_aug2[SF])
    except Exception:
        sev_contribs = {}

    # ── Factor registry contributions (consistent with Prediction Tool) ────────
    eng_dict = r.to_dict()
    factor_contribs = _compute_factor_contributions(eng_dict)

    tier    = str(r.get("risk_tier", "MODERATE"))
    tier_col = TIER_COLORS.get(tier, BLUE)
    pp_val  = float(r.get("pure_premium", 0))
    _ind_raw = r.get("indicated_premium", None)
    ind_val  = (float(_ind_raw) if _ind_raw is not None and float(_ind_raw) > 0
               else max(float(pp_val) * TOTAL_LOADING, MIN_PREMIUM_USD))

    # ── Header card ───────────────────────────────────────────────────────────
    header = html.Div([
        html.Div([
            html.Div([
                html.Span(tier.replace("_", " "),
                          style={"color": tier_col, "fontSize": "20px",
                                 "fontWeight": "800", "marginRight": "10px"}),
                html.Span("RISK TIER",
                          style={"color": MUTED, "fontSize": "11px",
                                 "fontWeight": "600", "letterSpacing": "1px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div(f"Policy {policy_id}  ·  "
                     f"{str(r.get('vehicle_type','')).replace('_',' ').title()}  ·  "
                     f"{str(r.get('operating_pattern','')).replace('_',' ').title()}",
                     style={"color": MUTED, "fontSize": "13px"}),
        ], style={"flex": "1"}),
        html.Div([
            html.Div([
                html.Div("Pure Premium",
                         style={"color": MUTED, "fontSize": "10px",
                                "fontWeight": "700", "textTransform": "uppercase",
                                "letterSpacing": "0.5px"}),
                html.Div(f"${pp_val:,.0f}",
                         style={"color": tier_col, "fontSize": "22px",
                                "fontWeight": "700", "marginTop": "2px"}),
            ], style={"textAlign": "center", "padding": "12px 22px",
                      "background": f"{tier_col}10", "borderRadius": "8px",
                      "border": f"1px solid {tier_col}40"}),
            html.Div([
                html.Div("Indicated Premium",
                         style={"color": MUTED, "fontSize": "10px",
                                "fontWeight": "700", "textTransform": "uppercase",
                                "letterSpacing": "0.5px"}),
                html.Div(f"${ind_val:,.0f}",
                         style={"color": CYAN, "fontSize": "22px",
                                "fontWeight": "700", "marginTop": "2px"}),
                html.Div(f"×{TOTAL_LOADING:.3f} load",
                         style={"color": MUTED, "fontSize": "10px"}),
            ], style={"textAlign": "center", "padding": "12px 22px",
                      "background": f"{CYAN}10", "borderRadius": "8px",
                      "border": f"1px solid {CYAN}40"}),
            html.Div([
                html.Div("P(Claim)",
                         style={"color": MUTED, "fontSize": "10px",
                                "fontWeight": "700", "textTransform": "uppercase",
                                "letterSpacing": "0.5px"}),
                html.Div(f"{float(r.get('p_claim',0)):.1%}",
                         style={"color": BLUE, "fontSize": "22px",
                                "fontWeight": "700", "marginTop": "2px"}),
            ], style={"textAlign": "center", "padding": "12px 22px",
                      "background": "#eff6ff", "borderRadius": "8px",
                      "border": "1px solid #bfdbfe"}),
        ], style={"display": "flex", "gap": "12px"}),
    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
              "background": CARD_BG, "border": f"2px solid {tier_col}40",
              "borderRadius": "10px", "padding": "18px 24px", "marginBottom": "14px",
              "boxShadow": f"0 2px 8px {tier_col}20"})

    # ── Factor registry chart (aligned with Prediction Tool) ──────────────────
    if factor_contribs:
        fc_sorted = sorted(factor_contribs, key=lambda x: x["magnitude"], reverse=True)[:10]
        fig_factors = go.Figure(go.Bar(
            x=[f["contribution"] for f in fc_sorted],
            y=[f["factor"] for f in fc_sorted],
            orientation="h",
            marker_color=[RED if f["direction"] == "risk" else GREEN for f in fc_sorted],
            hovertemplate="%{y}<br>Contribution: %{x:+.4f}<extra></extra>",
        ))
        fig_factors.add_vline(x=0, line_color=BORDER, line_width=1.5)
        fig_factors.update_layout(
            **_layout(
                margin=dict(l=20, r=20, t=48, b=20),
                xaxis=dict(gridcolor="#e5e7eb", zeroline=False),
                yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            ),
            height=max(220, len(fc_sorted) * 28 + 60),
            title="Factor Registry Contributions (log-odds contribution units) — aligned with Prediction Tool",
            showlegend=False,
        )
        factor_chart = dcc.Graph(figure=fig_factors, config={"displayModeBar": False})
    else:
        factor_chart = html.Div("Factor contributions unavailable.",
                                style={"color": MUTED, "fontSize": "12px"})

    # ── Key policy features ───────────────────────────────────────────────────
    feat_fields = [
        ("Aggression /100mi",          f"{r.get('aggression_index_per100mi',0):.2f}"),
        ("Speeding Rate /100mi",       f"{r.get('speeding_rate_per100mi',0):.2f}"),
        ("Night Driving %",            f"{r.get('night_driving_pct',0):.1f}%"),
        ("Max Continuous Hrs",         f"{r.get('max_continuous_driving_hrs',0):.1f}"),
        ("Behavioral Drift",           f"{r.get('behavioral_drift_signal',0):.2f}"),
        ("Fatigue Density",            f"{r.get('fatigue_exposure_density',0):.2f}"),
        ("Operational Exposure Index", f"{r.get('operational_exposure_risk_index',0):.3f}"),
        ("MVR Violations",             r.get('mvr_violations_3yr',0)),
        ("Prior At-Fault Claims",      r.get('prior_at_fault_claims',0)),
        ("ADAS Equipped",              "Yes" if r.get('adas_equipped_flag') else "No"),
        ("Fleet Safety Program",       "Yes" if r.get('fleet_safety_program_flag') else "No"),
        ("Had Claim",                  "✅ Yes" if r.get('had_claim_flag') else "No"),
    ]

    return html.Div([
        header,

        # Factor registry chart
        section([
            html.Div("Factor Registry Breakdown",
                     style={"color": TEXT, "fontWeight": "700", "fontSize": "14px",
                            "marginBottom": "4px"}),
            html.Div("Consistent with Prediction Tool.  Red = risk-increasing · Green = protective.",
                     style={"color": MUTED, "fontSize": "11px", "marginBottom": "8px"}),
            factor_chart,
        ]),

        # Key features
        section([
            html.Div("Key Policy Features", style={"color": TEXT, "fontWeight": "700",
                                                     "fontSize": "14px", "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Div(k, style={"color": MUTED, "fontSize": "11px"}),
                    html.Div(str(v), style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}),
                ], style={"padding": "7px 0", "borderBottom": f"1px solid {BORDER}"})
                for k, v in feat_fields
            ], style={"columns": "3", "gap": "24px"}),
        ]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CLAIMS ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_claims():
    claimed = df[df["had_claim_flag"] == 1].copy()

    # ── Loss distribution boxplot by model tier ────────────────────────────────
    fig_box = go.Figure()
    missing_tiers = []
    for tier in TIER_ORDER:
        sub = claimed.loc[claimed["risk_tier"] == tier, "incurred_loss_12mo_usd"]
        if len(sub) == 0:
            missing_tiers.append(tier); continue
        fig_box.add_trace(go.Box(
            y=sub, name=tier, marker_color=TIER_COLORS[tier], boxmean="sd",
            hovertemplate=tier + "<br>Loss: $%{y:,.0f}<extra></extra>",
        ))
    title_box = "Incurred Loss Distribution — All Risk Tiers (claimants only)"
    if missing_tiers:
        title_box += f"  [no claimants: {', '.join(missing_tiers)}]"
    fig_box.update_layout(**BASE_LAYOUT, height=380, title=title_box,
                           yaxis_title="Incurred Loss ($)")

    # ── Pareto curve ──────────────────────────────────────────────────────────
    cs = claimed.sort_values("incurred_loss_12mo_usd", ascending=False).copy()
    cs["cum_pct"]  = cs["incurred_loss_12mo_usd"].cumsum() / cs["incurred_loss_12mo_usd"].sum() * 100
    cs["rank_pct"] = np.arange(1, len(cs)+1) / len(cs) * 100
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=cs["rank_pct"], y=cs["cum_pct"],
        mode="lines", fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x:.1f}% of claimants → %{y:.1f}% of losses<extra></extra>",
    ))
    fig_pareto.add_hline(y=80, line_color=ORANGE, line_dash="dash",
                          annotation_text="80% of losses", annotation_font_color=ORANGE)
    fig_pareto.update_layout(**BASE_LAYOUT, height=320, title="Loss Concentration — Pareto",
                              xaxis_title="% Claimants (ranked by loss)",
                              yaxis_title="Cumulative Loss %")

    # ── State scatter ─────────────────────────────────────────────────────────
    state_s = df.groupby("state_of_domicile").agg(
        claim_rate=("had_claim_flag","mean"),
        avg_pp=("pure_premium","mean"),
        avg_ind=("indicated_premium","mean"),
        tot_loss=("incurred_loss_12mo_usd","sum"),
        cnt=("policy_id","count"),
    ).reset_index()
    fig_state = px.scatter(
        state_s, x="claim_rate", y="avg_pp", size="tot_loss",
        text="state_of_domicile",
        color="claim_rate", color_continuous_scale="Plasma",
        title="State: Claim Rate vs Avg Pure Premium (bubble = total loss)",
        labels={"claim_rate":"Claim Rate","avg_pp":"Avg Pure Premium ($)"},
        hover_data={"cnt": True, "tot_loss": ":,.0f", "avg_ind": ":,.0f"},
    )
    fig_state.update_traces(textposition="top center", textfont_color=TEXT)
    fig_state.update_layout(**BASE_LAYOUT, height=380, coloraxis_showscale=False)

    # ── Claim rate heatmap: vehicle × operating pattern ───────────────────────
    fig_mix = px.bar(
        df.groupby(["vehicle_type","operating_pattern"])["had_claim_flag"].mean().reset_index(),
        x="vehicle_type", y="had_claim_flag", color="operating_pattern",
        barmode="group", title="Claim Rate: Vehicle × Operating Pattern",
        labels={"had_claim_flag":"Claim Rate","vehicle_type":"Vehicle"},
        color_discrete_sequence=PALETTE,
    )
    fig_mix.update_layout(**BASE_LAYOUT, height=300)

    # ── Actual vs predicted claim rate by risk tier ────────────────────────────
    tier_lr = []
    for t in TIER_ORDER:
        sub = df[df["risk_tier"] == t]
        if len(sub) == 0: continue
        act_lr = (sub["incurred_loss_12mo_usd"].mean() / sub["pure_premium"].mean()
                  if sub["pure_premium"].mean() > 0 else 0)
        tier_lr.append({
            "tier":                   t,
            "Actual Claim Rate (%)":  round(sub["had_claim_flag"].mean() * 100, 2),
            "Predicted P(Claim) (%)": round(sub["p_claim"].mean() * 100, 2),
            "Actual Avg Loss ($)":    round(sub["incurred_loss_12mo_usd"].mean(), 0),
            "Avg Pure Premium ($)":   round(sub["pure_premium"].mean(), 0),
            "Avg Indicated Premium ($)": round(sub["indicated_premium"].mean(), 0),
            "Actual Loss Ratio":      round(act_lr, 3),
        })
    tier_lr_df = pd.DataFrame(tier_lr)

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Bar(
        name="Actual Claim Rate (%)", x=tier_lr_df["tier"],
        y=tier_lr_df["Actual Claim Rate (%)"],
        marker_color=[TIER_COLORS[t] for t in tier_lr_df["tier"]], opacity=1.0,
        hovertemplate="<b>%{x}</b><br>Actual Claim Rate: %{y:.2f}%<extra></extra>",
    ))
    fig_avp.add_trace(go.Scatter(
        name="Predicted P(Claim) (%)", x=tier_lr_df["tier"],
        y=tier_lr_df["Predicted P(Claim) (%)"],
        mode="lines+markers",
        line=dict(color=CYAN, width=2, dash="dot"),
        marker=dict(color=CYAN, size=10, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Predicted P(Claim): %{y:.2f}%<extra></extra>",
    ))
    fig_avp.update_layout(
        **_layout(legend=dict(orientation="h", y=1.08, x=0)),
        height=340,
        title="Actual vs Predicted Claim Rate by Risk Tier  [Model Calibration]",
        yaxis_title="Claim Rate / P(Claim) (%)",
        xaxis_title="Risk Tier", barmode="group",
    )

    # ── Severity cap waterfall — impact at portfolio level ────────────────────
    pred_capped = preds[preds["e_loss_capped"].notna()].copy()
    if "e_loss_raw" in preds.columns:
        n_cap = int((preds["e_loss_raw"] > SEVERITY_CAP_USD).sum())
        cap_pct = n_cap / len(preds) * 100
        cap_note = (f"{n_cap:,} predictions ({cap_pct:.2f}%) were capped at "
                    f"${SEVERITY_CAP_USD/1e6:.1f}M UW severity cap.")
    else:
        cap_note = f"Severity cap ${SEVERITY_CAP_USD/1e6:.1f}M applied in model_train.py."

    # ── Top loss-driving segments ─────────────────────────────────────────────
    seg = (df.groupby(["vehicle_type", "operating_pattern"]).agg(
        total_loss      =("incurred_loss_12mo_usd", "sum"),
        n_claims        =("had_claim_flag",          "sum"),
        n_policies      =("policy_id",               "count"),
        avg_p_claim_pct =("p_claim",                 lambda x: round(x.mean() * 100, 2)),
        avg_pure_prem   =("pure_premium",            "mean"),
        avg_ind_prem    =("indicated_premium",        "mean"),
    ).reset_index()
     .assign(
         claim_rate_pct =lambda d: (d["n_claims"] / d["n_policies"] * 100).round(2),
         loss_share_pct =lambda d: (d["total_loss"] / d["total_loss"].sum() * 100).round(2),
         avg_pure_prem  =lambda d: d["avg_pure_prem"].round(0).astype(int),
         avg_ind_prem   =lambda d: d["avg_ind_prem"].round(0).astype(int),
         total_loss     =lambda d: d["total_loss"].round(0).astype(int),
     )
     .sort_values("total_loss", ascending=False)
     .head(15)
     .rename(columns={
         "vehicle_type":     "Vehicle Type",
         "operating_pattern":"Operating Pattern",
         "total_loss":       "Total Loss ($)",
         "n_claims":         "# Claims",
         "n_policies":       "# Policies",
         "avg_p_claim_pct":  "Avg P(Claim) %",
         "claim_rate_pct":   "Actual Claim Rate %",
         "loss_share_pct":   "Loss Share %",
         "avg_pure_prem":    "Avg Pure Premium ($)",
         "avg_ind_prem":     "Avg Indicated Premium ($)",
     })
    )
    seg_table = dash_table.DataTable(
        data=seg.to_dict("records"),
        columns=[{"name": c, "id": c} for c in seg.columns],
        style_header={"background": CARD_BG, "color": TEXT, "fontWeight": "600",
                      "fontSize": "11px", "border": f"1px solid {BORDER}"},
        style_cell={"background": DARK_BG, "color": TEXT, "fontSize": "12px",
                    "border": f"1px solid {BORDER}", "padding": "7px 10px",
                    "textAlign": "left"},
        style_data_conditional=[
            {"if": {"filter_query": "{Loss Share %} > 10"},
             "color": RED, "fontWeight": "700"},
            {"if": {"filter_query": "{Loss Share %} > 5"},
             "color": ORANGE},
        ],
        page_size=15, sort_action="native", filter_action="native",
    )

    # ── KPI stats ─────────────────────────────────────────────────────────────
    c10  = int(len(claimed) * 0.1)
    top10_share = claimed["incurred_loss_12mo_usd"].nlargest(c10).sum() / \
                  claimed["incurred_loss_12mo_usd"].sum()
    high_pred = df.nlargest(int(len(df) * 0.1), "p_claim")
    hit_rate  = high_pred["had_claim_flag"].mean()

    return html.Div([
        html.H4("Claims Analytics",
                style={"color": TEXT, "fontWeight": "700",
                       "marginBottom": "6px", "fontSize": "22px"}),

        # Validation banner
        html.Div([
            html.Div("MODEL VALIDATION",
                     style={"color": CYAN, "fontSize": "10px", "fontWeight": "700",
                            "letterSpacing": "0.8px", "marginBottom": "8px"}),
            html.P(
                f"Model predictions are validated against actual loss behaviour.  "
                f"Top 10% of policies by P(Claim) realised a {hit_rate:.1%} actual claim rate vs "
                f"portfolio average {CLAIM_RATE:.1%}.  "
                f"Pareto: top 10% of claimants drive {top10_share:.0%} of total losses.  "
                f"Actual vs Predicted chart shows per-tier calibration (both in %).  "
                f"Pricing: expense {EXPENSE_LOADING_PCT:.0%} + profit {PROFIT_LOADING_PCT:.0%} "
                f"= {TOTAL_LOADING:.3f}× total load (pricing.json).  "
                f"{cap_note}",
                style={"color": TEXT, "fontSize": "13px", "lineHeight": "1.7", "margin": 0}),
        ], style={"background": CARD_BG, "border": f"1px solid {CYAN}40",
                  "borderLeft": f"3px solid {CYAN}", "borderRadius": "8px",
                  "padding": "14px 18px", "marginBottom": "20px"}),

        # KPI row
        html.Div([
            kpi("Total Claims",        f"{len(claimed):,}",
                f"of {N_POLICIES:,} policies"),
            kpi("Avg Claim",           f"${claimed['incurred_loss_12mo_usd'].mean():,.0f}",
                "Mean incurred loss",   ORANGE),
            kpi("Median Claim",        f"${claimed['incurred_loss_12mo_usd'].median():,.0f}",
                "50th percentile",      PURPLE),
            kpi("Largest Claim",       f"${claimed['incurred_loss_12mo_usd'].max():,.0f}",
                "Single loss",          RED),
            kpi("Top 10% Loss Share",  f"{top10_share:.1%}",
                "Loss concentration",   RED),
            kpi("Top-Decile Hit Rate", f"{hit_rate:.1%}",
                "Top 10% P(Claim) → actual", CYAN),
        ], style={"display":"grid","gridTemplateColumns":"repeat(6,1fr)",
                  "gap":"12px","marginBottom":"20px"}),

        section([html.Div([
            html.Div(dcc.Graph(figure=fig_box),   style={"flex":"1"}),
            html.Div(dcc.Graph(figure=fig_state), style={"flex":"1"}),
        ], style={"display":"flex","gap":"16px"})]),

        section([html.Div([
            html.Div("Actual vs Predicted Claim Rate by Risk Tier",
                     style={"color": TEXT, "fontWeight": "600", "fontSize": "13px",
                            "marginBottom": "4px"}),
            html.Div("Bars = realised claim rate; dotted line = model P(Claim). "
                     "Divergence = potential mis-calibration.",
                     style={"color": MUTED, "fontSize": "11px", "marginBottom": "8px"}),
            dcc.Graph(figure=fig_avp),
        ])]),

        section([dcc.Graph(figure=fig_pareto)]),
        section([dcc.Graph(figure=fig_mix)]),

        section([
            html.Div("Top Loss-Driving Segments  (Vehicle × Operating Pattern)",
                     style={"color": TEXT, "fontWeight": "600", "marginBottom": "4px",
                            "fontSize": "13px"}),
            html.Div(
                f"Ranked by total incurred loss.  "
                f"P(Claim) in %.  Loss Share % > 10% = red, > 5% = amber.  "
                f"Avg Indicated Premium = Pure Premium × {TOTAL_LOADING:.3f}× load.",
                style={"color": MUTED, "fontSize": "11px", "marginBottom": "12px"}),
            seg_table,
        ]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("page-content",     "children"),
    Output("exp-page-wrapper", "style"),
    Input("url", "pathname"),
)
def router(path):
    HIDDEN = {"display": "none"}
    SHOWN  = {"display": "block"}
    if path == "/explorer":
        return None, SHOWN
    elif path == "/underwriting":
        return page_underwriting(), HIDDEN
    elif path == "/decomp":
        return page_decomp(), HIDDEN
    elif path == "/claims":
        return page_claims(), HIDDEN
    else:
        return page_dashboard(), HIDDEN


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting Telematics Risk Intelligence Platform (v2)...")
    print(f"  Pricing config:  expense={EXPENSE_LOADING_PCT:.0%}  "
          f"profit={PROFIT_LOADING_PCT:.0%}  "
          f"sev_cap=${SEVERITY_CAP_USD/1e6:.1f}M  "
          f"min_premium=${MIN_PREMIUM_USD:,.0f}")
    print(f"  Total load:      {TOTAL_LOADING:.4f}× (from config/pricing.json)")
    print(f"  Tier thresholds: {TIER_THRESHOLDS}")
    print("\n  Open  http://127.0.0.1:8050  in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=8050)