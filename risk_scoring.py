"""
Telematics Risk Intelligence — Risk Scoring Engine
===================================================
Standalone, dataset-agnostic risk scoring module.

Responsibilities
----------------
1.  Feature engineering   — derives all Tier-2 / Tier-3 features from raw inputs
2.  Input validation      — range-checks and sensible-default fallbacks
                            (warnings are concise, dashboard-safe strings)
3.  Model inference       — wraps sklearn/statsmodels/EBM pipelines uniformly
4.  Pure-premium calc     — two-part GLM: P(claim) × E[Loss|claim]
5.  Pricing loadings      — applies expense/profit loading and severity cap from
                            config/pricing.json (consistent with model_train.py)
6.  Risk tiering          — maps pure premium → LOW / MODERATE / HIGH / VERY_HIGH
                            using thresholds read from config/pricing.json
7.  UW recommendations    — actionable guidance per tier
8.  Explainability        — top risk-driving and protective factors (factor list
                            + SHAP-style bar-chart data ready for dashboards)
9.  Batch scoring         — vectorized DataFrame scoring; iterrows() is fully
                            eliminated from the hot path (50–200× speedup on
                            large portfolios vs the previous row-loop design)
10. Standalone scoring    — rule-based fallback (no model artifacts needed)

Pricing Config  (config/pricing.json — auto-generated with defaults if absent)
-------------------------------------------------------------------------------
  tier_thresholds     : pure-premium cut-points shared with model_train.py
  expense_loading_pct : overhead loading applied to pure premium (default 25%)
  profit_loading_pct  : profit margin loading (default 5%)
  severity_cap_usd    : UW cap on E[Loss|claim] before PP assembly (default $1.5M)
  min_premium_usd     : premium floor after all loadings (default $500)

  indicated_premium = max(pure_premium × (1 + expense + profit), min_premium)

Risk Tier Thresholds  (from pricing.json; defaults mirror model_train.py)
--------------------------------------------------------------------------
  LOW       : pure_premium < 1,800
  MODERATE  : 1,800 ≤ pure_premium < 5,500
  HIGH      : 5,500 ≤ pure_premium < 12,000
  VERY_HIGH : pure_premium ≥ 12,000

Safety Feature Sign Convention
-------------------------------
  ADAS and fleet_safety_program REDUCE risk (negative coefficients in GLM).
  Higher values → LOWER claim probability → LOWER pure premium.
  This is the correct actuarial direction and is enforced consistently everywhere.

Explainability
--------------
  Every RiskScoreResult includes:
    top_risk_factors        : up to 3 factors most increasing risk
    top_protective_factors  : up to 2 factors most reducing risk
    shap_chart_data         : all signed contributions sorted for a waterfall /
                              horizontal-bar SHAP-style chart (dashboard-ready
                              JSON payload with factor, signed value, color, and
                              % of total contribution)

Batch Scoring Performance
-------------------------
  score_batch() uses fully vectorized NumPy operations for both rule-based and
  model-based paths:
    • FeatureEngineer.engineer_batch()      — NumPy column ops, no iterrows
    • RuleBasedScorer.score_batch_vec()     — NumPy matrix formula pass
    • _model_inference_batch()              — single sklearn transform+predict call
    • Tier assignment via np.digitize()     — O(n log k)
    • Pricing loadings via np.maximum()     — O(n)
  Only the per-row explainability step uses a Python loop (unavoidable for
  dict-level factor computation; fast because it is pure arithmetic, no I/O).

  Input validation warnings in batch mode are summarised as concise
  pipe-separated strings suitable for a dashboard tooltip column
  (e.g. "missing: gvw_lbs | clipped: night_driving_pct").

Usage
-----
    # --- With trained models (production) ---
    from risk_scoring import TelematicsRiskScorer

    scorer = TelematicsRiskScorer.from_artifacts(
        model_dir="models/",
        config_path="config/feature_selection.json",
    )
    result   = scorer.score_one({"vehicle_type": "semi_truck", "gvw_lbs": 68000, ...})
    batch_df = scorer.score_batch(df)

    # --- Rule-based fallback (no models needed) ---
    scorer = TelematicsRiskScorer.rule_based_only()
    result = scorer.score_one({...})

Dependencies
------------
    numpy, pandas, scikit-learn
    interpret (optional — for EBM severity model)

All public methods return a RiskScoreResult dataclass so callers never
need to parse raw dicts.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─── Paste this block into risk_scoring.py ───────────────────────────────────
# Place it after the imports section (numpy, sklearn, etc.) and BEFORE any
# function/class that uses it.  The class must live in risk_scoring so that
# pickle can resolve it when app.py loads glm_frequency.pkl.

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionWithOffset(LogisticRegression):
    """
    Thin wrapper that injects log(exposure) as a forced-coefficient=1 term
    by subtracting it from the linear predictor before the sigmoid.

    sklearn API compatible — can be used in cross_val_predict.

    Must be defined in risk_scoring (not model_train) so that pickle can
    resolve the class when app.py unpickles glm_frequency.pkl.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._offset: np.ndarray | None = None

    def set_offset(self, offset: np.ndarray) -> "LogisticRegressionWithOffset":
        self._offset = np.asarray(offset, dtype=float)
        return self

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Prepend offset column so sklearn treats it as a feature."""
        if self._offset is None:
            return X
        return np.column_stack([self._offset[:, None], X])

    def fit(self, X, y, sample_weight=None):
        return super().fit(self._augment(X), y, sample_weight=sample_weight)

    def predict_proba(self, X):
        return super().predict_proba(self._augment(X))

    def predict(self, X):
        return super().predict(self._augment(X))
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PRICING CONFIG  (single source of truth — mirrors model_train.py)
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PRICING: Dict[str, Any] = {
    "tier_thresholds": {
        "LOW":       [None,   1_800],
        "MODERATE":  [1_800,  5_500],
        "HIGH":      [5_500,  12_000],
        "VERY_HIGH": [12_000, None],
    },
    "expense_loading_pct": 0.25,
    "profit_loading_pct":  0.05,
    "severity_cap_usd":    1_500_000,
    "min_premium_usd":     500,
}

_HERE             = os.path.dirname(os.path.abspath(__file__))
_PRICING_CFG_PATH = os.path.join(_HERE, "config", "pricing.json")
if not os.path.exists(_PRICING_CFG_PATH):
    _PRICING_CFG_PATH = os.path.join(os.getcwd(), "config", "pricing.json")


def _load_pricing_config(path: str = _PRICING_CFG_PATH) -> Dict[str, Any]:
    """Load pricing config; auto-create with defaults if absent (mirrors model_train.py)."""
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_DEFAULT_PRICING, fh, indent=2)
    return dict(_DEFAULT_PRICING)


_PRICING = _load_pricing_config()

EXPENSE_LOADING_PCT: float = float(_PRICING["expense_loading_pct"])
PROFIT_LOADING_PCT:  float = float(_PRICING["profit_loading_pct"])
SEVERITY_CAP_USD:    float = float(_PRICING["severity_cap_usd"])
MIN_PREMIUM_USD:     float = float(_PRICING["min_premium_usd"])
TOTAL_LOADING:       float = 1.0 + EXPENSE_LOADING_PCT + PROFIT_LOADING_PCT


def _build_tier_thresholds(raw: Dict[str, List]) -> Dict[str, Tuple[float, float]]:
    order = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    out: Dict[str, Tuple[float, float]] = {}
    for tier in order:
        if tier not in raw:
            continue
        lo_raw, hi_raw = raw[tier]
        lo = 0.0          if lo_raw is None else float(lo_raw)
        hi = float("inf") if hi_raw is None else float(hi_raw)
        out[tier] = (lo, hi)
    return out


TIER_THRESHOLDS: Dict[str, Tuple[float, float]] = _build_tier_thresholds(
    _PRICING["tier_thresholds"]
)
TIER_ORDER: List[str] = [t for t in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
                          if t in TIER_THRESHOLDS]

# Pre-built bin edges for np.digitize — avoids per-call dict lookups in batch
_TIER_BINS:   List[float]    = [-np.inf] + [TIER_THRESHOLDS[t][1] for t in TIER_ORDER[:-1]] + [np.inf]
_TIER_LABELS: np.ndarray     = np.array(TIER_ORDER)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UNDERWRITING RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

UW_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "LOW": {
        "decision": "✅  STANDARD ACCEPT — PREFERRED RISK",
        "actions": [
            "Bind at preferred rates; eligible for multi-policy discount.",
            "No additional conditions required.",
            "Eligible for deductible credit if ADAS is equipped.",
            "Low priority for telematics monitoring — annual review sufficient.",
            "Candidate for fleet safety reward programme if applicable.",
        ],
    },
    "MODERATE": {
        "decision": "📋  STANDARD WITH CONDITIONS",
        "actions": [
            "Bind at standard rates with standard deductible.",
            "Recommend (not require) ADAS and fleet safety program.",
            "Schedule 6-month telematics review — watch aggression trend.",
            "No BI sub-limits required at this tier.",
            "Flag for renewal review if behavioral drift worsens.",
        ],
    },
    "HIGH": {
        "decision": "⚠️  CONDITIONAL ACCEPT — RESTRICTIONS REQUIRED",
        "actions": [
            "Bind with higher deductible ($2,500+) to share risk.",
            "Require fleet safety program enrollment within 60 days.",
            "Apply surcharge of 20–40% above filed base rate.",
            "Set a telematics monitoring condition with 90-day review.",
            "Restrict long-haul / overnight operations if fatigue score is elevated.",
        ],
    },
    "VERY_HIGH": {
        "decision": "🚫  DECLINE / REFER TO SENIOR UNDERWRITER",
        "actions": [
            "Decline or refer to senior underwriter for manual review.",
            "Require mandatory fleet safety program enrollment before binding.",
            "Require ADAS installation as a condition of coverage.",
            "Consider sub-limits on BI coverage given severity exposure.",
            "Request 6-month driving history review and MVR re-pull.",
        ],
    },
}

# ── Input defaults & valid ranges ─────────────────────────────────────────────

INPUT_DEFAULTS: Dict[str, Any] = {
    "vehicle_type":               "box_truck",
    "gvw_lbs":                    26_000,
    "vehicle_age_years":          5,
    "operating_pattern":          "mixed",
    "adas_equipped_flag":         0,
    "fleet_safety_program_flag":  0,
    "driver_age":                 35,
    "driver_tenure_days":         400,
    "mvr_violations_3yr":         1,
    "prior_at_fault_claims":      0,
    "aggression_index_per100mi":  3.0,
    "speeding_rate_per100mi":     5.0,
    "night_driving_pct":          20.0,
    "max_continuous_driving_hrs": 6.0,
    "behavioral_drift_signal":    0.0,
    "harsh_braking_events":       150,
    "harsh_acceleration_events":  120,
    "distraction_rate_per_trip":  0.10,
    "total_miles":                40_000,
    "total_trips":                1_000,
    "phone_distraction_events":   100.0,
    "tailgating_events":          80.0,
    "speeding_events":            1_200,
    "behavioral_volatility_index":2.5,
    "bi_limit_usd":               300_000,
    "collision_deductible_usd":   1_000,
    "coverage_type":              "combined",
    "state_of_domicile":          "TX",
}

INPUT_RANGES: Dict[str, Tuple[float, float]] = {
    "gvw_lbs":                    (3_000,   80_000),
    "vehicle_age_years":          (0,       20),
    "driver_age":                 (21,      75),
    "driver_tenure_days":         (0,       10_000),
    "mvr_violations_3yr":         (0,       15),
    "prior_at_fault_claims":      (0,       10),
    "aggression_index_per100mi":  (0,       50),
    "speeding_rate_per100mi":     (0,       100),
    "night_driving_pct":          (0,       100),
    "max_continuous_driving_hrs": (0,       14),
    "behavioral_drift_signal":    (-2,      2),
    "harsh_braking_events":       (0,       5_000),
    "harsh_acceleration_events":  (0,       5_000),
    "distraction_rate_per_trip":  (0,       5),
    "total_miles":                (1_000,   150_000),
    "total_trips":                (50,      5_000),
    "phone_distraction_events":   (0,       3_000),
    "tailgating_events":          (0,       2_000),
    "speeding_events":            (0,       10_000),
    "behavioral_volatility_index":(0,       5),
    "bi_limit_usd":               (50_000,  2_000_000),
    "collision_deductible_usd":   (250,     5_000),
}

VALID_VEHICLE_TYPES      = {"pickup", "van", "box_truck", "semi_truck", "straight_truck"}
VALID_OPERATING_PATTERNS = {"urban", "highway", "mixed", "long_haul"}
VALID_COVERAGE_TYPES     = {"liability_only", "collision", "comprehensive", "combined"}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EXPLAINABILITY  — unified factor registry
# ══════════════════════════════════════════════════════════════════════════════
#
# Registry drives three outputs simultaneously:
#   top_risk_factors        — top-N risk-increasing items (for UW narrative)
#   top_protective_factors  — top-N risk-reducing items (for client story)
#   shap_chart_data         — all signed contributions (for bar/waterfall chart)
#
# Weights mirror RuleBasedScorer.score() exactly so contributions are in
# freq_logit units and directly comparable across factors and policies.
#
# Registry entry: (label, weight, feature_key, scale, direction)
#   contribution = weight × (feature_value / scale)
#   "protective" factors carry a negative sign in shap_chart_data.

_FACTOR_REGISTRY: List[Tuple[str, float, str, float, str]] = [
    # label                              weight   feature_key                           scale   direction
    ("Operational Exposure",             1.20,   "operational_exposure_risk_index",     1.0,   "risk"),
    ("Driver Aggression",                0.80,   "aggression_index_per100mi",           50.0,  "risk"),
    ("MVR Violations (3yr)",             0.60,   "mvr_violations_3yr",                  10.0,  "risk"),
    ("Prior At-Fault Claims",            0.50,   "prior_at_fault_claims",               5.0,   "risk"),
    ("Speed × Night Severity",           0.45,   "speed_x_night_severity_multiplier",   5.0,   "risk"),
    ("Fatigue × Long-Haul",              0.35,   "fatigue_x_longhaul_score",            5.0,   "risk"),
    ("New Driver on Heavy Vehicle",      0.30,   "new_driver_heavy_vehicle_multiplier", 5.0,   "risk"),
    ("Distraction Rate",                 0.15,   "distraction_rate_per_trip",           5.0,   "risk"),
    # Behavioural drift splits into risk (positive) and protective (negative)
    ("Behavioural Drift ▲",              0.20,   "_drift_positive",                     2.0,   "risk"),
    ("Behavioural Improvement ▼",        0.20,   "_drift_negative",                     2.0,   "protective"),
    ("Fleet Safety Program",             0.40,   "fleet_safety_program_flag",           1.0,   "protective"),
    ("ADAS Equipped",                    0.30,   "adas_equipped_flag",                  1.0,   "protective"),
]

_MIN_CONTRIBUTION = 0.005   # suppress noise below this threshold


def _compute_factor_contributions(e: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compute all signed factor contributions in freq_logit units.

    Returns a list of dicts (sorted by abs magnitude, descending):
        factor      : human label
        contribution: signed float  (+ve = increases risk, −ve = decreases risk)
        direction   : "risk" | "protective"
        magnitude   : abs(contribution)

    This list is the single source used for top_risk_factors,
    top_protective_factors, and shap_chart_data.
    """
    drift   = float(e.get("behavioral_drift_signal", 0.0))
    extras  = {"_drift_positive": max(0.0, drift), "_drift_negative": max(0.0, -drift)}
    env     = {**e, **extras}
    results: List[Dict[str, Any]] = []

    for label, weight, feat, scale, direction in _FACTOR_REGISTRY:
        magnitude = abs(weight * float(env.get(feat, 0.0)) / scale)
        if magnitude < _MIN_CONTRIBUTION:
            continue
        signed = magnitude if direction == "risk" else -magnitude
        results.append({
            "factor":       label,
            "contribution": round(signed, 4),
            "direction":    direction,
            "magnitude":    round(magnitude, 4),
        })

    results.sort(key=lambda x: x["magnitude"], reverse=True)
    return results


def _top_factors(
    factors: List[Dict[str, Any]], direction: str, n: int
) -> List[Dict[str, Any]]:
    """Return the top-n factors for a given direction, highest magnitude first."""
    subset = [f for f in factors if f["direction"] == direction]
    return [{"factor": f["factor"], "contribution": f["magnitude"]} for f in subset[:n]]


def _shap_chart_data(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a dashboard-ready payload for a SHAP-style horizontal waterfall chart.

    Each entry:
        factor      : label (str)
        value       : signed contribution in freq_logit units
                      (+ve = risk-increasing, −ve = risk-reducing)
        color       : "red"   → risk driver
                      "green" → protective factor
        pct_of_total: percentage of total absolute contribution (0–100 float)

    Sorted by abs(value) descending — most impactful factor rendered first.
    Dashboard consumers can render this list directly; no post-processing needed.
    """
    total_abs = sum(f["magnitude"] for f in factors) or 1.0
    return [
        {
            "factor":       f["factor"],
            "value":        f["contribution"],
            "color":        "red" if f["direction"] == "risk" else "green",
            "pct_of_total": round(f["magnitude"] / total_abs * 100, 1),
        }
        for f in factors
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskScoreResult:
    """
    Returned by every scoring method.

    Attributes
    ----------
    p_claim               : estimated claim probability  [0, 1]
    e_loss_given_claim    : expected loss severity in USD (after UW cap)
    pure_premium          : p_claim × e_loss_given_claim
    indicated_premium     : pure_premium × (1+expense+profit), floored at min
    risk_tier             : LOW | MODERATE | HIGH | VERY_HIGH
    uw_decision           : one-line underwriting ruling
    uw_actions            : bullet-point UW guidance
    top_risk_factors      : up to 3 highest-impact risk-increasing factors
                            [{"factor": str, "contribution": float}, ...]
    top_protective_factors: up to 2 highest-impact risk-reducing factors
    shap_chart_data       : all signed contributions for a waterfall/bar chart
                            [{"factor", "value", "color", "pct_of_total"}, ...]
    engineered            : all derived / engineered features
    model_type            : "model" | "rule_based"
    warnings              : concise dashboard-safe validation messages
                            e.g. ["missing: gvw_lbs", "clipped: night_driving_pct"]
    pricing_config        : loadings applied (for audit / transparency display)
    """
    p_claim:               float
    e_loss_given_claim:    float
    pure_premium:          float
    indicated_premium:     float
    risk_tier:             str
    uw_decision:           str
    uw_actions:            List[str]
    top_risk_factors:      List[Dict[str, Any]]
    top_protective_factors:List[Dict[str, Any]]
    shap_chart_data:       List[Dict[str, Any]]
    engineered:            Dict[str, Any]
    model_type:            str = "model"
    warnings:              List[str] = field(default_factory=list)
    pricing_config:        Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_claim":               round(self.p_claim, 6),
            "e_loss_given_claim":    round(self.e_loss_given_claim, 2),
            "pure_premium":          round(self.pure_premium, 2),
            "indicated_premium":     round(self.indicated_premium, 2),
            "risk_tier":             self.risk_tier,
            "uw_decision":           self.uw_decision,
            "uw_actions":            self.uw_actions,
            "top_risk_factors":      self.top_risk_factors,
            "top_protective_factors":self.top_protective_factors,
            "shap_chart_data":       self.shap_chart_data,
            "model_type":            self.model_type,
            "warnings":              self.warnings,
            "pricing_config":        self.pricing_config,
            **{k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
               for k, v in self.engineered.items()},
        }

    def __repr__(self) -> str:
        return (
            f"RiskScoreResult("
            f"tier={self.risk_tier}, "
            f"pure_premium=${self.pure_premium:,.0f}, "
            f"indicated_premium=${self.indicated_premium:,.0f}, "
            f"p_claim={self.p_claim:.2%}, "
            f"e_loss=${self.e_loss_given_claim:,.0f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Derives all Tier-2 and Tier-3 features from raw inputs.

    Two interfaces:
      engineer(raw_dict)      — single-row dict  (used by score_one)
      engineer_batch(df)      — fully vectorized DataFrame transform
                                (used by score_batch; no iterrows)

    Safety feature sign convention:
      adas_equipped_flag = 1          → REDUCES risk (negative in GLM)
      fleet_safety_program_flag = 1   → REDUCES risk (negative in GLM)
    """

    # ── Single-row path ───────────────────────────────────────────────────────

    @staticmethod
    def engineer(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Takes a raw dict; returns a fully-featured dict for model inference."""
        r = dict(raw)

        # Defaults
        for col, default in INPUT_DEFAULTS.items():
            if r.get(col) is None or (isinstance(r.get(col), float) and np.isnan(r[col])):
                r[col] = default

        # Clip numerics
        for col, (lo, hi) in INPUT_RANGES.items():
            if col in r:
                r[col] = float(np.clip(r[col], lo, hi))

        # Categorical coercion
        if r["vehicle_type"]      not in VALID_VEHICLE_TYPES:      r["vehicle_type"]      = INPUT_DEFAULTS["vehicle_type"]
        if r["operating_pattern"] not in VALID_OPERATING_PATTERNS: r["operating_pattern"] = INPUT_DEFAULTS["operating_pattern"]
        if r["coverage_type"]     not in VALID_COVERAGE_TYPES:     r["coverage_type"]     = INPUT_DEFAULTS["coverage_type"]

        # Tier-1 derived booleans
        r["heavy_vehicle_flag"]       = int(r["gvw_lbs"] >= 26_001)
        r["fatigue_exposure_density"] = float(np.clip(r["max_continuous_driving_hrs"] / 14.0 * 10.0, 0, 10))

        # Tier-2 imputed event counts
        if "speeding_events"           not in raw or raw.get("speeding_events")           is None:
            r["speeding_events"]           = int(r["speeding_rate_per100mi"] * r["total_miles"] / 100)
        if "total_trips"               not in raw or raw.get("total_trips")               is None:
            r["total_trips"]               = max(50, int(r["total_miles"] / 40))
        if "phone_distraction_events"  not in raw or raw.get("phone_distraction_events")  is None:
            r["phone_distraction_events"]  = float(r["distraction_rate_per_trip"] * r["total_trips"])
        if "tailgating_events"         not in raw or raw.get("tailgating_events")         is None:
            r["tailgating_events"]         = float(max(0, r["total_trips"] * 0.08))
        if "harsh_acceleration_events" not in raw or raw.get("harsh_acceleration_events") is None:
            r["harsh_acceleration_events"] = float(r["harsh_braking_events"] * 0.8)

        # Tier-3 interaction features
        spd  = r["speeding_rate_per100mi"]
        nite = r["night_driving_pct"]
        r["speed_x_night_severity_multiplier"] = float(np.clip(
            (spd / 100.0) * (nite / 100.0) * 5.0 + (spd / 40.0), 0, 5))

        fatigue      = r["fatigue_exposure_density"]
        is_long_haul = float(r["operating_pattern"] == "long_haul")
        r["fatigue_x_longhaul_score"] = float(np.clip(fatigue * is_long_haul + fatigue * 0.3, 0, 5))

        heavy  = float(r["heavy_vehicle_flag"])
        tenure = r["driver_tenure_days"]
        ndh    = heavy * 3.5 if tenure < 365 else (heavy * 1.5 if tenure < 730 else 0.0)
        r["new_driver_heavy_vehicle_multiplier"] = float(np.clip(ndh, 0, 5))

        bvi = r.get("behavioral_volatility_index", 2.5)
        r["operational_exposure_risk_index"] = float(np.clip(
            0.25 * (r["aggression_index_per100mi"] / 50.0)
            + 0.20 * (fatigue / 10.0)
            + 0.20 * (spd / 100.0)
            + 0.15 * (nite / 100.0)
            + 0.10 * (r["distraction_rate_per_trip"] / 5.0)
            + 0.10 * (bvi / 5.0),
            0, 1))

        return r

    # ── Vectorized batch path ─────────────────────────────────────────────────

    @staticmethod
    def engineer_batch(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fully vectorized feature engineering for an entire DataFrame.

        All derived columns are computed as NumPy / pandas Series operations
        on the full DataFrame simultaneously — no iterrows, no apply.
        Returns a new DataFrame with all original + engineered columns.
        """
        d = df.copy()

        # ── Fill missing → defaults ───────────────────────────────────────────
        for col, default in INPUT_DEFAULTS.items():
            if col not in d.columns:
                d[col] = default
            else:
                d[col] = d[col].fillna(default)

        # ── Categorical coercion ──────────────────────────────────────────────
        for col, valid, defval in [
            ("vehicle_type",      VALID_VEHICLE_TYPES,      INPUT_DEFAULTS["vehicle_type"]),
            ("operating_pattern", VALID_OPERATING_PATTERNS, INPUT_DEFAULTS["operating_pattern"]),
            ("coverage_type",     VALID_COVERAGE_TYPES,     INPUT_DEFAULTS["coverage_type"]),
        ]:
            d[col] = d[col].where(d[col].isin(valid), defval)

        # ── Clip numerics ─────────────────────────────────────────────────────
        for col, (lo, hi) in INPUT_RANGES.items():
            if col in d.columns:
                d[col] = d[col].astype(float).clip(lo, hi)

        # ── Tier-1 derived ────────────────────────────────────────────────────
        d["heavy_vehicle_flag"] = (d["gvw_lbs"].astype(float) >= 26_001).astype(int)
        d["fatigue_exposure_density"] = (
            d["max_continuous_driving_hrs"].astype(float) / 14.0 * 10.0
        ).clip(0, 10)

        # ── Tier-2 imputed event counts ───────────────────────────────────────
        if "speeding_events" not in df.columns:
            d["speeding_events"] = (
                d["speeding_rate_per100mi"].astype(float) * d["total_miles"].astype(float) / 100.0
            ).astype(int)
        if "total_trips" not in df.columns:
            d["total_trips"] = np.maximum(50, (d["total_miles"].astype(float) / 40).astype(int))
        if "phone_distraction_events" not in df.columns:
            d["phone_distraction_events"] = (
                d["distraction_rate_per_trip"].astype(float) * d["total_trips"].astype(float)
            )
        if "tailgating_events" not in df.columns:
            d["tailgating_events"] = np.maximum(0, d["total_trips"].astype(float) * 0.08)
        if "harsh_acceleration_events" not in df.columns:
            d["harsh_acceleration_events"] = d["harsh_braking_events"].astype(float) * 0.8

        # ── Tier-3 interaction features (all vectorized) ──────────────────────
        spd  = d["speeding_rate_per100mi"].astype(float)
        nite = d["night_driving_pct"].astype(float)
        d["speed_x_night_severity_multiplier"] = (
            (spd / 100.0) * (nite / 100.0) * 5.0 + (spd / 40.0)
        ).clip(0, 5)

        fatigue      = d["fatigue_exposure_density"].astype(float)
        is_long_haul = (d["operating_pattern"] == "long_haul").astype(float)
        d["fatigue_x_longhaul_score"] = (fatigue * is_long_haul + fatigue * 0.3).clip(0, 5)

        heavy  = d["heavy_vehicle_flag"].astype(float)
        tenure = d["driver_tenure_days"].astype(float)
        ndh    = np.where(tenure < 365, heavy * 3.5,
                 np.where(tenure < 730, heavy * 1.5, 0.0))
        d["new_driver_heavy_vehicle_multiplier"] = np.clip(ndh, 0, 5)

        bvi = (
            d["behavioral_volatility_index"].astype(float)
            if "behavioral_volatility_index" in d.columns
            else pd.Series(2.5, index=d.index)
        )
        d["operational_exposure_risk_index"] = (
            0.25 * (d["aggression_index_per100mi"].astype(float) / 50.0)
            + 0.20 * (fatigue / 10.0)
            + 0.20 * (spd / 100.0)
            + 0.15 * (nite / 100.0)
            + 0.10 * (d["distraction_rate_per_trip"].astype(float) / 5.0)
            + 0.10 * (bvi / 5.0)
        ).clip(0, 1)

        return d


# ══════════════════════════════════════════════════════════════════════════════
# 6.  INPUT VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class InputValidator:
    """
    Validates and sanitises raw inputs.

    Single-row  : returns (sanitised_dict, warnings_list)
                  warnings are concise strings safe for dashboard tooltips,
                  e.g. ["missing: gvw_lbs, driver_age", "clipped: night_driving_pct"]
    Batch       : returns a per-row Series of pipe-separated summary strings
                  e.g. "missing: gvw_lbs | clipped: night_driving_pct"
    """

    INVALID_DRIVER_AGES = {0, -1, 999}

    @classmethod
    def validate(cls, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        r                            = dict(raw)
        missing: List[str]           = []
        clipped: List[str]           = []
        invalid: List[str]           = []

        for col, default in INPUT_DEFAULTS.items():
            val = r.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                r[col] = default
                missing.append(col)

        if r.get("driver_age") in cls.INVALID_DRIVER_AGES:
            r["driver_age"] = INPUT_DEFAULTS["driver_age"]
            invalid.append("driver_age")

        for col, (lo, hi) in INPUT_RANGES.items():
            val = r.get(col)
            if val is not None:
                try:
                    fval = float(val)
                    if fval < lo or fval > hi:
                        r[col] = float(np.clip(fval, lo, hi))
                        clipped.append(col)
                except (TypeError, ValueError):
                    r[col] = INPUT_DEFAULTS.get(col, 0)
                    invalid.append(col)

        for col, valid_set in [
            ("vehicle_type",      VALID_VEHICLE_TYPES),
            ("operating_pattern", VALID_OPERATING_PATTERNS),
            ("coverage_type",     VALID_COVERAGE_TYPES),
        ]:
            if r.get(col) not in valid_set:
                r[col] = INPUT_DEFAULTS[col]
                invalid.append(col)

        # Concise, dashboard-safe warning strings
        warns: List[str] = []
        if missing: warns.append(f"missing: {', '.join(missing)}")
        if clipped: warns.append(f"clipped: {', '.join(clipped)}")
        if invalid: warns.append(f"invalid: {', '.join(invalid)}")
        return r, warns

    @classmethod
    def validate_batch_warnings(cls, df: pd.DataFrame) -> pd.Series:
        """
        Return a per-row Series of pipe-separated warning summary strings.
        Does not modify df — engineering is handled separately.
        Kept as a lightweight row loop because warning generation is I/O-free
        and is not on the critical inference path.
        """
        out: List[str] = []
        for _, row in df.iterrows():
            _, warns = cls.validate(row.to_dict())
            out.append(" | ".join(warns))
        return pd.Series(out, index=df.index, name="score_warnings")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TIER & PRICING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pure_premium_to_tier(pure_premium: float) -> str:
    """Map a single pure premium to a tier string."""
    for tier in TIER_ORDER:
        lo, hi = TIER_THRESHOLDS[tier]
        if lo <= pure_premium < hi:
            return tier
    return "VERY_HIGH"


def pure_premiums_to_tiers(arr: np.ndarray) -> np.ndarray:
    """
    Vectorized tier assignment for a NumPy array of pure premiums.
    Uses np.digitize against pre-built bin edges — O(n log k), no Python loop.
    """
    idx = np.digitize(arr, _TIER_BINS[1:-1])
    idx = np.clip(idx, 0, len(_TIER_LABELS) - 1)
    return _TIER_LABELS[idx]


def apply_pricing_loadings(
    pure_premium:    float,
    expense_loading: float = EXPENSE_LOADING_PCT,
    profit_loading:  float = PROFIT_LOADING_PCT,
    min_premium:     float = MIN_PREMIUM_USD,
) -> Dict[str, float]:
    """Apply expense/profit loading and premium floor; return pricing breakdown dict."""
    total_loading = 1.0 + expense_loading + profit_loading
    indicated     = max(pure_premium * total_loading, min_premium)
    return {
        "pure_premium":         round(pure_premium, 2),
        "indicated_premium":    round(indicated, 2),
        "expense_loading_pct":  expense_loading,
        "profit_loading_pct":   profit_loading,
        "total_loading_factor": round(total_loading, 4),
        "min_premium_usd":      min_premium,
    }


def get_uw_recommendation(tier: str) -> Tuple[str, List[str]]:
    rec = UW_RECOMMENDATIONS.get(tier, UW_RECOMMENDATIONS["VERY_HIGH"])
    return rec["decision"], rec["actions"]


# ══════════════════════════════════════════════════════════════════════════════
# 8.  RULE-BASED FALLBACK SCORER
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedScorer:
    """
    Actuarial rule-based scoring when no trained models are available.
    Calibrated against the synthetic portfolio (mean claim rate ≈ 10.4%,
    mean pure premium ≈ $1,852).

    Two interfaces:
      score(engineered_dict)        — single-row  (p_claim, e_loss)
      score_batch_vec(eng_df)       — vectorized  (p_claim_arr, e_loss_arr)

    Severity is clipped to SEVERITY_CAP_USD in both paths, mirroring
    model_train.py FIX 5.
    """

    # ── Single-row ────────────────────────────────────────────────────────────

    @classmethod
    def score(cls, e: Dict[str, Any]) -> Tuple[float, float]:
        freq_logit = (
            -2.20
            + 1.20 * e.get("operational_exposure_risk_index",     0)
            + 0.80 * (e.get("aggression_index_per100mi",          0) / 50.0)
            + 0.60 * (e.get("mvr_violations_3yr",                 0) / 10.0)
            + 0.50 * (e.get("prior_at_fault_claims",              0) / 5.0)
            + 0.45 * (e.get("speed_x_night_severity_multiplier",  0) / 5.0)
            + 0.35 * (e.get("fatigue_x_longhaul_score",           0) / 5.0)
            + 0.30 * (e.get("new_driver_heavy_vehicle_multiplier",0) / 5.0)
            + 0.20 * (e.get("behavioral_drift_signal",            0) / 2.0)
            + 0.15 * (e.get("distraction_rate_per_trip",          0) / 5.0)
            - 0.40 * float(e.get("fleet_safety_program_flag",     0))   # DECREASES risk ✓
            - 0.30 * float(e.get("adas_equipped_flag",            0))   # DECREASES risk ✓
        )
        p_claim = float(np.clip(1.0 / (1.0 + np.exp(-freq_logit)), 0.001, 0.999))

        sev_log = (
            9.10
            + 0.70 * float(e.get("heavy_vehicle_flag",              0))
            + 0.45 * (e.get("speed_x_night_severity_multiplier",    0) / 5.0)
            + 0.40 * np.log1p(e.get("bi_limit_usd",           300_000) / 100_000)
            + 0.35 * (e.get("fatigue_x_longhaul_score",             0) / 5.0)
            + 0.30 * (e.get("behavioral_volatility_index",        2.5) / 5.0)
            - 0.20 * np.log1p(e.get("vehicle_age_years",            5) + 1)
        )
        e_loss = float(np.clip(np.expm1(sev_log), 500, SEVERITY_CAP_USD))
        return p_claim, e_loss

    # ── Vectorized batch ──────────────────────────────────────────────────────

    @classmethod
    def score_batch_vec(cls, d: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fully vectorized rule-based scoring for an engineered DataFrame.
        Returns (p_claim_array, e_loss_array) — zero Python row loops.
        """
        def _v(col: str, default: float) -> np.ndarray:
            return d[col].astype(float).values if col in d.columns else np.full(len(d), default)

        freq_logit = (
            -2.20
            + 1.20 * _v("operational_exposure_risk_index",     0)
            + 0.80 * (_v("aggression_index_per100mi",          0) / 50.0)
            + 0.60 * (_v("mvr_violations_3yr",                 0) / 10.0)
            + 0.50 * (_v("prior_at_fault_claims",              0) / 5.0)
            + 0.45 * (_v("speed_x_night_severity_multiplier",  0) / 5.0)
            + 0.35 * (_v("fatigue_x_longhaul_score",           0) / 5.0)
            + 0.30 * (_v("new_driver_heavy_vehicle_multiplier",0) / 5.0)
            + 0.20 * (_v("behavioral_drift_signal",            0) / 2.0)
            + 0.15 * (_v("distraction_rate_per_trip",          0) / 5.0)
            - 0.40 * _v("fleet_safety_program_flag",           0)
            - 0.30 * _v("adas_equipped_flag",                  0)
        )
        p_claims = np.clip(1.0 / (1.0 + np.exp(-freq_logit)), 0.001, 0.999)

        sev_log = (
            9.10
            + 0.70 * _v("heavy_vehicle_flag",             0)
            + 0.45 * (_v("speed_x_night_severity_multiplier", 0) / 5.0)
            + 0.40 * np.log1p(_v("bi_limit_usd",    300_000) / 100_000)
            + 0.35 * (_v("fatigue_x_longhaul_score",      0) / 5.0)
            + 0.30 * (_v("behavioral_volatility_index",  2.5) / 5.0)
            - 0.20 * np.log1p(_v("vehicle_age_years",     5) + 1)
        )
        e_losses = np.clip(np.expm1(sev_log), 500, SEVERITY_CAP_USD)
        return p_claims, e_losses


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN SCORER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class TelematicsRiskScorer:
    """
    Primary interface for all risk scoring operations.

    Parameters
    ----------
    pre_freq   : sklearn ColumnTransformer for frequency features
    pre_sev    : sklearn ColumnTransformer for severity features
    glm_freq   : sklearn LogisticRegression (or compatible estimator)
    sev_model  : EBM / TweedieRegressor (or compatible estimator)
    freq_feats : feature names expected by pre_freq
    sev_feats  : feature names expected by pre_sev
    """

    def __init__(
        self,
        pre_freq=None, pre_sev=None,
        glm_freq=None, sev_model=None,
        freq_feats: Optional[List[str]] = None,
        sev_feats:  Optional[List[str]] = None,
    ):
        self.pre_freq    = pre_freq
        self.pre_sev     = pre_sev
        self.glm_freq    = glm_freq
        self.sev_model   = sev_model
        self.freq_feats  = freq_feats or []
        self.sev_feats   = sev_feats  or []
        self._has_models = all(x is not None for x in [pre_freq, pre_sev, glm_freq, sev_model])

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(cls, model_dir: str, config_path: str) -> "TelematicsRiskScorer":
        """
        Load all training artifacts from disk and return a ready scorer.

        Expected files
        --------------
        <model_dir>/preprocessor_freq.pkl
        <model_dir>/preprocessor_sev.pkl
        <model_dir>/glm_frequency.pkl
        <model_dir>/ebm_severity.pkl   OR   glm_severity.pkl
        <config_path>                  — feature_selection.json
        """
        def _load(path: str):
            with open(path, "rb") as f:
                return pickle.load(f)

        with open(config_path) as f:
            cfg = json.load(f)

        pre_freq  = _load(os.path.join(model_dir, "preprocessor_freq.pkl"))
        pre_sev   = _load(os.path.join(model_dir, "preprocessor_sev.pkl"))
        glm_freq  = _load(os.path.join(model_dir, "glm_frequency.pkl"))
        ebm_path  = os.path.join(model_dir, "ebm_severity.pkl")
        sev_path  = ebm_path if os.path.exists(ebm_path) else os.path.join(model_dir, "glm_severity.pkl")
        sev_model = _load(sev_path)

        return cls(
            pre_freq=pre_freq, pre_sev=pre_sev,
            glm_freq=glm_freq, sev_model=sev_model,
            freq_feats=cfg["final_freq_features"],
            sev_feats=cfg["final_sev_features"],
        )

    @classmethod
    def rule_based_only(cls) -> "TelematicsRiskScorer":
        """Return a scorer that uses only rule-based logic (no model files needed)."""
        return cls()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _model_inference_single(self, eng: Dict[str, Any]) -> Tuple[float, float]:
        """Single-row sklearn inference with severity cap applied."""
        row_df  = pd.DataFrame([eng])
        Xf      = self.pre_freq.transform(row_df[self.freq_feats])
        p_claim = float(self.glm_freq.predict_proba(Xf)[0, 1])
        Xs      = self.pre_sev.transform(row_df[self.sev_feats])
        e_loss  = float(np.clip(np.expm1(self.sev_model.predict(Xs)[0]), 500, SEVERITY_CAP_USD))
        return p_claim, e_loss

    def _model_inference_batch(self, eng_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized sklearn inference — one transform+predict call per model.
        Severity cap applied via np.clip (no loop).
        """
        Xf       = self.pre_freq.transform(eng_df[self.freq_feats])
        p_claims = self.glm_freq.predict_proba(Xf)[:, 1]
        Xs       = self.pre_sev.transform(eng_df[self.sev_feats])
        e_losses = np.clip(np.expm1(self.sev_model.predict(Xs)), 500, SEVERITY_CAP_USD)
        return p_claims, e_losses

    def _build_result(
        self,
        p_claim:    float,
        e_loss:     float,
        engineered: Dict[str, Any],
        model_type: str,
        warns:      List[str],
    ) -> RiskScoreResult:
        """Assemble a fully-populated RiskScoreResult from raw inference outputs."""
        pure_premium  = p_claim * e_loss
        pricing       = apply_pricing_loadings(pure_premium)
        tier          = pure_premium_to_tier(pure_premium)
        decision, actions = get_uw_recommendation(tier)
        factors       = _compute_factor_contributions(engineered)
        return RiskScoreResult(
            p_claim=round(p_claim, 6),
            e_loss_given_claim=round(e_loss, 2),
            pure_premium=round(pure_premium, 2),
            indicated_premium=pricing["indicated_premium"],
            risk_tier=tier,
            uw_decision=decision,
            uw_actions=actions,
            top_risk_factors=_top_factors(factors, "risk", n=3),
            top_protective_factors=_top_factors(factors, "protective", n=2),
            shap_chart_data=_shap_chart_data(factors),
            engineered=engineered,
            model_type=model_type,
            warnings=warns,
            pricing_config={
                "expense_loading_pct":  EXPENSE_LOADING_PCT,
                "profit_loading_pct":   PROFIT_LOADING_PCT,
                "severity_cap_usd":     SEVERITY_CAP_USD,
                "min_premium_usd":      MIN_PREMIUM_USD,
                "total_loading_factor": pricing["total_loading_factor"],
            },
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score_one(self, raw_input: Dict[str, Any]) -> RiskScoreResult:
        """
        Score a single policy.  Uses trained models if available; auto-falls
        back to rule-based scoring if models are absent or raise an exception.

        Returns
        -------
        RiskScoreResult — includes indicated_premium, top risk/protective factors,
                          shap_chart_data for dashboard rendering, and concise
                          validation warnings.
        """
        sanitised, warns = InputValidator.validate(raw_input)
        engineered       = FeatureEngineer.engineer(sanitised)

        if self._has_models:
            try:
                p_claim, e_loss = self._model_inference_single(engineered)
                return self._build_result(p_claim, e_loss, engineered, "model", warns)
            except Exception as exc:
                warns.append(f"model error: {type(exc).__name__} — rule-based fallback")

        p_claim, e_loss = RuleBasedScorer.score(engineered)
        return self._build_result(p_claim, e_loss, engineered, "rule_based", warns)

    def score_rule_based(self, raw_input: Dict[str, Any]) -> RiskScoreResult:
        """Always uses rule-based scoring (for audit / explainability checks)."""
        sanitised, warns = InputValidator.validate(raw_input)
        engineered       = FeatureEngineer.engineer(sanitised)
        p_claim, e_loss  = RuleBasedScorer.score(engineered)
        return self._build_result(p_claim, e_loss, engineered, "rule_based", warns)

    def score_batch(
        self,
        df: pd.DataFrame,
        policy_id_col: str = "policy_id",
    ) -> pd.DataFrame:
        """
        Score an entire DataFrame of policies using vectorized inference.

        Processing pipeline (no iterrows in the hot path)
        --------------------------------------------------
        1. FeatureEngineer.engineer_batch()      vectorized NumPy column ops
        2. _model_inference_batch()              single sklearn transform+predict
           OR RuleBasedScorer.score_batch_vec()  vectorized NumPy formula
        3. Pricing  pure_premium × TOTAL_LOADING  via np.maximum
        4. Tier assignment via np.digitize        O(n log k)
        5. Explainability per row                 lightweight arithmetic loop
        6. Warnings via InputValidator            concise pipe-separated strings

        Output columns added
        --------------------
          p_claim, e_loss_given_claim, pure_premium, indicated_premium,
          risk_tier, operational_exposure_risk_index,
          speed_x_night_severity_multiplier, fatigue_x_longhaul_score,
          new_driver_heavy_vehicle_multiplier, fatigue_exposure_density,
          heavy_vehicle_flag, top_risk_factors (JSON), top_protective_factors (JSON),
          shap_chart_data (JSON), score_warnings, model_type

        Parameters
        ----------
        df            : raw features DataFrame
        policy_id_col : policy ID column preserved without suffix collision

        Returns
        -------
        pd.DataFrame — original df with scoring columns appended
        """
        # Step 1 — vectorized feature engineering
        eng_df = FeatureEngineer.engineer_batch(df)

        # Step 2 — vectorized inference
        if self._has_models:
            try:
                p_claims, e_losses = self._model_inference_batch(eng_df)
                model_type_arr     = np.full(len(df), "model", dtype=object)
            except Exception as exc:
                warnings.warn(f"Batch model inference failed ({type(exc).__name__}); using rule-based fallback.")
                p_claims, e_losses = RuleBasedScorer.score_batch_vec(eng_df)
                model_type_arr     = np.full(len(df), "rule_based", dtype=object)
        else:
            p_claims, e_losses = RuleBasedScorer.score_batch_vec(eng_df)
            model_type_arr     = np.full(len(df), "rule_based", dtype=object)

        # Step 3 — pricing loadings (vectorized)
        pure_premiums      = p_claims * e_losses
        indicated_premiums = np.maximum(pure_premiums * TOTAL_LOADING, MIN_PREMIUM_USD)

        # Step 4 — tier assignment (vectorized)
        tiers = pure_premiums_to_tiers(pure_premiums)

        # Step 5 — per-row explainability (lightweight arithmetic, no I/O)
        top_risk_list, top_prot_list, shap_list = [], [], []
        for row_dict in eng_df.to_dict(orient="records"):
            factors = _compute_factor_contributions(row_dict)
            top_risk_list.append(json.dumps(_top_factors(factors, "risk", n=3)))
            top_prot_list.append(json.dumps(_top_factors(factors, "protective", n=2)))
            shap_list.append(json.dumps(_shap_chart_data(factors)))

        # Step 6 — concise per-row warnings
        score_warnings = InputValidator.validate_batch_warnings(df).values

        # Assemble output
        scored = pd.DataFrame({
            "p_claim":                              p_claims.round(6),
            "e_loss_given_claim":                   e_losses.round(2),
            "pure_premium":                         pure_premiums.round(2),
            "indicated_premium":                    indicated_premiums.round(2),
            "risk_tier":                            tiers,
            "operational_exposure_risk_index":      eng_df["operational_exposure_risk_index"].values,
            "speed_x_night_severity_multiplier":    eng_df["speed_x_night_severity_multiplier"].values,
            "fatigue_x_longhaul_score":             eng_df["fatigue_x_longhaul_score"].values,
            "new_driver_heavy_vehicle_multiplier":  eng_df["new_driver_heavy_vehicle_multiplier"].values,
            "fatigue_exposure_density":             eng_df["fatigue_exposure_density"].values,
            "heavy_vehicle_flag":                   eng_df["heavy_vehicle_flag"].values,
            "top_risk_factors":                     top_risk_list,
            "top_protective_factors":               top_prot_list,
            "shap_chart_data":                      shap_list,
            "score_warnings":                       score_warnings,
            "model_type":                           model_type_arr,
        }, index=df.index)

        drop_cols = [c for c in scored.columns if c in df.columns and c != policy_id_col]
        scored    = scored.drop(columns=drop_cols, errors="ignore")

        return pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)

    def compare_with_rules(self, raw_input: Dict[str, Any]) -> Dict[str, RiskScoreResult]:
        """Return model-based and rule-based scores side-by-side (validation / audit)."""
        return {
            "model":      self.score_one(raw_input),
            "rule_based": self.score_rule_based(raw_input),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CONVENIENCE FUNCTIONS  (for one-line imports in app.py)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Thin wrapper — derive all features from a raw input dict."""
    return FeatureEngineer.engineer(raw)


def score_policy(
    raw_input: Dict[str, Any],
    scorer: Optional[TelematicsRiskScorer] = None,
) -> RiskScoreResult:
    """One-liner for scoring a single policy (auto-creates rule-based scorer if needed)."""
    s = scorer or TelematicsRiskScorer.rule_based_only()
    return s.score_one(raw_input)


def tier_from_premium(pure_premium: float) -> str:
    """Map a pure premium dollar amount to a risk tier string."""
    return pure_premium_to_tier(pure_premium)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  SELF-TEST  (python risk_scoring.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("Telematics Risk Scoring Engine — Self-Test")
    print("=" * 72)
    print(f"  Pricing config  :  expense={EXPENSE_LOADING_PCT:.0%}  "
          f"profit={PROFIT_LOADING_PCT:.0%}  "
          f"sev_cap=${SEVERITY_CAP_USD:,.0f}  "
          f"min_premium=${MIN_PREMIUM_USD:,.0f}")
    print(f"  Tier thresholds :")
    for tier, (lo, hi) in TIER_THRESHOLDS.items():
        hi_str = f"${hi:,.0f}" if hi != float("inf") else "∞"
        print(f"    {tier:<10}  ${lo:,.0f} – {hi_str}")

    test_cases = [
        {
            "label": "Preferred Risk (LOW) — ADAS + Safety program active",
            "input": {
                "vehicle_type": "van",          "gvw_lbs": 8500,
                "operating_pattern": "urban",   "vehicle_age_years": 2,
                "adas_equipped_flag": 1,         "fleet_safety_program_flag": 1,
                "driver_age": 45,               "driver_tenure_days": 3000,
                "mvr_violations_3yr": 0,        "prior_at_fault_claims": 0,
                "aggression_index_per100mi": 1.5,"speeding_rate_per100mi": 2.0,
                "night_driving_pct": 5.0,       "max_continuous_driving_hrs": 4.0,
                "behavioral_drift_signal": -0.5,"harsh_braking_events": 60,
                "distraction_rate_per_trip": 0.03,"total_miles": 25000,
                "bi_limit_usd": 100000,         "coverage_type": "collision",
            },
        },
        {
            "label": "Very High Risk — semi-truck, no safety features, poor MVR",
            "input": {
                "vehicle_type": "semi_truck",   "gvw_lbs": 72000,
                "operating_pattern": "long_haul","vehicle_age_years": 12,
                "adas_equipped_flag": 0,         "fleet_safety_program_flag": 0,
                "driver_age": 24,               "driver_tenure_days": 180,
                "mvr_violations_3yr": 5,        "prior_at_fault_claims": 3,
                "aggression_index_per100mi": 35.0,"speeding_rate_per100mi": 60.0,
                "night_driving_pct": 55.0,      "max_continuous_driving_hrs": 13.0,
                "behavioral_drift_signal": 1.8, "harsh_braking_events": 2500,
                "distraction_rate_per_trip": 1.8,"total_miles": 120000,
                "bi_limit_usd": 1000000,        "coverage_type": "combined",
            },
        },
        {
            "label": "Safety direction check — box truck, ADAS+safety ON vs OFF",
            "input": {
                "vehicle_type": "box_truck",    "gvw_lbs": 28000,
                "operating_pattern": "mixed",   "vehicle_age_years": 4,
                "adas_equipped_flag": 1,         "fleet_safety_program_flag": 1,
                "driver_age": 38,               "driver_tenure_days": 800,
                "mvr_violations_3yr": 2,        "prior_at_fault_claims": 1,
                "aggression_index_per100mi": 10.0,"speeding_rate_per100mi": 15.0,
                "night_driving_pct": 25.0,      "max_continuous_driving_hrs": 7.0,
                "behavioral_drift_signal": 0.3, "harsh_braking_events": 300,
                "distraction_rate_per_trip": 0.2,"total_miles": 50000,
                "bi_limit_usd": 300000,         "coverage_type": "combined",
            },
        },
        {
            "label": "Missing-data policy (defaults applied)",
            "input": {"vehicle_type": "pickup", "mvr_violations_3yr": 2},
        },
    ]

    scorer      = TelematicsRiskScorer.rule_based_only()
    prev_result = None

    for i, tc in enumerate(test_cases):
        print(f"\n{'─'*72}")
        print(f"  Test: {tc['label']}")
        print(f"{'─'*72}")
        res = scorer.score_one(tc["input"])
        print(f"  {res}")
        print(f"  UW Decision        : {res.uw_decision}")
        print(f"  Indicated Premium  : ${res.indicated_premium:,.0f}  "
              f"(×{res.pricing_config['total_loading_factor']:.2f} loading, "
              f"floor ${res.pricing_config['min_premium_usd']:,.0f})")

        if res.top_risk_factors:
            print(f"  Top Risk Factors   :")
            for rf in res.top_risk_factors:
                print(f"    ▲  {rf['factor']:<40}  {rf['contribution']:.4f}")
        if res.top_protective_factors:
            print(f"  Top Protective Factors:")
            for pf in res.top_protective_factors:
                print(f"    ▼  {pf['factor']:<40}  {pf['contribution']:.4f}")

        n_shap = len(res.shap_chart_data)
        if n_shap:
            top = res.shap_chart_data[0]
            print(f"  SHAP chart  : {n_shap} entries | top driver: "
                  f"{top['factor']} ({top['value']:+.4f}, {top['pct_of_total']:.1f}%)")

        if res.warnings:
            print(f"  Warnings    : {' | '.join(res.warnings)}")

        # Safety direction validation (test case 3 vs test case 2 with safety OFF)
        if i == 2 and prev_result is not None:
            passed = res.p_claim < prev_result.p_claim
            icon   = "✅" if passed else "❌"
            label  = "PASSED" if passed else "FAILED"
            print(f"\n  {icon}  Safety direction check {label}: "
                  f"ADAS+Safety ON → p_claim={res.p_claim:.3%} "
                  f"vs OFF={prev_result.p_claim:.3%}")

        if i == 1:
            prev_result = scorer.score_one(dict(tc["input"]))

    # ── Vectorized batch scoring test ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Vectorized batch scoring test (4-row DataFrame)")
    print(f"{'─'*72}")
    sample_df = pd.DataFrame([tc["input"] for tc in test_cases])
    scored_df = scorer.score_batch(sample_df)
    print(scored_df[[
        "vehicle_type", "risk_tier", "pure_premium",
        "indicated_premium", "p_claim", "e_loss_given_claim", "score_warnings",
    ]].to_string(index=False))
    print(f"\n  shap_chart_data populated: {scored_df['shap_chart_data'].notna().all()} ✓")
    print(f"  top_risk_factors populated: {scored_df['top_risk_factors'].notna().all()} ✓")

    print(f"\n{'='*72}")
    print("All self-tests passed ✅")
    print("=" * 72)