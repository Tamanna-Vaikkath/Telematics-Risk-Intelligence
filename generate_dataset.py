"""
Telematics Risk Intelligence — Synthetic Data Generator  v3
============================================================
Changes from v2
---------------
1. CATASTROPHIC TAIL — TRUCKING REALISM
   ~2% of heavy-vehicle claims are escalated to a catastrophic severity
   layer (semi/straight-truck + high-risk exposure combinations).
   These represent cargo spills, rollovers, multi-vehicle crashes, and
   major BI settlements that characterise real commercial trucking loss
   development.  The base lognormal is unchanged (σ = 0.65); the cat
   layer uses a separate Pareto-shaped draw mixed in post-hoc, keeping
   the bulk of the severity distribution realistic while producing a
   credible right tail ($500 K – $5 M range) for a small subset.

2. NEW UNDERWRITING / REAL-WORLD FIELDS
   ┌─────────────────────────┬────────────────────────────────────────────┐
   │ Column                  │ Notes                                      │
   ├─────────────────────────┼────────────────────────────────────────────┤
   │ policy_term_months      │ 3 / 6 / 12 — drives earned_exposure cap    │
   │ fleet_size              │ number of vehicles on policy               │
   │ cargo_type              │ general / refrigerated / hazmat /          │
   │                         │ oversized / livestock / dry_van            │
   │ cargo_value_usd         │ declared cargo value (correlated to type   │
   │                         │ and fleet_size)                            │
   │ region_weather_risk     │ low / moderate / high / severe             │
   └─────────────────────────┴────────────────────────────────────────────┘
   hazmat / oversized / severe-weather records receive a severity premium
   and a small frequency lift.  OERI formula is unchanged so tier
   calibration is preserved.

3. EXPOSURE AS SINGLE SOURCE OF TRUTH  (hardened)
   earned_exposure is now bounded by policy_term_years, not a fixed 1.0.
   The three canonical KPI columns are the ONLY legitimate definitions
   in this project.  Public constants EXPOSURE_COLS and KPI_COLS name
   them so downstream code never hard-codes the strings.

Run:  python generate_dataset.py
Output: telematics_raw.csv

np.random.seed(42) — fully deterministic.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ── Public contracts — import these in dashboards / models ────────────────────
EXPOSURE_COLS: tuple[str, ...] = ("vehicle_years", "earned_exposure")
KPI_COLS: tuple[str, ...] = ("claim_frequency", "loss_per_vehicle_year", "pure_premium")
TIER_COL: str = "uw_risk_tier"

# ── Fixed seed ────────────────────────────────────────────────────────────────
np.random.seed(42)
N = 10_000


def rc(options, probs, n: int = N) -> np.ndarray:
    """np.random.choice with an explicit probability vector."""
    return np.random.choice(options, size=n, p=probs)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FLEET / VEHICLE / DRIVER CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
state_of_domicile = rc(
    ["CA", "TX", "FL", "NY", "IL", "GA", "PA", "OH", "TN", "AZ"],
    [0.18, 0.17, 0.12, 0.11, 0.09, 0.08, 0.08, 0.07, 0.05, 0.05],
)

operating_pattern = rc(
    ["urban", "highway", "mixed", "long_haul"],
    [0.28, 0.22, 0.35, 0.15],
)

vehicle_type = rc(
    ["pickup", "van", "box_truck", "semi_truck", "straight_truck"],
    [0.18, 0.22, 0.25, 0.20, 0.15],
)

_gvw_base = {
    "pickup": 6_000, "van": 8_500, "box_truck": 26_000,
    "semi_truck": 70_000, "straight_truck": 33_000,
}
gvw_lbs = np.array([
    np.clip(np.random.normal(_gvw_base[v], 3_000), 3_000, 80_000)
    for v in vehicle_type
]).astype(int)

heavy_vehicle_flag    = (gvw_lbs >= 26_001).astype(int)
vehicle_age_years     = np.round(np.clip(
    np.random.exponential(5, N) + np.random.normal(0, 0.8, N), 0, 20), 1)
adas_equipped_flag    = (
    np.random.rand(N) < np.where(vehicle_age_years < 5, 0.55, 0.20)
).astype(int)
driver_tenure_days    = np.round(np.clip(
    np.random.exponential(800, N) + np.random.normal(0, 50, N), 0, 10_000)
).astype(int)
driver_age            = np.round(np.clip(
    np.random.normal(42, 10, N) + np.random.normal(0, 0.5, N), 21, 75)
).astype(int)
mvr_violations_3yr    = np.round(np.clip(
    np.random.poisson(0.9, N) + np.random.binomial(2, 0.15, N), 0, 15)
).astype(int)
prior_at_fault_claims = np.round(np.clip(
    np.random.poisson(0.4, N) + np.random.binomial(1, 0.10, N), 0, 10)
).astype(int)
fleet_safety_program_flag = (np.random.rand(N) < 0.55).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  NEW UNDERWRITING FIELDS  (v3)
# ══════════════════════════════════════════════════════════════════════════════

# ── Policy term ───────────────────────────────────────────────────────────────
# Commercial fleets: 70% annual, 20% semi-annual, 10% quarterly.
policy_term_months = rc([3, 6, 12], [0.10, 0.20, 0.70])
policy_term_years  = policy_term_months / 12.0   # float used in exposure formula

# ── Fleet size ────────────────────────────────────────────────────────────────
# Log-normal: median ~5 units, long right tail to 200+ for large carriers.
fleet_size = np.round(np.clip(
    np.random.lognormal(1.6, 1.1, N) + np.random.normal(0, 0.5, N),
    1, 300)
).astype(int)

# ── Cargo type — correlated with vehicle_type ─────────────────────────────────
_cargo_probs = {
    #                gen    refrig  hazmat  oversize  livestock  dry_van
    "pickup":       [0.55,  0.05,   0.05,   0.05,     0.05,      0.25],
    "van":          [0.35,  0.20,   0.05,   0.03,     0.02,      0.35],
    "box_truck":    [0.30,  0.20,   0.08,   0.05,     0.05,      0.32],
    "semi_truck":   [0.20,  0.15,   0.15,   0.12,     0.10,      0.28],
    "straight_truck":[0.25, 0.18,   0.12,   0.10,     0.08,      0.27],
}
_cargo_labels = ["general", "refrigerated", "hazmat", "oversized", "livestock", "dry_van"]
cargo_type = np.array([
    np.random.choice(_cargo_labels, p=_cargo_probs[v])
    for v in vehicle_type
])

# ── Region weather risk — derived from state ──────────────────────────────────
_state_weather = {
    "CA": "low",     "TX": "moderate", "FL": "high",    "NY": "high",
    "IL": "severe",  "GA": "moderate", "PA": "high",    "OH": "severe",
    "TN": "moderate","AZ": "low",
}
region_weather_risk = np.array([_state_weather[s] for s in state_of_domicile])

# ── Declared cargo value — correlated with cargo_type and fleet_size ──────────
_cargo_value_mean = {
    "general": 40_000, "refrigerated": 80_000, "hazmat": 120_000,
    "oversized": 150_000, "livestock": 70_000, "dry_van": 50_000,
}
cargo_value_base = np.array([_cargo_value_mean[c] for c in cargo_type], dtype=float)
# Larger fleets tend to insure higher-value loads
cargo_value_usd = np.round(np.clip(
    np.random.lognormal(np.log(cargo_value_base), 0.40, N)
    * np.clip(np.log1p(fleet_size) / np.log1p(5), 0.5, 3.0),
    5_000, 2_000_000)
).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TELEMATICS — TIER-1 RAW
# ══════════════════════════════════════════════════════════════════════════════
total_miles = np.round(np.clip(
    np.random.lognormal(10.5, 0.55, N) + np.random.normal(0, 500, N),
    1_000, 150_000)
).astype(int)
total_trips = np.round(np.clip(
    total_miles / np.random.uniform(25, 80, N) + np.random.normal(0, 15, N),
    50, 5_000)
).astype(int)

risk_mult = (
    1.0
    + 0.30 * (mvr_violations_3yr / 5)
    + 0.25 * (prior_at_fault_claims / 3)
    + 0.20 * (1 - fleet_safety_program_flag)
    + 0.15 * (driver_age < 28).astype(int)
)

harsh_braking_events      = np.round(np.clip(
    np.random.poisson(total_trips * 0.25 * risk_mult, N)
    + np.random.normal(0, 20, N), 0, 5_000)).astype(int)
harsh_acceleration_events = np.round(np.clip(
    np.random.poisson(total_trips * 0.20 * risk_mult, N)
    + np.random.normal(0, 15, N), 0, 5_000)).astype(int)
speeding_events           = np.round(np.clip(
    np.random.poisson(total_miles * 0.03 * risk_mult, N)
    + np.random.normal(0, 50, N), 0, 10_000)).astype(int)
phone_distraction_events  = np.round(np.clip(
    np.random.poisson(total_trips * 0.10 * risk_mult, N)
    + np.random.normal(0, 8, N), 0, 3_000)).astype(int)
tailgating_events         = np.round(np.clip(
    np.random.poisson(total_trips * 0.08 * risk_mult, N)
    + np.random.normal(0, 5, N), 0, 2_000)).astype(int)

night_driving_pct = np.round(np.clip(
    np.where(operating_pattern == "long_haul",
             np.random.beta(3, 5, N) * 100,
             np.random.beta(1.5, 7, N) * 100)
    + np.random.normal(0, 1.5, N), 0, 100), 1)

max_continuous_driving_hrs = np.round(np.clip(
    np.where(operating_pattern == "long_haul",
             np.random.normal(9, 2, N),
             np.random.normal(5, 2, N))
    + np.random.normal(0, 0.3, N), 0, 14), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DERIVED FEATURES — TIER-2
# ══════════════════════════════════════════════════════════════════════════════
aggression_index_per100mi = np.round(np.clip(
    (harsh_braking_events + harsh_acceleration_events + speeding_events * 0.3)
    / (total_miles / 100 + 1e-6)
    + np.random.normal(0, 0.5, N), 0, 50), 2)

speeding_rate_per100mi    = np.round(np.clip(
    speeding_events / (total_miles / 100 + 1e-6)
    + np.random.normal(0, 0.8, N), 0, 100), 2)
fatigue_exposure_density  = np.round(np.clip(
    max_continuous_driving_hrs / 14 * 10
    + np.random.normal(0, 0.3, N), 0, 10), 2)
distraction_rate_per_trip = np.round(np.clip(
    phone_distraction_events / (total_trips + 1e-6)
    + np.random.normal(0, 0.05, N), 0, 5), 3)

behavioral_volatility_index = np.round(np.clip(
    np.std(np.column_stack([
        harsh_braking_events / (total_trips + 1),
        speeding_events / (total_miles + 1) * 100,
        phone_distraction_events / (total_trips + 1),
    ]), axis=1) * 5
    + np.random.normal(0, 0.1, N), 0, 5), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INTERACTION FEATURES — TIER-3
# ══════════════════════════════════════════════════════════════════════════════
speed_x_night_severity_multiplier = np.round(np.clip(
    (speeding_rate_per100mi / 100) * (night_driving_pct / 100) * 5
    + (speeding_rate_per100mi / 40)
    + np.random.normal(0, 0.05, N), 0, 5), 3)

fatigue_x_longhaul_score = np.round(np.clip(
    fatigue_exposure_density * (operating_pattern == "long_haul").astype(float)
    + fatigue_exposure_density * 0.3
    + np.random.normal(0, 0.1, N), 0, 5), 3)

new_driver_heavy_vehicle_multiplier = np.round(np.clip(
    (driver_tenure_days < 365).astype(float) * heavy_vehicle_flag
    * np.random.uniform(2.5, 5, N)
    + (driver_tenure_days < 730).astype(float) * heavy_vehicle_flag
    * np.random.uniform(1, 2, N)
    + np.random.normal(0, 0.05, N), 0, 5), 3)

behavioral_drift_signal = np.round(np.clip(
    np.random.normal(0, 0.5, N)
    + 0.3 * (mvr_violations_3yr > 3).astype(float)
    - 0.2 * fleet_safety_program_flag,
    -2, 2), 3)

operational_exposure_risk_index = np.round(np.clip(
    0.25 * (aggression_index_per100mi / 50)
    + 0.20 * (fatigue_exposure_density / 10)
    + 0.20 * (speeding_rate_per100mi / 100)
    + 0.15 * (night_driving_pct / 100)
    + 0.10 * (distraction_rate_per_trip / 5)
    + 0.10 * (behavioral_volatility_index / 5)
    + np.random.normal(0, 0.02, N), 0, 1), 4)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  POLICY / INSURANCE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
bi_limit_usd = np.round(rc(
    [50_000, 100_000, 300_000, 500_000, 1_000_000, 2_000_000],
    [0.05, 0.20, 0.30, 0.25, 0.15, 0.05],
) * np.random.uniform(0.95, 1.05, N)).astype(int)

collision_deductible_usd = rc(
    [250, 500, 1_000, 2_500, 5_000], [0.10, 0.30, 0.35, 0.20, 0.05])
coverage_type = rc(
    ["liability_only", "collision", "comprehensive", "combined"],
    [0.15, 0.25, 0.20, 0.40])


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EXPOSURE VARIABLES — SINGLE SOURCE OF TRUTH  (v3 hardened)
# ══════════════════════════════════════════════════════════════════════════════
# Annual mileage norms (miles/year at full utilisation, industry-typical)
_annual_mile_norm: dict[str, float] = {
    "pickup":         25_000.0,
    "van":            30_000.0,
    "box_truck":      35_000.0,
    "semi_truck":    100_000.0,
    "straight_truck": 60_000.0,
}
annual_norm = np.array([_annual_mile_norm[v] for v in vehicle_type], dtype=float)

# vehicle_years: fractional policy year implied by observed mileage.
#   Upper bound = policy_term_years (a 3-month policy cannot exceed 0.25 VY).
#   Lower bound = 0.05 to prevent zero-division in KPI denominators.
vehicle_years = np.round(
    np.clip(total_miles / annual_norm, 0.05, policy_term_years), 4
)

# earned_exposure: identical alias kept for dashboard semantic clarity.
#   Actuarial reports → "vehicle_years"; dashboards / KPIs → "earned_exposure".
#   Both columns always hold the same values.
earned_exposure = vehicle_years.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TARGET VARIABLES — CALIBRATED
# ══════════════════════════════════════════════════════════════════════════════
# Calibration targets (approximate, before noise):
#   Preferred  (OERI ≤ p40 ≈ 0.216)  →  ~4–6%
#   Standard   (p40–p70)             →  ~8–12%
#   High-Risk  (> p70 ≈ 0.255)       →  ~18–24%
#
# Intercept and OERI coefficient derived analytically from target logits
# (see v2 notes).  New-field contributions are small (+0.05–0.10) and
# directionally correct; they do not materially shift the tier calibration.

_is_hazmat_oversized = np.isin(cargo_type, ["hazmat", "oversized"]).astype(float)
_is_high_weather     = np.isin(region_weather_risk, ["high", "severe"]).astype(float)
_fleet_size_logit    = 0.05 * np.log1p(fleet_size) / np.log1p(50)

claim_prob_logit = (
    -6.32
    + 16.4  * operational_exposure_risk_index      # primary signal
    + 0.35  * (aggression_index_per100mi / 50)
    + 0.25  * (mvr_violations_3yr / 10)
    + 0.20  * (prior_at_fault_claims / 5)
    + 0.20  * (speed_x_night_severity_multiplier / 5)
    + 0.12  * (fatigue_x_longhaul_score / 5)
    + 0.12  * (new_driver_heavy_vehicle_multiplier / 5)
    + 0.08  * np.where(np.isnan(behavioral_drift_signal), 0.0, behavioral_drift_signal)
    + 0.10  * _is_hazmat_oversized                 # cargo risk lift
    + 0.08  * _is_high_weather                     # weather risk lift
    + _fleet_size_logit                            # fleet size (small)
    - 0.20  * fleet_safety_program_flag            # protective
    - 0.12  * adas_equipped_flag                   # protective
    + np.random.logistic(0, 0.35, N)               # irreducible noise
)
claim_prob     = 1.0 / (1.0 + np.exp(-claim_prob_logit))
had_claim_flag = (np.random.rand(N) < claim_prob).astype(int)

num_claims_12mo = np.where(
    had_claim_flag == 1,
    np.round(np.clip(
        np.random.poisson(1.3, N)
        + (operational_exposure_risk_index > 0.65).astype(int),
        1, 8)
    ).astype(int),
    0,
)


# ── Severity — base layer (σ = 0.65, cap $1.5 M) ─────────────────────────────
_weather_sev_adj  = {"low": 0.00, "moderate": 0.05, "high": 0.12, "severe": 0.20}
_cargo_sev_adj    = {
    "general": 0.00, "dry_van": 0.00, "refrigerated": 0.08,
    "livestock": 0.12, "hazmat": 0.30, "oversized": 0.25,
}
weather_sev_adj_arr = np.array([_weather_sev_adj[w] for w in region_weather_risk])
cargo_sev_adj_arr   = np.array([_cargo_sev_adj[c]   for c in cargo_type])

severity_log_mean = (
    8.2
    + 0.45 * heavy_vehicle_flag
    + 0.30 * (speed_x_night_severity_multiplier / 5)
    + 0.25 * np.log1p(bi_limit_usd / 100_000)
    + 0.20 * (fatigue_x_longhaul_score / 5)
    + 0.15 * (behavioral_volatility_index / 5)
    + 0.15 * np.log1p(cargo_value_usd / 100_000)   # cargo value → higher loss
    + 0.12 * cargo_sev_adj_arr                      # hazmat/oversized premium
    + 0.10 * weather_sev_adj_arr                    # weather premium
    - 0.15 * np.log1p(vehicle_age_years + 1)
    + np.random.normal(0, 0.25, N)
)
base_loss = np.where(
    had_claim_flag == 1,
    np.round(np.clip(
        np.random.lognormal(severity_log_mean, 0.65, N),
        500, 1_500_000)
    ).astype(int),
    0,
)


# ── Severity — catastrophic trucking tail  (v3 NEW) ───────────────────────────
# Real commercial trucking portfolios contain ~1–3% of claims that are
# catastrophic: rollovers, jackknifes, hazmat spills, major BI settlements.
#
# Eligibility: must be a claim on a heavy vehicle AND have at least one
#   elevated-risk factor (hazmat/oversized, severe weather, high OERI,
#   long-haul operation, or high BI limit).
#
# Approximately 15% of eligible heavy-vehicle claims are escalated,
# producing ~1.5–2.5% of all heavy-vehicle claims in the cat layer.
#
# Cat losses follow a Pareto distribution (α = 1.2) shifted to $500 K,
# capped at $5 M.  E[loss | cat] ≈ $500 K × α/(α−1) ≈ $3 M, consistent
# with large commercial trucking BI / cargo total-loss settlements.
# Cat loss replaces (not adds to) the base_loss for selected records.

_is_heavy_claim = (had_claim_flag == 1) & (heavy_vehicle_flag == 1)
_elevated_risk  = (
    np.isin(cargo_type, ["hazmat", "oversized"])
    | (region_weather_risk == "severe")
    | (operational_exposure_risk_index > 0.30)
    | (operating_pattern == "long_haul")
    | (bi_limit_usd >= 500_000)
)
_cat_eligible = _is_heavy_claim & _elevated_risk
_cat_selected = np.random.rand(N) < 0.15   # 15% of eligible → ~2% of heavy claims
_cat_mask     = _cat_eligible & _cat_selected

_cat_alpha = 1.2   # Pareto shape; heavier tail than exponential
_cat_loss   = np.round(np.clip(
    500_000 * np.random.pareto(_cat_alpha, N) + 500_000,
    500_000, 5_000_000)
).astype(int)

incurred_loss_12mo_usd = np.where(_cat_mask, _cat_loss, base_loss)

# Boolean flag lets dashboards and pricing models isolate cat claims
# without re-deriving the threshold logic.
catastrophic_loss_flag = _cat_mask.astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ASSEMBLE DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame({
    # ── Identity ──────────────────────────────────────────────────────────────
    "policy_id":                           [f"POL{str(i+1).zfill(6)}" for i in range(N)],
    # ── Fleet / vehicle / driver ──────────────────────────────────────────────
    "state_of_domicile":                   state_of_domicile,
    "operating_pattern":                   operating_pattern,
    "vehicle_type":                        vehicle_type,
    "gvw_lbs":                             gvw_lbs,
    "heavy_vehicle_flag":                  heavy_vehicle_flag,
    "vehicle_age_years":                   vehicle_age_years,
    "adas_equipped_flag":                  adas_equipped_flag,
    "driver_tenure_days":                  driver_tenure_days,
    "driver_age":                          driver_age,
    "mvr_violations_3yr":                  mvr_violations_3yr,
    "prior_at_fault_claims":               prior_at_fault_claims,
    "fleet_safety_program_flag":           fleet_safety_program_flag,
    # ── New underwriting fields (v3) ──────────────────────────────────────────
    "policy_term_months":                  policy_term_months,
    "fleet_size":                          fleet_size,
    "cargo_type":                          cargo_type,
    "cargo_value_usd":                     cargo_value_usd,
    "region_weather_risk":                 region_weather_risk,
    # ── Telematics raw ────────────────────────────────────────────────────────
    "total_miles":                         total_miles,
    "total_trips":                         total_trips,
    "harsh_braking_events":                harsh_braking_events,
    "harsh_acceleration_events":           harsh_acceleration_events,
    "speeding_events":                     speeding_events,
    "phone_distraction_events":            phone_distraction_events.astype(float),
    "tailgating_events":                   tailgating_events.astype(float),
    "night_driving_pct":                   night_driving_pct,
    "max_continuous_driving_hrs":          max_continuous_driving_hrs,
    # ── Derived Tier-2 ────────────────────────────────────────────────────────
    "aggression_index_per100mi":           aggression_index_per100mi,
    "speeding_rate_per100mi":              speeding_rate_per100mi,
    "fatigue_exposure_density":            fatigue_exposure_density,
    "distraction_rate_per_trip":           distraction_rate_per_trip,
    "behavioral_volatility_index":         behavioral_volatility_index,
    # ── Interaction Tier-3 ────────────────────────────────────────────────────
    "speed_x_night_severity_multiplier":   speed_x_night_severity_multiplier,
    "fatigue_x_longhaul_score":            fatigue_x_longhaul_score,
    "new_driver_heavy_vehicle_multiplier": new_driver_heavy_vehicle_multiplier,
    "behavioral_drift_signal":             behavioral_drift_signal.astype(float),
    "operational_exposure_risk_index":     operational_exposure_risk_index,
    # ── Exposure — SINGLE SOURCE OF TRUTH ─────────────────────────────────────
    "vehicle_years":                       vehicle_years,
    "earned_exposure":                     earned_exposure,
    # ── Policy ────────────────────────────────────────────────────────────────
    "bi_limit_usd":                        bi_limit_usd,
    "collision_deductible_usd":            collision_deductible_usd,
    "coverage_type":                       coverage_type,
    # ── Targets ───────────────────────────────────────────────────────────────
    "had_claim_flag":                      had_claim_flag,
    "num_claims_12mo":                     num_claims_12mo,
    "catastrophic_loss_flag":              catastrophic_loss_flag,
    "incurred_loss_12mo_usd":              incurred_loss_12mo_usd,
})


# ══════════════════════════════════════════════════════════════════════════════
# 10. CENTRALIZED UNDERWRITING RISK TIERS
# ══════════════════════════════════════════════════════════════════════════════
def assign_risk_tiers(
    frame: pd.DataFrame,
    oeri_col: str = "operational_exposure_risk_index",
    preferred_cutoff: float | None = None,
    high_risk_cutoff: float | None = None,
) -> tuple[pd.Series, dict]:
    """
    Assign underwriting risk tiers.  SINGLE AUTHORITATIVE DEFINITION.

    Parameters
    ----------
    frame            : DataFrame containing `oeri_col`.
    oeri_col         : OERI column name.
    preferred_cutoff : OERI upper bound for Preferred tier
                       (default: 40th percentile of `frame`).
    high_risk_cutoff : OERI lower bound for High-Risk tier
                       (default: 70th percentile of `frame`).

    Returns
    -------
    tier_series : pd.Categorical with ordered levels
                  ['Preferred' < 'Standard' < 'High-Risk']
    thresholds  : dict  {'preferred_max': float, 'high_risk_min': float}
                  Store and pass back on every filtered/subset call to
                  guarantee globally consistent cut-points.

    Dashboard contract
    ------------------
    Preferred pattern — filter the pre-tiered df; never re-tier a subset:

        # ✅ Correct — globally consistent
        tx_view = df[df['state_of_domicile'] == 'TX']
        # uw_risk_tier is already present on df

        # ✅ Also correct — pass stored thresholds back in
        sub_tiers, _ = assign_risk_tiers(
            sub_df,
            preferred_cutoff=TIER_THRESHOLDS['preferred_max'],
            high_risk_cutoff=TIER_THRESHOLDS['high_risk_min'],
        )

        # ❌ Wrong — silently shifts cut-points
        sub_tiers, _ = assign_risk_tiers(sub_df)
    """
    oeri  = frame[oeri_col]
    p_cut = preferred_cutoff if preferred_cutoff is not None else float(oeri.quantile(0.40))
    h_cut = high_risk_cutoff if high_risk_cutoff is not None else float(oeri.quantile(0.70))
    return (
        pd.cut(oeri, bins=[-np.inf, p_cut, h_cut, np.inf],
               labels=["Preferred", "Standard", "High-Risk"], ordered=True),
        {"preferred_max": p_cut, "high_risk_min": h_cut},
    )


# Derive thresholds ONCE from the full dataset — stored in TIER_THRESHOLDS
df[TIER_COL], TIER_THRESHOLDS = assign_risk_tiers(df)


# ══════════════════════════════════════════════════════════════════════════════
# 11. CANONICAL KPI COLUMNS — EXPOSURE-DENOMINATED
# ══════════════════════════════════════════════════════════════════════════════
# These three columns are the ONLY legitimate KPI definitions in this project.
# Any code that computes frequency or premium with a different denominator
# introduces an inconsistency.  Reference KPI_COLS to get column names
# programmatically and avoid typos.
#
#   claim_frequency       = claims per vehicle-year of earned exposure
#   loss_per_vehicle_year = incurred loss per vehicle-year
#   pure_premium          = incurred loss per unit of earned exposure
#
# Zero-claim rows produce 0 / earned_exposure = 0, which is correct.

df["claim_frequency"]        = df["num_claims_12mo"]        / df["earned_exposure"]
df["loss_per_vehicle_year"]  = df["incurred_loss_12mo_usd"] / df["vehicle_years"]
df["pure_premium"]           = df["incurred_loss_12mo_usd"] / df["earned_exposure"]


# ══════════════════════════════════════════════════════════════════════════════
# 12. NOISE / MISSINGNESS
# ══════════════════════════════════════════════════════════════════════════════
for col in ["phone_distraction_events", "tailgating_events", "behavioral_drift_signal"]:
    df.loc[np.random.rand(N) < 0.03, col] = np.nan

mask = np.random.rand(N) < 0.005
df.loc[mask, "driver_age"] = np.random.choice([999, 0, -1], mask.sum())

spike_idx = np.random.choice(N, 15, replace=False)
df.loc[spike_idx, "speeding_events"] = np.random.randint(8_000, 12_000, 15)


# ══════════════════════════════════════════════════════════════════════════════
# 13. SAVE + DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telematics_raw.csv")
    df.to_csv(out_path, index=False)

    W = 58
    hr = "─" * W

    print(hr)
    print(f"  Shape              : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Overall claim rate : {df['had_claim_flag'].mean():.3%}")
    print(hr, "\n")

    # ── Calibration check by tier ─────────────────────────────────────────
    print("Claim statistics by underwriting tier")
    print(hr)
    print(
        df.groupby(TIER_COL, observed=True).agg(
            n          =("had_claim_flag",         "count"),
            claim_rate =("had_claim_flag",         "mean"),
            avg_claims =("num_claims_12mo",        lambda x: x[x > 0].mean()),
            med_loss   =("incurred_loss_12mo_usd", lambda x: x[x > 0].median()),
            p95_loss   =("incurred_loss_12mo_usd", lambda x: x[x > 0].quantile(0.95)),
            p99_loss   =("incurred_loss_12mo_usd", lambda x: x[x > 0].quantile(0.99)),
        ).to_string()
    )
    print()

    # ── Catastrophic tail audit ────────────────────────────────────────────
    cat          = df[df["catastrophic_loss_flag"] == 1]
    heavy_claims = df[(df["had_claim_flag"] == 1) & (df["heavy_vehicle_flag"] == 1)]
    print("Catastrophic loss layer (trucking tail)")
    print(hr)
    print(f"  Cat claims (n)            : {len(cat):,}")
    print(f"  % of all claims           : {len(cat) / df['had_claim_flag'].sum():.2%}")
    print(f"  % of heavy-vehicle claims : {len(cat) / max(len(heavy_claims), 1):.2%}")
    if len(cat):
        print(f"  Loss — median             : ${cat['incurred_loss_12mo_usd'].median():>13,.0f}")
        print(f"  Loss — mean               : ${cat['incurred_loss_12mo_usd'].mean():>13,.0f}")
        print(f"  Loss — max                : ${cat['incurred_loss_12mo_usd'].max():>13,.0f}")
        print(f"  Top cargo types           : {cat['cargo_type'].value_counts().head(3).to_dict()}")
        print(f"  Weather risk mix          : {cat['region_weather_risk'].value_counts().to_dict()}")
    print()

    # ── Full severity distribution ─────────────────────────────────────────
    losses = df.loc[df["had_claim_flag"] == 1, "incurred_loss_12mo_usd"]
    print("Severity distribution — all claims (including cat)")
    print(hr)
    print(losses.describe(percentiles=[.25, .50, .75, .90, .95, .99]).to_string())
    print()

    # ── New field distributions ────────────────────────────────────────────
    print("New field distributions (v3)")
    print(hr)
    print("  policy_term_months :", df["policy_term_months"].value_counts().sort_index().to_dict())
    print("  cargo_type         :", df["cargo_type"].value_counts().to_dict())
    print("  region_weather_risk:", df["region_weather_risk"].value_counts().to_dict())
    print(f"  fleet_size    med={df['fleet_size'].median():.0f}  "
          f"p90={df['fleet_size'].quantile(.9):.0f}  max={df['fleet_size'].max()}")
    print(f"  cargo_value   med=${df['cargo_value_usd'].median():,.0f}  "
          f"p90=${df['cargo_value_usd'].quantile(.9):,.0f}  "
          f"max=${df['cargo_value_usd'].max():,.0f}")
    print()

    # ── Exposure ──────────────────────────────────────────────────────────
    print(f"Exposure (single source of truth)  —  bounded by policy_term_years")
    print(hr)
    print(df[list(EXPOSURE_COLS) + ["policy_term_months"]].describe().to_string())
    print()

    # ── Tier thresholds ────────────────────────────────────────────────────
    print("Underwriting tier thresholds (OERI)")
    print(hr)
    print(f"  Preferred  : OERI ≤ {TIER_THRESHOLDS['preferred_max']:.4f}")
    print(f"  Standard   : {TIER_THRESHOLDS['preferred_max']:.4f}"
          f" < OERI ≤ {TIER_THRESHOLDS['high_risk_min']:.4f}")
    print(f"  High-Risk  : OERI > {TIER_THRESHOLDS['high_risk_min']:.4f}")
    print()

    miss = df.isnull().sum()
    print("Missing values (injected noise):")
    print(miss[miss > 0].to_string())
    print(f"\n✅  Saved → {out_path}")