"""
Telematics Risk Intelligence — Two-Part Model Training  (v2)
=============================================================
Part 1 — Frequency : Logistic Regression GLM (statsmodels + sklearn)
           with actuarial exposure offset  log(earned_exposure)
Part 2 — Severity  : GA2M/EBM via interpret-learn (falls back to Tweedie GLM)
           with underwriting-level severity cap before pure-premium assembly

Key fixes vs v1
---------------
1. CONFIG MISMATCH FIXED
   Reads `cat_cols_freq` and `cat_cols_sev` from feature_selection.json
   instead of the defunct single `cat_cols` key.  Each preprocessor now
   uses the correct categorical list for its feature set.

2. ACTUARIAL EXPOSURE OFFSET
   Frequency GLM includes log(earned_exposure) as a statsmodels offset so
   that policies with shorter policy terms contribute proportionally less
   risk mass.  The sklearn wrapper applies the same offset via a custom
   LogisticRegressionWithOffset class so CV metrics remain consistent.

3. CALIBRATION / LIFT ANALYSIS
   After fitting, a full calibration and decile-lift table is produced:
   - Observed vs predicted frequency per ventile (5% probability bucket)
   - Decile lift chart (predicted rank vs actual claim rate)
   - Both saved to outputs/calibration_freq.csv and outputs/lift_freq.csv

4. CONFIG-DRIVEN PRICING LOGIC
   Pure-premium tier thresholds and expense loadings are read from
   config/pricing.json (auto-generated with defaults if absent).
   No thresholds are hard-coded in this script.

5. SEVERITY CAP
   Predicted log-losses are exponentiated and then clipped to a
   configurable underwriting severity cap (default $1 500 000) before
   pure-premium assembly.  This prevents a handful of extreme EBM
   extrapolations from producing dashboard-unrealistic premiums.

Run: python src/model_train.py

Saved artefacts
---------------
  models/glm_frequency.pkl
  models/glm_frequency_sm.pkl
  models/ebm_severity.pkl          (if interpret installed)
  models/glm_severity.pkl          (Tweedie fallback)
  models/preprocessor_freq.pkl
  models/preprocessor_sev.pkl
  outputs/model_predictions.csv
  outputs/model_metrics.json
  outputs/calibration_freq.csv
  outputs/lift_freq.csv
  config/pricing.json              (auto-created if missing)
"""

from __future__ import annotations

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = HERE
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR   = os.path.join(ROOT, "outputs")
CFG_DIR   = os.path.join(ROOT, "config")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CFG_DIR,   exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — feature selection + pricing
# ═══════════════════════════════════════════════════════════════════════════════

with open(os.path.join(CFG_DIR, "feature_selection.json")) as fh:
    feat_cfg = json.load(fh)

FREQ_FEATS     = feat_cfg["final_freq_features"]
SEV_FEATS      = feat_cfg["final_sev_features"]
# FIX 1: separate cat lists per model, not a single shared key
CAT_COLS_FREQ  = feat_cfg["cat_cols_freq"]
CAT_COLS_SEV   = feat_cfg["cat_cols_sev"]

# ── Pricing config (auto-generated with defaults if absent) ───────────────────
PRICING_CFG_PATH = os.path.join(CFG_DIR, "pricing.json")
DEFAULT_PRICING = {
    "tier_thresholds": {
        "LOW":       [None, 1800],
        "MODERATE":  [1800, 5500],
        "HIGH":      [5500, 12000],
        "VERY_HIGH": [12000, None],
    },
    "expense_loading_pct": 0.25,          # 25% expense loading on top of pure premium
    "profit_loading_pct":  0.05,          # 5% profit margin
    "severity_cap_usd":    1_500_000,     # UW severity cap applied before PP assembly
    "min_premium_usd":     500,           # floor after all loadings
}

if not os.path.exists(PRICING_CFG_PATH):
    with open(PRICING_CFG_PATH, "w") as fh:
        json.dump(DEFAULT_PRICING, fh, indent=2)
    print(f"ℹ️  Created default pricing config: {PRICING_CFG_PATH}")

with open(PRICING_CFG_PATH) as fh:
    pricing = json.load(fh)

SEV_CAP       = float(pricing["severity_cap_usd"])
EXPENSE_LOAD  = float(pricing["expense_loading_pct"])
PROFIT_LOAD   = float(pricing["profit_loading_pct"])
MIN_PREMIUM   = float(pricing["min_premium_usd"])
TIER_THRESHOLDS = pricing["tier_thresholds"]   # used in pd.cut at the end

print(f"Pricing config: expense={EXPENSE_LOAD:.0%}  profit={PROFIT_LOAD:.0%}  "
      f"sev_cap=${SEV_CAP:,.0f}  min_premium=${MIN_PREMIUM:,.0f}")


# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("telematics_clean.csv")

# Exposure column — must exist (generated by generate_dataset.py v3)
if "earned_exposure" not in df.columns:
    raise KeyError(
        "'earned_exposure' not found in telematics_clean.csv. "
        "Re-run generate_dataset.py and cleaning.py."
    )
exposure_all = df["earned_exposure"].values.astype(float)

X_freq = df[FREQ_FEATS].copy()
y_freq = df["had_claim_flag"].copy()

mask_c = df["had_claim_flag"] == 1
X_sev  = df.loc[mask_c, SEV_FEATS].copy()
y_sev  = df.loc[mask_c, "incurred_loss_12mo_usd"].copy()
y_sev_log = np.log1p(y_sev)

exposure_freq = exposure_all                        # full fleet for frequency
exposure_sev  = exposure_all[mask_c.values]        # claimants only (unused in sev model)

print(f"Frequency  : {X_freq.shape}   claim rate={y_freq.mean():.3%}")
print(f"Severity   : {X_sev.shape}   mean loss=${y_sev.mean():,.0f}")
print(f"Exposure   : min={exposure_freq.min():.3f}  "
      f"mean={exposure_freq.mean():.3f}  max={exposure_freq.max():.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessor factory
# FIX 1 (cont.): cat_cols argument is now model-specific
# ═══════════════════════════════════════════════════════════════════════════════

def make_preprocessor(feat_list: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerics and OHE-encodes the
    categoricals that are actually present in feat_list.
    """
    num_cols     = [c for c in feat_list if c not in cat_cols]
    actual_cats  = [c for c in feat_list if c in cat_cols]
    transformers: list = [("num", StandardScaler(), num_cols)]
    if actual_cats:
        transformers.append(
            ("cat",
             OneHotEncoder(drop="first", sparse_output=False,
                           handle_unknown="ignore"),
             actual_cats)
        )
    return ColumnTransformer(transformers, remainder="drop")


pre_freq = make_preprocessor(FREQ_FEATS, CAT_COLS_FREQ)   # FIX 1
pre_sev  = make_preprocessor(SEV_FEATS,  CAT_COLS_SEV)    # FIX 1

X_freq_t = pre_freq.fit_transform(X_freq)
X_sev_t  = pre_sev.fit_transform(X_sev)

pickle.dump(pre_freq, open(os.path.join(MODEL_DIR, "preprocessor_freq.pkl"), "wb"))
pickle.dump(pre_sev,  open(os.path.join(MODEL_DIR, "preprocessor_sev.pkl"),  "wb"))
print("✅  Preprocessors saved")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Frequency: Logistic GLM with exposure offset
# FIX 2: log(earned_exposure) enters as an offset (coefficient fixed at 1.0)
#         so the model estimates rate per unit exposure, not per policy.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nPART 1  —  FREQUENCY MODEL (Logistic GLM + Exposure Offset)\n{'='*60}")

log_exposure_freq = np.log(np.clip(exposure_freq, 1e-6, None))  # offset vector

# statsmodels GLM — offset is log(earned_exposure)
X_sm = sm.add_constant(X_freq_t, has_constant="add")
glm_sm = sm.GLM(
    y_freq,
    X_sm,
    family=sm.families.Binomial(link=sm.families.links.Logit()),
    offset=log_exposure_freq,
).fit(maxiter=200, disp=0)
print(glm_sm.summary())

# sklearn wrapper — LogisticRegression has no native offset, so we absorb the
# offset into a synthetic intercept shift via sample_weight is not the right
# tool here; instead we carry the offset as an additional known-coefficient
# term by constructing an adjusted response matrix approach:
#   We use predict_proba on (X | offset) by augmenting the design matrix with
#   the offset as a fixed feature whose coefficient is constrained = 1.
#
# Practical implementation: expose glm_sm predictions for CV-like evaluation,
# but fit a standard sklearn LR on the offset-adjusted features for the
# pipeline-compatible scorer (consistent with industry practice for
# GLM-offset in ML pipelines).

from risk_scoring import LogisticRegressionWithOffset  # noqa: E402


glm_sk = LogisticRegressionWithOffset(
    C=1.0, penalty="l2", solver="lbfgs",
    max_iter=500, class_weight="balanced", random_state=42,
)
glm_sk.set_offset(log_exposure_freq)
glm_sk.fit(X_freq_t, y_freq)

# 5-fold stratified CV — offset must travel with each fold
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_cv = cross_val_predict(glm_sk, X_freq_t, y_freq, cv=cv5, method="predict_proba")[:, 1]
p_is = glm_sk.predict_proba(X_freq_t)[:, 1]

# Also compute statsmodels-offset predictions (for actuarial reporting)
p_sm_is = glm_sm.predict(X_sm, offset=log_exposure_freq)

freq_metrics = {
    "AUC_ROC_CV":  round(roc_auc_score(y_freq, p_cv), 4),
    "AUC_ROC_IS":  round(roc_auc_score(y_freq, p_is), 4),
    "PR_AUC_CV":   round(average_precision_score(y_freq, p_cv), 4),
    "Brier_CV":    round(brier_score_loss(y_freq, p_cv), 4),
    "Gini_CV":     round(2 * roc_auc_score(y_freq, p_cv) - 1, 4),
    "LogLik":      round(float(glm_sm.llf), 2),
    "AIC":         round(float(glm_sm.aic), 2),
    "Pseudo_R2":   round(float(glm_sm.pseudo_rsquared(kind="cs")), 6),
}
print("\nFrequency metrics:")
for k, v in freq_metrics.items():
    print(f"  {k:<18}: {v}")

pickle.dump(glm_sk, open(os.path.join(MODEL_DIR, "glm_frequency.pkl"), "wb"))
pickle.dump(glm_sm, open(os.path.join(MODEL_DIR, "glm_frequency_sm.pkl"), "wb"))
print("✅  GLM frequency models saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3 — Calibration + Lift Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nCALIBRATION & LIFT ANALYSIS\n{'='*60}")

cal_df = pd.DataFrame({
    "policy_id":     df["policy_id"].values,
    "had_claim":     y_freq.values,
    "p_pred_cv":     p_cv,
    "p_pred_sm":     p_sm_is,
    "exposure":      exposure_freq,
})

# Ventile buckets (5% width) on CV predictions
cal_df["ventile"] = pd.qcut(cal_df["p_pred_cv"], q=20, labels=False, duplicates="drop") + 1

calibration_tbl = (
    cal_df.groupby("ventile")
    .agg(
        n_policies      = ("policy_id",   "count"),
        pred_freq_mean  = ("p_pred_cv",   "mean"),
        obs_freq_mean   = ("had_claim",   "mean"),
        total_exposure  = ("exposure",    "sum"),
        actual_claims   = ("had_claim",   "sum"),
    )
    .assign(
        expected_claims = lambda d: d["pred_freq_mean"] * d["n_policies"],
        A_E_ratio       = lambda d: (d["actual_claims"] / d["expected_claims"]).round(4),
    )
    .round(6)
)
calibration_tbl.to_csv(os.path.join(OUT_DIR, "calibration_freq.csv"))
print("Calibration table (observed vs predicted by ventile):")
print(calibration_tbl[["n_policies", "pred_freq_mean", "obs_freq_mean", "A_E_ratio"]].to_string())

# Decile lift — rank policies by descending predicted risk, show actual claim rate
cal_df["decile"] = pd.qcut(cal_df["p_pred_cv"], q=10, labels=False, duplicates="drop")
cal_df["decile"] = 10 - cal_df["decile"]  # decile 1 = highest risk

lift_tbl = (
    cal_df.groupby("decile")
    .agg(
        n_policies    = ("policy_id", "count"),
        avg_pred_prob = ("p_pred_cv", "mean"),
        obs_claim_rate = ("had_claim",  "mean"),
    )
    .sort_index()
    .assign(
        overall_rate = y_freq.mean(),
        lift         = lambda d: (d["obs_claim_rate"] / y_freq.mean()).round(4),
    )
    .round(6)
)
lift_tbl.to_csv(os.path.join(OUT_DIR, "lift_freq.csv"))
print("\nDecile lift (decile 1 = highest predicted risk):")
print(lift_tbl[["n_policies", "avg_pred_prob", "obs_claim_rate", "lift"]].to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Severity: GA2M / EBM  (or Tweedie GLM fallback)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nPART 2  —  SEVERITY MODEL (GA2M / EBM)\n{'='*60}")

sev_metrics: dict = {}
sev_type: str | None = None

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    print("  [GA2M] interpret-learn EBM detected — using GA2M")

    ebm = ExplainableBoostingRegressor(
        interactions=10,
        learning_rate=0.01,
        max_bins=256,
        max_rounds=5000,
        early_stopping_rounds=200,
        n_jobs=-1,
        random_state=42,
    )
    ebm.fit(X_sev_t, y_sev_log)

    p_sev_is = ebm.predict(X_sev_t)
    cv5r = KFold(n_splits=5, shuffle=True, random_state=42)
    p_sev_cv = cross_val_predict(ebm, X_sev_t, y_sev_log, cv=cv5r)

    sev_metrics = {
        "Model":       "EBM_GA2M",
        "MAE_log_CV":  round(mean_absolute_error(y_sev_log, p_sev_cv), 4),
        "RMSE_log_CV": round(np.sqrt(mean_squared_error(y_sev_log, p_sev_cv)), 4),
        "R2_log_CV":   round(r2_score(y_sev_log, p_sev_cv), 4),
        "MAE_USD_IS":  round(mean_absolute_error(y_sev, np.expm1(p_sev_is)), 2),
    }

    pickle.dump(ebm, open(os.path.join(MODEL_DIR, "ebm_severity.pkl"), "wb"))
    sev_type = "EBM"
    print("✅  EBM severity model saved")

except ImportError:
    print("  [Fallback] interpret not installed — using Tweedie GLM (power=1.5)")
    from sklearn.linear_model import TweedieRegressor

    tweedie = TweedieRegressor(power=1.5, alpha=0.5, max_iter=1000)
    tweedie.fit(X_sev_t, y_sev_log)

    p_sev_is = tweedie.predict(X_sev_t)
    cv5r = KFold(n_splits=5, shuffle=True, random_state=42)
    p_sev_cv = cross_val_predict(tweedie, X_sev_t, y_sev_log, cv=cv5r)

    sev_metrics = {
        "Model":       "Tweedie_GLM",
        "MAE_log_CV":  round(mean_absolute_error(y_sev_log, p_sev_cv), 4),
        "RMSE_log_CV": round(np.sqrt(mean_squared_error(y_sev_log, p_sev_cv)), 4),
        "R2_log_CV":   round(r2_score(y_sev_log, p_sev_cv), 4),
        "MAE_USD_IS":  round(mean_absolute_error(y_sev, np.expm1(p_sev_is)), 2),
    }

    pickle.dump(tweedie, open(os.path.join(MODEL_DIR, "glm_severity.pkl"), "wb"))
    sev_type = "Tweedie"
    print("✅  Tweedie severity model saved")

print("\nSeverity metrics:")
for k, v in sev_metrics.items():
    print(f"  {k:<18}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED — Pure Premium = P(claim) × E[Loss | claim]
# FIX 4 (config-driven tiers) + FIX 5 (severity cap)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nCOMBINED  —  Pure Premium (config-driven + UW severity cap)\n{'='*60}")

sev_pkl = "ebm_severity.pkl" if sev_type == "EBM" else "glm_severity.pkl"
sev_mdl = pickle.load(open(os.path.join(MODEL_DIR, sev_pkl), "rb"))

X_all_sev_t  = pre_sev.transform(df[SEV_FEATS])
log_pred_all  = sev_mdl.predict(X_all_sev_t)
e_loss_raw    = np.expm1(log_pred_all)

# FIX 5 — UW severity cap: clip extreme predicted losses before PP assembly
#   Prevents a handful of heavy-vehicle / hazmat outliers from producing
#   implausible premiums in the dashboard.  Cap is configurable in pricing.json.
e_loss_capped = np.clip(e_loss_raw, 0, SEV_CAP)
n_capped = int((e_loss_raw > SEV_CAP).sum())
print(f"  Severity cap @ ${SEV_CAP:,.0f}  — {n_capped} predictions clipped "
      f"({n_capped/len(e_loss_raw):.2%})")

p_claim_all  = p_is                                    # from sklearn GLM (offset-aware)
pure_premium = p_claim_all * e_loss_capped

# FIX 4 — config-driven pricing tiers and loadings
def _tier_bins(thresholds: dict) -> tuple[list, list]:
    """Convert pricing.json tier dict → pd.cut bins + labels."""
    order  = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    labels = [k for k in order if k in thresholds]
    bins   = [-np.inf]
    for lbl in labels:
        lo, hi = thresholds[lbl]
        bins.append(hi if hi is not None else np.inf)
    return bins, labels

tier_bins, tier_labels = _tier_bins(TIER_THRESHOLDS)

pred_df = df[["policy_id", "had_claim_flag", "incurred_loss_12mo_usd",
              "earned_exposure"]].copy()
pred_df["p_claim"]              = p_claim_all.round(6)
pred_df["e_loss_raw"]           = e_loss_raw.round(2)
pred_df["e_loss_capped"]        = e_loss_capped.round(2)
pred_df["pure_premium"]         = pure_premium.round(2)
pred_df["log_loss_pred"]        = log_pred_all.round(4)

# Indicated premium = pure premium × (1 + expense + profit), floored at minimum
indicated_premium = pure_premium * (1 + EXPENSE_LOAD + PROFIT_LOAD)
pred_df["indicated_premium"]    = np.maximum(indicated_premium, MIN_PREMIUM).round(2)

pred_df["risk_tier"] = pd.cut(
    pred_df["pure_premium"],
    bins=tier_bins,
    labels=tier_labels,
)

print("\nPure Premium by Risk Tier:")
summary = (
    pred_df.groupby("risk_tier", observed=True)
    .agg(
        Count             = ("policy_id",           "count"),
        Pct               = ("policy_id",           lambda x: len(x) / len(pred_df) * 100),
        AvgPurePremium    = ("pure_premium",         "mean"),
        AvgIndicatedPrem  = ("indicated_premium",    "mean"),
        ClaimRate         = ("had_claim_flag",        "mean"),
        AvgActualLoss     = ("incurred_loss_12mo_usd","mean"),
    )
    .round(2)
)
print(summary.to_string())

# ─── Save artefacts ───────────────────────────────────────────────────────────
pred_df.to_csv(os.path.join(OUT_DIR, "model_predictions.csv"), index=False)

all_metrics = {
    "frequency":      freq_metrics,
    "severity":       sev_metrics,
    "sev_model_type": sev_type,
    "pricing_config": {
        "expense_loading_pct": EXPENSE_LOAD,
        "profit_loading_pct":  PROFIT_LOAD,
        "severity_cap_usd":    SEV_CAP,
        "min_premium_usd":     MIN_PREMIUM,
        "n_predictions_capped": n_capped,
    },
}
with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as fh:
    json.dump(all_metrics, fh, indent=2)

print("\n✅  model_predictions.csv saved")
print("✅  model_metrics.json saved")
print("✅  calibration_freq.csv saved")
print("✅  lift_freq.csv saved")
print(f"\n{'='*60}\nTWO-PART MODEL TRAINING COMPLETE\n{'='*60}")