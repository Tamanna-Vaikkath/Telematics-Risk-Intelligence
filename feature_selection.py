"""
Telematics Risk Intelligence — Feature Selection Pipeline
=========================================================
Aligned with: generate_dataset.py v3 + cleaning.py (aligned)

Four-stage selection:
  1. VIF          — drop collinear NUMERIC features only (threshold = 10)
  2. Mutual Information — ranked signal vs each target
  3. RF Hyperparameter Tuning — grid search over (max_depth, min_samples_leaf)
                                scored by cross-validated ROC-AUC to find the
                                best-fitting model before permutation importance
  4. Permutation Importance — on the tuned model, held-out validation split
  5. Composite score + domain-knowledge overrides → final feature sets

Three analytical comparisons are run and reported alongside the main pipeline:

  A. uw_risk_tier ablation
     uw_risk_tier is a monotone bucketing of operational_exposure_risk_index
     (OERI), which is itself a candidate feature.  If OERI is present, the tier
     adds zero independent information in a strict predictive sense.  Two models
     are trained — with and without uw_risk_tier — and their ROC-AUC scores on
     the held-out split are compared.  The result informs whether to include it
     in the final feature set for modeling (vs. keeping it only for reporting).

  B. policy_term_months predictive value test
     policy_term_months governs earned_exposure bounds in the generator and has
     no direct causal path to claim probability or severity.  Any correlation
     is likely a selection artefact (e.g. riskier fleets choosing shorter terms)
     rather than a stable insurance signal.  Its permutation importance on the
     validation split is reported explicitly so the user can decide whether it
     belongs in core risk models or only in exposure/business analysis.

  C. RF sensitivity to hyperparameters
     Small permutation-importance scores can indicate underfitting (model too
     shallow) or over-regularisation (min_samples_leaf too large).  A lightweight
     grid search over (max_depth, min_samples_leaf) with 3-fold stratified CV
     finds the best configuration before the final importance run, avoiding the
     bias of a single arbitrarily fixed depth.

Key design decisions (carried forward)
---------------------------------------
* EXCLUDE covers all target-derived leakage columns (KPIs, exposure cols,
  catastrophic_loss_flag).
* VIF runs on numeric columns only — categoricals are excluded from VIF and
  enter MI/RF via OHE (nominals) or ordinal codes (uw_risk_tier).
* Permutation importance on a 25% held-out split — unbiased toward cardinality.
* class_weight="balanced" on the classifier — corrects for ~10-20% claim rate.
* Leakage guard at the end of Step 5 raises ValueError if any excluded column
  appears in either final feature set.

Run: python src/feature_selection.py
Outputs:
  config/feature_selection.json
  outputs/feature_scores_freq.csv
  outputs/feature_scores_sev.csv
  outputs/vif_scores.csv
  outputs/permutation_importance_freq.csv
  outputs/permutation_importance_sev.csv
  outputs/rf_tuning_freq.csv
  outputs/ablation_report.txt
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ── Fixed seed for reproducibility ───────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = HERE
OUT_DIR = os.path.join(ROOT, "outputs")
CFG_DIR = os.path.join(ROOT, "config")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)


# ─── Column taxonomy ─────────────────────────────────────────────────────────

TARGET_FREQ = "had_claim_flag"
TARGET_SEV  = "incurred_loss_12mo_usd"

# Columns that must NEVER appear as model inputs.
#   (a) raw targets
#   (b) target-derived KPI leakage (computed from targets ÷ exposure)
#   (c) exposure denominators mechanically linked to KPIs
#   (d) generator artefact derived from the claim outcome
EXCLUDE: set[str] = {
    "policy_id",
    # Raw targets
    "had_claim_flag", "num_claims_12mo", "incurred_loss_12mo_usd",
    # Target-derived KPIs
    "claim_frequency", "pure_premium", "loss_per_vehicle_year",
    # Exposure denominators
    "earned_exposure", "vehicle_years",
    # Generator artefact
    "catastrophic_loss_flag",
}

# Nominal categoricals — one-hot encoded for MI and RF steps.
CAT_COLS_NOMINAL: list[str] = [
    "state_of_domicile",
    "operating_pattern",
    "vehicle_type",
    "coverage_type",
    "cargo_type",           # v3: hazmat/oversized severity premium
    "region_weather_risk",  # v3: weather frequency / severity lift
]

# Ordinal categorical — integer codes carry genuine ordering.
# 'Preferred'=0, 'Standard'=1, 'High-Risk'=2
# NOTE: uw_risk_tier is a monotone bucketing of operational_exposure_risk_index.
#       See ablation comparison (Section A) before including in final models.
CAT_COLS_ORDINAL: list[str] = ["uw_risk_tier"]

ALL_CAT_COLS: list[str] = CAT_COLS_NOMINAL + CAT_COLS_ORDINAL

# Domain-knowledge must-keep (never dropped by VIF)
FORCE_KEEP: set[str] = {
    "aggression_index_per100mi",
    "operational_exposure_risk_index",
    "speeding_rate_per100mi",
    "fatigue_exposure_density",
    "mvr_violations_3yr",
    "prior_at_fault_claims",
    "heavy_vehicle_flag",
    "adas_equipped_flag",
    "speed_x_night_severity_multiplier",
    "fatigue_x_longhaul_score",
    "new_driver_heavy_vehicle_multiplier",
    "behavioral_drift_signal",
    "night_driving_pct",
    "bi_limit_usd",
    "fleet_safety_program_flag",
    "driver_tenure_days",
    "operating_pattern",
    "vehicle_type",
    "cargo_type",
    "region_weather_risk",
    "uw_risk_tier",
}

# Hyperparameter grid for frequency RF tuning.
# Kept deliberately small (9 combos × 3-fold CV = 27 fits) to stay fast
# while covering the under/over-regularisation spectrum.
RF_PARAM_GRID: list[dict] = [
    {"max_depth": d, "min_samples_leaf": m}
    for d in [6, 10, 15]
    for m in [5, 20, 50]
]


# ─── Encoding helpers ─────────────────────────────────────────────────────────

def encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a fully numeric feature matrix for MI and RF steps.

    * Ordinal (uw_risk_tier) → .cat.codes  (preserves 0/1/2 ordering)
    * Nominal categoricals   → one-hot (drop_first=False)
    * All other columns passed through unchanged
    """
    df = df.copy()
    for col in CAT_COLS_ORDINAL:
        if col not in df.columns:
            continue
        if hasattr(df[col], "cat"):
            df[col] = df[col].cat.codes
        else:
            order = ["Preferred", "Standard", "High-Risk"]
            df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes
    present_nominals = [c for c in CAT_COLS_NOMINAL if c in df.columns]
    if present_nominals:
        df = pd.get_dummies(df, columns=present_nominals, drop_first=False)
    return df


def _is_force_kept(col_name: str) -> bool:
    if col_name in FORCE_KEEP:
        return True
    for fk in FORCE_KEEP:
        if col_name.startswith(fk + "_"):
            return True
    return False


def ohe_to_raw(feature_set: set[str]) -> set[str]:
    """Map OHE dummy names (e.g. 'cargo_type_hazmat') back to raw column names."""
    raw = set()
    for f in feature_set:
        matched = False
        for cat in CAT_COLS_NOMINAL:
            if f == cat or f.startswith(cat + "_"):
                raw.add(cat)
                matched = True
                break
        if not matched:
            raw.add(f)
    return raw


# ─── VIF (numeric columns only) ──────────────────────────────────────────────

def compute_vif(X_num: pd.DataFrame) -> pd.DataFrame:
    """VIF via matrix OLS. Input must be purely numeric."""
    Xs = StandardScaler().fit_transform(X_num.values.astype(float))
    vif_vals = []
    for i in range(Xs.shape[1]):
        y  = Xs[:, i]
        Xr = np.delete(Xs, i, axis=1)
        beta  = np.linalg.lstsq(Xr, y, rcond=None)[0]
        yhat  = Xr @ beta
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        vif_vals.append(1.0 / (1.0 - r2 + 1e-10))
    return (
        pd.DataFrame({"Feature": X_num.columns, "VIF": vif_vals})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


# ─── RF hyperparameter tuning ─────────────────────────────────────────────────

def tune_rf_classifier(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    param_grid: list[dict],
    n_splits: int = 3,
) -> tuple[dict, pd.DataFrame]:
    """
    Grid search over RF classifier hyperparameters using stratified k-fold
    cross-validation scored by ROC-AUC (appropriate for imbalanced claims).

    Returns
    -------
    best_params : dict with max_depth and min_samples_leaf
    results_df  : full grid with mean / std CV scores, sorted best-first
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for params in param_grid:
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params,
        )
        scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        records.append({
            "max_depth":         params["max_depth"],
            "min_samples_leaf":  params["min_samples_leaf"],
            "roc_auc_mean":      scores.mean(),
            "roc_auc_std":       scores.std(),
        })
    results_df  = pd.DataFrame(records).sort_values("roc_auc_mean", ascending=False)
    best_params = results_df.iloc[0][["max_depth", "min_samples_leaf"]].to_dict()
    best_params = {k: int(v) for k, v in best_params.items()}
    return best_params, results_df


# ─── Utility ─────────────────────────────────────────────────────────────────

def normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-10)


def _build_pi_df(model, X_val, y_val, feature_names) -> pd.DataFrame:
    """Run permutation importance and return a tidy DataFrame."""
    perm = permutation_importance(
        model, X_val, y_val,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    df = pd.DataFrame({
        "Feature": feature_names,
        "PI_mean": np.clip(perm.importances_mean, 0, None),
        "PI_std":  perm.importances_std,
    }).sort_values("PI_mean", ascending=False).reset_index(drop=True)
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_feature_selection(df_clean: pd.DataFrame) -> dict:
    print(f"\n{'='*60}\nFEATURE SELECTION PIPELINE\n{'='*60}")

    # ── Candidate pool ───────────────────────────────────────────────────────
    present_exclude = EXCLUDE & set(df_clean.columns)
    df_feat = df_clean.drop(columns=list(present_exclude))

    print(f"\n  Columns in input   : {df_clean.shape[1]}")
    print(f"  Excluded (leakage) : {sorted(present_exclude)}")
    print(f"  Candidate features : {df_feat.shape[1]}")

    y_freq = df_clean[TARGET_FREQ].astype(int)
    mask_c = y_freq == 1
    y_sev  = np.log1p(df_clean.loc[mask_c, TARGET_SEV])

    print(f"  Freq sample        : {len(y_freq):,} rows  "
          f"(claim rate {y_freq.mean():.1%})")
    print(f"  Sev  sample        : {mask_c.sum():,} rows (claimants only)")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1 — VIF on numeric features only
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Step 1] Computing VIF on numeric features only...")

    numeric_candidates = [
        c for c in df_feat.columns
        if c not in ALL_CAT_COLS and pd.api.types.is_numeric_dtype(df_feat[c])
    ]
    X_num  = df_feat[numeric_candidates].fillna(0)
    vif_df = compute_vif(X_num)

    high_vif = vif_df[vif_df["VIF"] > 10]["Feature"].tolist()
    vif_drop = [f for f in high_vif if not _is_force_kept(f)]

    print(f"  Numeric features   : {len(numeric_candidates)}")
    print(f"  High-VIF (>10)     : {high_vif}")
    print(f"  Actually dropped   : {vif_drop}")
    print(vif_df.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2 — Encode + MI
    # ══════════════════════════════════════════════════════════════════════════
    df_enc = encode_for_model(df_feat)
    df_enc = df_enc.drop(columns=[c for c in vif_drop if c in df_enc.columns],
                         errors="ignore")

    print(f"\n  Post-VIF encoded features : {df_enc.shape[1]}")

    X_enc     = df_enc.fillna(0)
    X_sev_enc = X_enc.loc[mask_c]

    print("\n[Step 2] Mutual Information (frequency)...")
    mi_f    = mutual_info_classif(X_enc, y_freq, random_state=RANDOM_STATE)
    mi_f_df = (
        pd.DataFrame({"Feature": X_enc.columns, "MI": mi_f})
        .sort_values("MI", ascending=False).reset_index(drop=True)
    )
    print(mi_f_df.head(20).to_string(index=False))

    print("\n[Step 2] Mutual Information (severity — log-loss on claimants)...")
    mi_s    = mutual_info_regression(X_sev_enc, y_sev, random_state=RANDOM_STATE)
    mi_s_df = (
        pd.DataFrame({"Feature": X_sev_enc.columns, "MI": mi_s})
        .sort_values("MI", ascending=False).reset_index(drop=True)
    )
    print(mi_s_df.head(20).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3 — RF hyperparameter tuning (frequency model)
    #
    # Rationale: weak permutation-importance scores can reflect underfitting
    # (model too shallow to capture non-linear claim signal) or over-
    # regularisation (min_samples_leaf too large, splitting too conservative).
    # A 3-fold stratified CV grid search over (max_depth, min_samples_leaf)
    # scored by ROC-AUC finds the best configuration before the final
    # permutation importance run, without overfitting to any single split.
    #
    # The severity model uses a simpler fixed config: severity has far fewer
    # training rows (claimants only) so a large grid risks overfitting the
    # tuning itself on a small sample.
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Step 3] Tuning RF classifier hyperparameters "
          f"({len(RF_PARAM_GRID)} combos × 3-fold CV, scored by ROC-AUC)...")

    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_enc, y_freq,
        test_size=0.25, random_state=RANDOM_STATE, stratify=y_freq,
    )

    best_params_f, tuning_results_f = tune_rf_classifier(
        X_tr_f, y_tr_f, RF_PARAM_GRID
    )
    print(f"\n  Tuning results (sorted by ROC-AUC):")
    print(tuning_results_f.to_string(index=False))
    print(f"\n  Best params → max_depth={best_params_f['max_depth']}  "
          f"min_samples_leaf={best_params_f['min_samples_leaf']}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4 — Permutation Importance on tuned models
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Step 4] Permutation importance on tuned models...")

    # ── Frequency classifier ─────────────────────────────────────────────────
    rf_f = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        **best_params_f,
    )
    rf_f.fit(X_tr_f, y_tr_f)

    val_auc_f = roc_auc_score(y_val_f, rf_f.predict_proba(X_val_f)[:, 1])
    print(f"  Frequency model validation ROC-AUC : {val_auc_f:.4f}")

    pi_f_df = _build_pi_df(rf_f, X_val_f, y_val_f, X_enc.columns)
    print("\n  Frequency permutation importance (Top 15):")
    print(pi_f_df.head(15).to_string(index=False))

    # ── Severity regressor ───────────────────────────────────────────────────
    X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
        X_sev_enc, y_sev, test_size=0.25, random_state=RANDOM_STATE,
    )
    rf_s = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,          # fixed: claimant-only sample is smaller;
        min_samples_leaf=10,   # tuning a full grid here risks overfitting it
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_s.fit(X_tr_s, y_tr_s)
    pi_s_df = _build_pi_df(rf_s, X_val_s, y_val_s, X_sev_enc.columns)
    print("\n  Severity permutation importance (Top 15):")
    print(pi_s_df.head(15).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # Comparison A — uw_risk_tier ablation
    #
    # uw_risk_tier = pd.cut(OERI, bins=[p40, p70]) — a monotone discretisation
    # of operational_exposure_risk_index (OERI), which is itself a candidate
    # feature.  If OERI is present, the tier contains zero additional information
    # in a strictly predictive sense; it is a coarser version of the same signal.
    # Including both is redundant and can slightly inflate apparent performance.
    #
    # Two models are trained on the validation-set features:
    #   Model W : with    uw_risk_tier (ordinal-encoded)
    #   Model WO: without uw_risk_tier
    # ROC-AUC on the held-out split determines whether the tier adds anything
    # beyond what OERI alone provides.
    #
    # Recommendation logic:
    #   |AUC_W - AUC_WO| < 0.005 → no meaningful gain; exclude from strict
    #                               predictive models; keep for reporting only.
    #   AUC_W > AUC_WO + 0.005  → tier adds signal (e.g. via bucketing noise);
    #                               safe to include.
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Comparison A] uw_risk_tier ablation...")

    tier_col_enc = "uw_risk_tier"   # ordinal-encoded column name (unchanged by OHE)

    ablation_lines: list[str] = []
    ablation_lines.append("=" * 60)
    ablation_lines.append("COMPARISON A — uw_risk_tier ABLATION")
    ablation_lines.append("=" * 60)
    ablation_lines.append(
        "uw_risk_tier is a monotone bucketing of operational_exposure_risk_index\n"
        "(OERI).  If OERI is present, the tier adds no independent predictive\n"
        "information.  This test quantifies the actual AUC difference."
    )

    if tier_col_enc in X_tr_f.columns:
        # Model WITHOUT uw_risk_tier
        X_tr_wo  = X_tr_f.drop(columns=[tier_col_enc])
        X_val_wo = X_val_f.drop(columns=[tier_col_enc])
        rf_wo = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **best_params_f,
        )
        rf_wo.fit(X_tr_wo, y_tr_f)
        auc_wo = roc_auc_score(y_val_f, rf_wo.predict_proba(X_val_wo)[:, 1])

        # Model WITH uw_risk_tier (already trained as rf_f above)
        auc_w  = val_auc_f
        delta  = auc_w - auc_wo

        ablation_lines.append(f"\n  ROC-AUC WITH    uw_risk_tier : {auc_w:.4f}")
        ablation_lines.append(f"  ROC-AUC WITHOUT uw_risk_tier : {auc_wo:.4f}")
        ablation_lines.append(f"  Delta (W - WO)               : {delta:+.4f}")

        if abs(delta) < 0.005:
            tier_recommendation = "EXCLUDE_FROM_MODELS"
            ablation_lines.append(
                "\n  → Delta < 0.005: uw_risk_tier adds no meaningful predictive gain.\n"
                "    Recommendation: EXCLUDE from strict predictive models.\n"
                "    Keep for underwriting reporting / tier-based pricing only.\n"
                "    OERI alone is the better model input."
            )
        elif delta > 0.005:
            tier_recommendation = "INCLUDE"
            ablation_lines.append(
                f"\n  → Delta = {delta:.4f}: uw_risk_tier adds genuine signal\n"
                "    (likely via bucketing noise in OERI into stable risk bands).\n"
                "    Recommendation: INCLUDE in predictive models."
            )
        else:
            tier_recommendation = "EXCLUDE_FROM_MODELS"
            ablation_lines.append(
                f"\n  → Delta = {delta:.4f}: uw_risk_tier slightly HURTS performance.\n"
                "    Recommendation: EXCLUDE from predictive models."
            )
    else:
        ablation_lines.append(
            "\n  uw_risk_tier not present in encoded feature matrix — skipping ablation."
        )
        tier_recommendation = "NOT_PRESENT"

    ablation_text_a = "\n".join(ablation_lines)
    print(ablation_text_a)

    # ══════════════════════════════════════════════════════════════════════════
    # Comparison B — policy_term_months predictive value test
    #
    # policy_term_months (3 / 6 / 12) governs earned_exposure bounds in the
    # generator and has no direct causal path to claim probability or severity.
    # Any correlation is a selection artefact from the synthetic data-generating
    # process, not a stable generalizable insurance signal.
    #
    # The test measures:
    #   (i)  Permutation importance of policy_term_months on the validation split.
    #        If PI ≈ 0 (within ±1 std of zero), the feature contributes nothing.
    #   (ii) ROC-AUC with vs without policy_term_months.
    #
    # Recommendation logic:
    #   PI_mean < PI_std  → indistinguishable from noise; use for exposure only.
    #   |AUC_delta| < 0.003 → no meaningful AUC contribution; same conclusion.
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Comparison B] policy_term_months predictive value test...")

    ptm_col = "policy_term_months"
    ablation_lines_b: list[str] = []
    ablation_lines_b.append("\n" + "=" * 60)
    ablation_lines_b.append("COMPARISON B — policy_term_months PREDICTIVE VALUE TEST")
    ablation_lines_b.append("=" * 60)
    ablation_lines_b.append(
        "policy_term_months governs earned_exposure bounds (generator design).\n"
        "No direct causal path to claim probability exists.  Any correlation is\n"
        "a selection artefact.  This test checks whether it adds real predictive\n"
        "power or should remain an exposure/business-analysis variable only."
    )

    if ptm_col in X_tr_f.columns:
        # Permutation importance of ptm_col specifically
        ptm_pi_row = pi_f_df[pi_f_df["Feature"] == ptm_col]
        if not ptm_pi_row.empty:
            ptm_pi_mean = ptm_pi_row.iloc[0]["PI_mean"]
            ptm_pi_std  = ptm_pi_row.iloc[0]["PI_std"]
            ptm_pi_rank = ptm_pi_row.index[0] + 1   # 1-indexed rank
        else:
            ptm_pi_mean, ptm_pi_std, ptm_pi_rank = 0.0, 0.0, len(pi_f_df)

        # AUC with vs without policy_term_months
        X_tr_noptm  = X_tr_f.drop(columns=[ptm_col])
        X_val_noptm = X_val_f.drop(columns=[ptm_col])
        rf_noptm = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **best_params_f,
        )
        rf_noptm.fit(X_tr_noptm, y_tr_f)
        auc_noptm  = roc_auc_score(y_val_f, rf_noptm.predict_proba(X_val_noptm)[:, 1])
        auc_with   = val_auc_f
        auc_delta  = auc_with - auc_noptm

        ablation_lines_b.append(f"\n  PI_mean (freq, val split) : {ptm_pi_mean:.5f}  "
                                 f"± {ptm_pi_std:.5f}  (rank {ptm_pi_rank} of {len(pi_f_df)})")
        ablation_lines_b.append(f"  ROC-AUC WITH    ptm      : {auc_with:.4f}")
        ablation_lines_b.append(f"  ROC-AUC WITHOUT ptm      : {auc_noptm:.4f}")
        ablation_lines_b.append(f"  Delta (with - without)   : {auc_delta:+.4f}")

        noise_flag  = ptm_pi_mean < ptm_pi_std
        auc_flag    = abs(auc_delta) < 0.003

        if noise_flag and auc_flag:
            ptm_recommendation = "EXPOSURE_ONLY"
            ablation_lines_b.append(
                "\n  → PI indistinguishable from noise AND AUC delta < 0.003.\n"
                "    Recommendation: EXCLUDE from core risk models.\n"
                "    Use policy_term_months only for exposure calculation and\n"
                "    business/portfolio-mix analysis."
            )
        elif not auc_flag:
            ptm_recommendation = "INCLUDE_WITH_CAUTION"
            ablation_lines_b.append(
                f"\n  → AUC delta = {auc_delta:.4f} (>=0.003): policy_term_months\n"
                "    contributes measurably to AUC.  This may reflect a genuine\n"
                "    selection effect (riskier fleets choosing shorter terms) or\n"
                "    data-generating artefact.  Include cautiously and re-evaluate\n"
                "    on real data before production use."
            )
        else:
            ptm_recommendation = "EXPOSURE_ONLY"
            ablation_lines_b.append(
                "\n  → Mixed signals. Defaulting to EXCLUDE from core risk models.\n"
                "    Use for exposure/business analysis only."
            )
    else:
        ablation_lines_b.append(
            "\n  policy_term_months not present in encoded features — skipping."
        )
        ptm_recommendation = "NOT_PRESENT"

    ablation_text_b = "\n".join(ablation_lines_b)
    print(ablation_text_b)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5 — Composite score & final feature sets
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Step 5] Composite scoring and final feature selection...")

    def composite(mi_df: pd.DataFrame, pi_df: pd.DataFrame) -> pd.DataFrame:
        mi_idx = mi_df.set_index("Feature")["MI"]
        # PI clipped to 0 before normalisation — negative PI means noise
        pi_idx = pi_df.set_index("Feature")["PI_mean"].clip(lower=0)
        score  = pd.DataFrame({
            "MI_norm": normalize(mi_idx),
            "PI_norm": normalize(pi_idx),
        }).fillna(0)
        score["Score"] = score.mean(axis=1)
        return score.sort_values("Score", ascending=False)

    score_f = composite(mi_f_df, pi_f_df)
    score_s = composite(mi_s_df, pi_s_df)

    # TOP_N operates on the post-OHE encoded feature space (which is wider than
    # the raw column count due to one-hot expansion of 6 nominal categoricals).
    TOP_N    = 20
    top_freq = ohe_to_raw(set(score_f.head(TOP_N).index.tolist()))
    top_sev  = ohe_to_raw(set(score_s.head(TOP_N).index.tolist()))

    # Apply ablation recommendations: remove uw_risk_tier and/or
    # policy_term_months from modeling sets if evidence warrants it.
    ablation_exclusions: set[str] = set()
    if tier_recommendation == "EXCLUDE_FROM_MODELS":
        ablation_exclusions.add("uw_risk_tier")
    if ptm_recommendation == "EXPOSURE_ONLY":
        ablation_exclusions.add("policy_term_months")

    effective_force_keep = FORCE_KEEP - ablation_exclusions

    final_freq = sorted(
        (top_freq | effective_force_keep) - set(vif_drop) - ablation_exclusions
    )
    final_sev  = sorted(
        (top_sev  | effective_force_keep) - set(vif_drop) - ablation_exclusions
    )

    # Ensure core categoricals are present
    core_cats = ["operating_pattern", "vehicle_type", "coverage_type",
                 "cargo_type", "region_weather_risk"]
    for c in core_cats:
        if c in df_clean.columns:
            if c not in final_freq: final_freq.append(c)
            if c not in final_sev:  final_sev.append(c)

    # Conditionally add uw_risk_tier based on ablation outcome
    if tier_recommendation == "INCLUDE" and "uw_risk_tier" in df_clean.columns:
        if "uw_risk_tier" not in final_freq: final_freq.append("uw_risk_tier")
        if "uw_risk_tier" not in final_sev:  final_sev.append("uw_risk_tier")

    final_freq = sorted(final_freq)
    final_sev  = sorted(final_sev)

    # Hard leakage guard
    leaked_freq = set(final_freq) & EXCLUDE
    leaked_sev  = set(final_sev)  & EXCLUDE
    if leaked_freq or leaked_sev:
        raise ValueError(
            f"Leakage in final feature sets!\n"
            f"  freq: {leaked_freq}\n  sev: {leaked_sev}"
        )

    print(f"\n  Ablation-driven exclusions : {sorted(ablation_exclusions) or 'none'}")
    print(f"\n  FINAL Frequency features ({len(final_freq)}):")
    for f in final_freq: print(f"    - {f}")
    print(f"\n  FINAL Severity features ({len(final_sev)}):")
    for f in final_sev:  print(f"    - {f}")

    cat_in_freq = [c for c in ALL_CAT_COLS if c in final_freq]
    cat_in_sev  = [c for c in ALL_CAT_COLS if c in final_sev]

    return {
        "vif_df":                vif_df,
        "score_freq":            score_f,
        "score_sev":             score_s,
        "pi_freq":               pi_f_df,
        "pi_sev":                pi_s_df,
        "tuning_results_freq":   tuning_results_f,
        "best_params_freq":      best_params_f,
        "final_freq_features":   final_freq,
        "final_sev_features":    final_sev,
        "cat_cols_freq":         cat_in_freq,
        "cat_cols_sev":          cat_in_sev,
        "vif_dropped":           vif_drop,
        "ablation_exclusions":   sorted(ablation_exclusions),
        "tier_recommendation":   tier_recommendation,
        "ptm_recommendation":    ptm_recommendation,
        "ablation_report":       ablation_text_a + "\n" + ablation_text_b,
    }


# ─── Standalone runner ────────────────────────────────────────────────────────
if __name__ == "__main__":
    clean_path = os.path.join(HERE, "telematics_clean.csv")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {clean_path}\n"
            "Run `python cleaning.py` first."
        )

    df = pd.read_csv(clean_path)

    # Restore ordered Categorical for uw_risk_tier (lost during CSV round-trip)
    if "uw_risk_tier" in df.columns:
        df["uw_risk_tier"] = pd.Categorical(
            df["uw_risk_tier"],
            categories=["Preferred", "Standard", "High-Risk"],
            ordered=True,
        )

    res = run_feature_selection(df)

    # ── Save JSON config ─────────────────────────────────────────────────────
    cfg = {
        "final_freq_features":   res["final_freq_features"],
        "final_sev_features":    res["final_sev_features"],
        "cat_cols_freq":         res["cat_cols_freq"],
        "cat_cols_sev":          res["cat_cols_sev"],
        "cat_cols_nominal":      CAT_COLS_NOMINAL,
        "cat_cols_ordinal":      CAT_COLS_ORDINAL,
        "vif_dropped":           res["vif_dropped"],
        "ablation_exclusions":   res["ablation_exclusions"],
        "tier_recommendation":   res["tier_recommendation"],
        "ptm_recommendation":    res["ptm_recommendation"],
        "best_params_freq":      res["best_params_freq"],
    }
    cfg_path = os.path.join(CFG_DIR, "feature_selection.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    # ── Save CSVs ────────────────────────────────────────────────────────────
    res["vif_df"].to_csv(
        os.path.join(OUT_DIR, "vif_scores.csv"), index=False)
    res["score_freq"].to_csv(
        os.path.join(OUT_DIR, "feature_scores_freq.csv"))
    res["score_sev"].to_csv(
        os.path.join(OUT_DIR, "feature_scores_sev.csv"))
    res["pi_freq"].to_csv(
        os.path.join(OUT_DIR, "permutation_importance_freq.csv"), index=False)
    res["pi_sev"].to_csv(
        os.path.join(OUT_DIR, "permutation_importance_sev.csv"), index=False)
    res["tuning_results_freq"].to_csv(
        os.path.join(OUT_DIR, "rf_tuning_freq.csv"), index=False)

    # ── Save ablation report ─────────────────────────────────────────────────
    rpt_path = os.path.join(OUT_DIR, "ablation_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as fh:
        fh.write(res["ablation_report"])

    print(f"\n[OK]  {cfg_path}")
    print(f"[OK]  {OUT_DIR}/vif_scores.csv")
    print(f"[OK]  {OUT_DIR}/feature_scores_freq.csv")
    print(f"[OK]  {OUT_DIR}/feature_scores_sev.csv")
    print(f"[OK]  {OUT_DIR}/permutation_importance_freq.csv")
    print(f"[OK]  {OUT_DIR}/permutation_importance_sev.csv")
    print(f"[OK]  {OUT_DIR}/rf_tuning_freq.csv")
    print(f"[OK]  {rpt_path}")