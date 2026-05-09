"""
Telematics Risk Intelligence — Data Cleaning Pipeline
======================================================
Handles: erroneous values, missing imputation, outlier winsorisation,
         derived-feature recomputation, type enforcement, sanity checks,
         exposure-based KPI recomputation, exposure consistency validation,
         underwriting categorical enforcement, and heavy-tail loss preservation.

Aligned with: generate_dataset.py  v3
  • KPI denominators match the generator's SINGLE SOURCE OF TRUTH:
      claim_frequency       = num_claims_12mo       / earned_exposure
      loss_per_vehicle_year = incurred_loss_12mo_usd / vehicle_years
      pure_premium          = incurred_loss_12mo_usd / earned_exposure
  • policy_term_years is DERIVED here from policy_term_months (not a raw col).
  • vehicle_years is validated and kept in sync with earned_exposure.
  • uw_risk_tier preserves the ordered Categorical from assign_risk_tiers().
  • catastrophic_loss_flag is integrity-checked against had_claim_flag.

Public contract (mirrors generate_dataset.py):
    EXPOSURE_COLS = ("vehicle_years", "earned_exposure")
    KPI_COLS      = ("claim_frequency", "loss_per_vehicle_year", "pure_premium")
    TIER_COL      = "uw_risk_tier"

Run standalone:
    python src/cleaning.py

Or import:
    from src.cleaning import clean_telematics
    df_clean = clean_telematics(df_raw)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ── Public contracts — mirror generate_dataset.py so downstream code imports
#    from one place only.
EXPOSURE_COLS: tuple[str, ...] = ("vehicle_years", "earned_exposure")
KPI_COLS:      tuple[str, ...] = ("claim_frequency", "loss_per_vehicle_year", "pure_premium")
TIER_COL:      str             = "uw_risk_tier"

# ── Columns that must NEVER be winsorised ────────────────────────────────────
# Commercial auto exhibits genuine heavy-tail severity. Catastrophic trucking
# losses ($500 K–$5 M Pareto layer) are real events, not measurement errors.
# Clipping them would bias frequency-severity models and dashboard KPIs.
_HEAVY_TAIL_PROTECTED: frozenset = frozenset([
    "incurred_loss_12mo_usd",
    "pure_premium",
    "loss_per_vehicle_year",
])

# ── Annual mileage norms — identical to generate_dataset.py ──────────────────
# Used to re-derive vehicle_years if the column is missing or corrupted.
_ANNUAL_MILE_NORM: dict[str, float] = {
    "pickup":          25_000.0,
    "van":             30_000.0,
    "box_truck":       35_000.0,
    "semi_truck":     100_000.0,
    "straight_truck":  60_000.0,
}


def clean_telematics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns cleaned DataFrame.

    Parameters
    ----------
    df      : raw telematics DataFrame (output of generate_dataset.py v3)
    verbose : print step-by-step log

    Returns
    -------
    df_clean : cleaned DataFrame, zero NaNs, correct dtypes,
               consistent KPIs, validated exposure.
    """
    df = df.copy()
    log: list[str] = []

    def info(msg: str) -> None:
        if verbose:
            print(msg)
        log.append(msg)

    info(f"\n{'='*60}\nCLEANING PIPELINE  |  rows={len(df):,}  cols={df.shape[1]}\n{'='*60}")

    # ── Pre-flight: derive policy_term_years from policy_term_months ────────
    # policy_term_years is NOT a raw column in the dataset; it is computed here
    # and used throughout the pipeline for exposure validation. It is NOT written
    # back to df (to avoid a misleading "new" column).
    if "policy_term_months" in df.columns:
        # Fill any NaN policy_term_months with the mode (12 months = annual policy)
        df["policy_term_months"] = df["policy_term_months"].fillna(12)
        _policy_term_years: pd.Series = df["policy_term_months"] / 12.0
    else:
        # Fallback: assume annual policy for every row
        _policy_term_years = pd.Series(1.0, index=df.index)

    info(f"\n[Pre-flight] policy_term_years derived from policy_term_months")
    info(f"  Values present: {_policy_term_years.value_counts().sort_index().to_dict()}")


    # ══════════════════════════════════════════════════════════════════════════
    # Step 1 — Fix erroneous out-of-range values
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 1] Fixing erroneous values")

    # driver_age — sentinel values 999 / 0 / -1 are data-entry errors
    bad_age = (df["driver_age"] < 21) | (df["driver_age"] > 75)
    n = bad_age.sum()
    median_age = int(df.loc[~bad_age, "driver_age"].median())
    df.loc[bad_age, "driver_age"] = median_age
    info(f"  driver_age              : fixed {n} invalid → median {median_age}")

    # speeding_events — sensor spikes (8 000–12 000) injected by the generator;
    # cap at 99.9th percentile of legitimate values.
    p999 = df["speeding_events"].quantile(0.999)
    n = (df["speeding_events"] > p999).sum()
    df["speeding_events"] = df["speeding_events"].clip(upper=int(p999))
    info(f"  speeding_events         : capped {n} spikes at {p999:.0f}")

    # catastrophic_loss_flag must be 0 when had_claim_flag is 0
    if "catastrophic_loss_flag" in df.columns:
        n_bad_cat = ((df["had_claim_flag"] == 0) & (df["catastrophic_loss_flag"] == 1)).sum()
        df.loc[df["had_claim_flag"] == 0, "catastrophic_loss_flag"] = 0
        info(f"  catastrophic_loss_flag  : zeroed {n_bad_cat} rows where had_claim_flag=0")


    # ══════════════════════════════════════════════════════════════════════════
    # Step 2 — Missing value imputation
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 2] Imputing missing values")

    # Telematics fields with ~3% missingness injected by the generator
    telematics_impute = {
        "phone_distraction_events": "median",
        "tailgating_events":        "median",
        "behavioral_drift_signal":  0.0,   # 0.0 = neutral drift (generator intent)
    }
    for col, strategy in telematics_impute.items():
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        fill = df[col].median() if strategy == "median" else strategy
        df[col] = df[col].fillna(fill)
        info(f"  {col:<35} filled {n:>4} NaN → {fill}")

    # Exposure columns — re-derive vehicle_years from mileage if missing
    if "vehicle_years" in df.columns:
        n_missing_vy = df["vehicle_years"].isna().sum()
        if n_missing_vy and "vehicle_type" in df.columns and "total_miles" in df.columns:
            annual_norm = df["vehicle_type"].map(_ANNUAL_MILE_NORM).fillna(35_000.0)
            derived_vy  = (df["total_miles"] / annual_norm).clip(0.05, _policy_term_years)
            df["vehicle_years"] = df["vehicle_years"].fillna(derived_vy)
            info(f"  {'vehicle_years':<35} filled {n_missing_vy:>4} NaN → re-derived from mileage")
        elif n_missing_vy:
            df["vehicle_years"] = df["vehicle_years"].fillna(_policy_term_years)
            info(f"  {'vehicle_years':<35} filled {n_missing_vy:>4} NaN → policy_term_years fallback")

    if "earned_exposure" in df.columns:
        n_missing_ee = df["earned_exposure"].isna().sum()
        if n_missing_ee:
            # Best fallback: use vehicle_years if now available, else policy term
            if "vehicle_years" in df.columns:
                df["earned_exposure"] = df["earned_exposure"].fillna(df["vehicle_years"])
            else:
                df["earned_exposure"] = df["earned_exposure"].fillna(_policy_term_years)
            info(f"  {'earned_exposure':<35} filled {n_missing_ee:>4} NaN → vehicle_years / policy fallback")


    # ══════════════════════════════════════════════════════════════════════════
    # Step 3 — Winsorise outliers (1st / 99th percentile)
    # NOTE: All loss / severity columns are intentionally excluded.
    #       The catastrophic trucking tail ($500 K–$5 M Pareto) consists of
    #       genuine insured events. Winsorising them would suppress the very
    #       signal that separates Preferred from High-Risk tiers.
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 3] Winsorising outliers (1st–99th pct) — loss columns EXCLUDED")

    winsor_cols = [
        "harsh_braking_events",
        "harsh_acceleration_events",
        "total_miles",
        "total_trips",
        "tailgating_events",
        "phone_distraction_events",
        "fatigue_x_longhaul_score",
        # incurred_loss_12mo_usd  <- intentionally omitted; heavy-tail preserved
        # cargo_value_usd         <- intentionally omitted; genuine lognormal tail
    ]

    # Hard guard — fail loudly rather than silently corrupt losses.
    overlap = _HEAVY_TAIL_PROTECTED & set(winsor_cols)
    if overlap:
        raise ValueError(
            f"Heavy-tail protected columns must not be winsorised: {overlap}"
        )

    for col in winsor_cols:
        if col not in df.columns:
            continue
        lo, hi    = df[col].quantile(0.01), df[col].quantile(0.99)
        clipped   = df[col].clip(lower=lo, upper=hi)
        n_changed = (df[col] != clipped).sum()
        df[col]   = clipped
        info(f"  {col:<35} winsorised {n_changed:>4}  [{lo:.1f} , {hi:.1f}]")

    # Catastrophic loss audit — count and log, never clip.
    if "incurred_loss_12mo_usd" in df.columns:
        cat_threshold = df["incurred_loss_12mo_usd"].quantile(0.999)
        n_cat = (df["incurred_loss_12mo_usd"] > cat_threshold).sum()
        info(
            f"  incurred_loss_12mo_usd  : {n_cat} catastrophic losses "
            f"(> ${cat_threshold:,.0f}) preserved intact — NOT winsorised"
        )


    # ══════════════════════════════════════════════════════════════════════════
    # Step 4 — Recompute distraction_rate_per_trip from cleaned inputs
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 4] Recomputing distraction_rate_per_trip")
    df["distraction_rate_per_trip"] = (
        df["phone_distraction_events"] / (df["total_trips"] + 1e-6)
    ).clip(0, 5).round(4)
    info("  distraction_rate_per_trip recomputed from cleaned inputs")


    # ══════════════════════════════════════════════════════════════════════════
    # Step 5 — Enforce dtypes
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 5] Enforcing dtypes")

    int_cols = [
        # Fleet / vehicle / driver
        "gvw_lbs", "heavy_vehicle_flag", "adas_equipped_flag",
        "driver_tenure_days", "driver_age", "mvr_violations_3yr",
        "prior_at_fault_claims", "fleet_safety_program_flag",
        # New underwriting (v3)
        "policy_term_months", "fleet_size", "cargo_value_usd",
        # Telematics raw
        "total_miles", "total_trips", "harsh_braking_events",
        "harsh_acceleration_events", "speeding_events",
        # Policy
        "bi_limit_usd", "collision_deductible_usd",
        # Targets
        "had_claim_flag", "num_claims_12mo", "catastrophic_loss_flag",
        "incurred_loss_12mo_usd",
    ]
    int_enforced = []
    for col in int_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].round().astype(int)
        int_enforced.append(col)

    # Standard categoricals
    cat_cols = [
        "state_of_domicile", "operating_pattern",
        "vehicle_type", "coverage_type",
        # New underwriting (v3)
        "cargo_type", "region_weather_risk",
    ]
    cat_enforced = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")
        cat_enforced.append(col)

    # uw_risk_tier — MUST preserve the ordered Categorical dtype produced by
    # assign_risk_tiers() in generate_dataset.py.  Casting to plain "category"
    # would drop the ordering and break downstream pd.cut / comparisons.
    if TIER_COL in df.columns:
        col = df[TIER_COL]
        if not (hasattr(col, "cat") and col.cat.ordered):
            df[TIER_COL] = pd.Categorical(
                col,
                categories=["Preferred", "Standard", "High-Risk"],
                ordered=True,
            )
        info(f"  {TIER_COL:<35} ordered Categorical enforced  "
             f"{list(df[TIER_COL].cat.categories)}")

    info("  int  cols enforced: " + ", ".join(int_enforced[:6]) + " ...")
    info("  cat  cols enforced: " + ", ".join(cat_enforced))


    # ══════════════════════════════════════════════════════════════════════════
    # Step 6 — Claim / loss sanity checks
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 6] Claim / loss sanity checks")

    # had_claim_flag = 0  →  num_claims and loss must also be 0
    n1 = ((df["had_claim_flag"] == 0) & (df["num_claims_12mo"] > 0)).sum()
    df.loc[df["had_claim_flag"] == 0, "num_claims_12mo"] = 0
    info(f"  Corrected {n1:>4} rows: claim_flag=0 but num_claims>0")

    n2 = ((df["had_claim_flag"] == 0) & (df["incurred_loss_12mo_usd"] > 0)).sum()
    df.loc[df["had_claim_flag"] == 0, "incurred_loss_12mo_usd"] = 0
    info(f"  Corrected {n2:>4} rows: claim_flag=0 but loss>0")

    # No negative claim counts or losses
    n3 = (df["num_claims_12mo"] < 0).sum()
    df.loc[df["num_claims_12mo"] < 0, "num_claims_12mo"] = 0
    info(f"  Corrected {n3:>4} rows: negative num_claims_12mo → 0")

    n4 = (df["incurred_loss_12mo_usd"] < 0).sum()
    df.loc[df["incurred_loss_12mo_usd"] < 0, "incurred_loss_12mo_usd"] = 0
    info(f"  Corrected {n4:>4} rows: negative incurred_loss_12mo_usd → 0")

    # Verify zero NaNs
    remaining_nan = df.isnull().sum().sum()
    info(f"\n  Remaining NaN count : {remaining_nan}  {'✅' if remaining_nan == 0 else '⚠️'}")


    # ══════════════════════════════════════════════════════════════════════════
    # Step 7 — Exposure consistency validation
    # Both vehicle_years and earned_exposure are always equal in the generator
    # (earned_exposure = vehicle_years.copy()). After any correction here we
    # re-sync them so KPI denominators remain identical to the raw intent.
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 7] Exposure consistency validation")

    for exp_col in ("vehicle_years", "earned_exposure"):
        if exp_col not in df.columns:
            info(f"  ⚠️  {exp_col} absent — skipping its validation")
            continue

        # Rule 1: exposure cannot exceed the policy term
        over = df[exp_col] > _policy_term_years
        n_over = over.sum()
        if n_over:
            df.loc[over, exp_col] = _policy_term_years[over]
            info(f"  ⚠️  {exp_col:<20}: capped {n_over} rows to policy_term_years")
        else:
            info(f"  ✅ {exp_col:<20}: all <= policy_term_years")

        # Rule 2: exposure must be strictly positive (avoid zero-division in KPIs)
        non_pos = (df[exp_col] <= 0).sum()
        if non_pos:
            df.loc[df[exp_col] <= 0, exp_col] = 1e-6
            info(f"  ⚠️  {exp_col:<20}: floored {non_pos} non-positive rows to 1e-6")
        else:
            info(f"  ✅ {exp_col:<20}: all > 0")

    # Re-sync: earned_exposure = vehicle_years (generator invariant)
    if "vehicle_years" in df.columns and "earned_exposure" in df.columns:
        out_of_sync = (df["vehicle_years"] != df["earned_exposure"]).sum()
        if out_of_sync:
            df["earned_exposure"] = df["vehicle_years"].copy()
            info(f"  Resync earned_exposure = vehicle_years for {out_of_sync} rows")
        else:
            info("  ✅ vehicle_years == earned_exposure for all rows")


    # ══════════════════════════════════════════════════════════════════════════
    # Step 8 — Recompute KPI columns from cleaned inputs
    #
    # Denominators MUST match generate_dataset.py exactly:
    #   claim_frequency       = num_claims_12mo       / earned_exposure
    #   loss_per_vehicle_year = incurred_loss_12mo_usd / vehicle_years  <- distinct
    #   pure_premium          = incurred_loss_12mo_usd / earned_exposure
    #
    # vehicle_years and earned_exposure are numerically equal post-sync (Step 7),
    # but we use the correct semantic denominator for each KPI so the column
    # definitions remain aligned with the generator's public contract.
    # ══════════════════════════════════════════════════════════════════════════
    info("\n[Step 8] Recomputing exposure-based KPI columns")

    _has_exposure = "earned_exposure" in df.columns
    _has_vy       = "vehicle_years"   in df.columns
    _has_claims   = "num_claims_12mo" in df.columns
    _has_loss     = "incurred_loss_12mo_usd" in df.columns

    if not (_has_exposure or _has_vy):
        info("  ⚠️  No exposure columns present — KPIs not recomputed")
    else:
        if _has_exposure and _has_claims:
            ee = df["earned_exposure"].clip(lower=1e-6)
            df["claim_frequency"] = (df["num_claims_12mo"] / ee).round(6)
            assert (df["claim_frequency"] >= 0).all(), \
                "claim_frequency has negative values after recompute"
            info("  claim_frequency        <- num_claims_12mo / earned_exposure")

        if _has_vy and _has_loss:
            vy = df["vehicle_years"].clip(lower=1e-6)
            df["loss_per_vehicle_year"] = (df["incurred_loss_12mo_usd"] / vy).round(2)
            assert (df["loss_per_vehicle_year"] >= 0).all(), \
                "loss_per_vehicle_year has negative values after recompute"
            info("  loss_per_vehicle_year  <- incurred_loss_12mo_usd / vehicle_years")

        if _has_exposure and _has_loss:
            ee = df["earned_exposure"].clip(lower=1e-6)
            df["pure_premium"] = (df["incurred_loss_12mo_usd"] / ee).round(2)
            assert (df["pure_premium"] >= 0).all(), \
                "pure_premium has negative values after recompute"
            info("  pure_premium           <- incurred_loss_12mo_usd / earned_exposure")

        # Informational cross-check: implied severity = pure_premium / claim_frequency
        if "claim_frequency" in df.columns and "pure_premium" in df.columns:
            claimant_mask = df["claim_frequency"] > 0
            if claimant_mask.any():
                med_sev = (
                    df.loc[claimant_mask, "pure_premium"]
                    / df.loc[claimant_mask, "claim_frequency"]
                ).median()
                info(f"  Median implied severity (claimants only): ${med_sev:,.0f}")

        # Catastrophic tail cross-check
        if "catastrophic_loss_flag" in df.columns and _has_loss:
            cat_df = df[df["catastrophic_loss_flag"] == 1]
            if len(cat_df):
                info(
                    f"  Cat layer — n={len(cat_df):,}  "
                    f"median=${cat_df['incurred_loss_12mo_usd'].median():,.0f}  "
                    f"max=${cat_df['incurred_loss_12mo_usd'].max():,.0f}"
                )

    info(f"\n{'='*60}\nCLEANING DONE  |  rows={len(df):,}  cols={df.shape[1]}\n{'='*60}\n")
    return df


# ─── Standalone runner ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # generate_dataset.py writes telematics_raw.csv to its own directory,
    # not to a data/ subfolder — match that path convention here.
    HERE     = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(HERE, "telematics_raw.csv")
    OUT_PATH = os.path.join(HERE, "telematics_clean.csv")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}\n"
            "Run `python generate_dataset.py` first."
        )

    df_raw   = pd.read_csv(RAW_PATH)
    df_clean = clean_telematics(df_raw, verbose=True)
    df_clean.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved → {OUT_PATH}")