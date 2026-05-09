"""
src/feature_engineering.py
---------------------------
Transforms raw NJDG / IBBI data into model-ready feature matrices.

Feature groups:
  A. Case characteristics  (type, court, hierarchy, sector)
  B. Financial signals     (claim amount, log-scaled)
  C. Party & counsel       (govt respondent, senior counsel, win rate)
  D. Process signals       (adjournments, interim order)
  E. Temporal features     (filing year, filing quarter, age at assessment)
  F. Aggregate stats       (historical court-level disposal rates)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["case_type", "court", "state", "sector"]


def _encode_categoricals(df: pd.DataFrame, fit: bool = True,
                          encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical columns using LabelEncoder.
    If fit is True, fits new encoders on the data.
    If fit is False, uses existing encoders to transform data, handling unseen categories gracefully.
    """
    if encoders is None:
        encoders = {}
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "Unknown")
            if "Unknown" not in known:
                le.classes_ = np.append(le.classes_, "Unknown")
            df[col + "_enc"] = le.transform(df[col].astype(str))
    return df, encoders


def _court_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical average duration and win rate per court."""
    agg = (
        df.groupby("court")
        .agg(
            court_avg_duration=("duration_months", "median"),
            court_avg_win_rate=("favourable_outcome", "mean"),
        )
        .reset_index()
    )
    return df.merge(agg, on="court", how="left")


def engineer_njdg_features(df: pd.DataFrame, fit: bool = True,
                            encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Main feature engineering pipeline for National Judicial Data Grid (NJDG) data.
    Generates temporal, financial, and process signal features, and encodes categoricals.
    """
    df = df.copy()

    # Temporal
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["filing_year"] = df["filing_date"].dt.year
    df["filing_quarter"] = df["filing_date"].dt.quarter
    today = pd.Timestamp.today()
    df["case_age_months"] = ((today - df["filing_date"]).dt.days / 30).clip(0)

    # Financial
    df["log_claim_amount"] = np.log1p(df["claim_amount_lakhs"])
    df["claim_bucket"] = pd.cut(
        df["claim_amount_lakhs"],
        bins=[0, 10, 50, 200, 1000, np.inf],
        labels=["<10L", "10-50L", "50-200L", "0.2-1Cr", ">1Cr"],
    ).astype(str)

    # Court aggregates (fitted on training set only)
    df = _court_aggregates(df)

    # Adjournment density
    df["adjournment_density"] = df["num_prior_adjournments"] / df["case_age_months"].clip(1)

    # Encode categoricals
    df, encoders = _encode_categoricals(df, fit=fit, encoders=encoders)

    # Also encode claim_bucket
    if fit:
        le_cb = LabelEncoder()
        df["claim_bucket_enc"] = le_cb.fit_transform(df["claim_bucket"])
        encoders["claim_bucket"] = le_cb
    else:
        le_cb = encoders["claim_bucket"]
        known = set(le_cb.classes_)
        df["claim_bucket"] = df["claim_bucket"].apply(
            lambda x: x if x in known else le_cb.classes_[0]
        )
        df["claim_bucket_enc"] = le_cb.transform(df["claim_bucket"])

    return df, encoders


def get_feature_cols() -> list[str]:
    """
    Returns the canonical list of model input features for NJDG cases.
    Used to ensure the model receives features in the exact expected order.
    """
    return [
        "case_type_enc", "court_enc", "court_hierarchy", "state_enc",
        "sector_enc", "filing_year", "filing_quarter", "case_age_months",
        "log_claim_amount", "claim_bucket_enc",
        "claimant_lawyer_win_rate", "respondent_is_govt", "respondent_is_psu",
        "num_prior_adjournments", "adjournment_density",
        "has_interim_order", "represented_by_senior_counsel",
        "court_avg_duration", "court_avg_win_rate",
    ]


def build_ibc_features(df: pd.DataFrame, fit: bool = True,
                        encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Feature engineering for IBBI/IBC dataset.
    Reads from ibbi_real.csv produced by ibbi_channel.py.

    Key fix: real IBBI data uses cirp_start_date not admission_date.
    """
    if encoders is None:
        encoders = {}
    df = df.copy()

    # cirp_start_date is the real column name from ibbi_channel.py output
    df["cirp_start_date"] = pd.to_datetime(df["cirp_start_date"], errors="coerce")
    df["admission_year"]  = df["cirp_start_date"].dt.year

    df["log_admitted_claim"] = np.log1p(df["admitted_claim_cr"])

    # Creditor/applicant ratio — handle missing gracefully
    if "resolution_applicants_received" in df.columns and "no_of_financial_creditors" in df.columns:
        df["applicants_per_creditor"] = (
            df["resolution_applicants_received"] / df["no_of_financial_creditors"].clip(1)
        )
    else:
        df["applicants_per_creditor"] = 0.0

    # Fill missing process flags with 0 — not all quarters report these
    for flag_col in ["ip_changed", "litigation_pending",
                     "no_of_financial_creditors", "resolution_applicants_received"]:
        if flag_col not in df.columns:
            df[flag_col] = 0
        df[flag_col] = df[flag_col].fillna(0)

    # Encode categoricals
    # bench may not exist in all quarters — default to "Unknown"
    if "bench" not in df.columns:
        df["bench"] = "Unknown"
    df["bench"] = df["bench"].fillna("Unknown")

    ibc_cats = ["sector", "bench", "resolution_status"]
    for col in ibc_cats:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown")
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[f"ibc_{col}"] = le
        else:
            le = encoders[f"ibc_{col}"]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col + "_enc"] = le.transform(df[col].astype(str))

    return df, encoders


def get_ibc_feature_cols() -> list[str]:
    """
    Feature cols for the IBBI realisation model.
    Uses duration_days and favourable_outcome as features — both
    are known for closed cases which is the only subset realisation is trained on.
    """
    return [
        "log_admitted_claim",
        "duration_days",
        "no_of_financial_creditors",
        "resolution_applicants_received",
        "applicants_per_creditor",
        "ip_changed",
        "litigation_pending",
        "sector_enc",
        "bench_enc",
        "admission_year",
    ]


def get_ibc_duration_feature_cols() -> list[str]:
    """
    Feature cols for the IBBI duration model.
    Excludes duration_days (that's the target).
    Excludes favourable_outcome (leakage — outcome is determined at resolution,
    same time as duration).
    """
    return [
        "resolution_status_enc",
        "log_admitted_claim",
        "no_of_financial_creditors",
        "resolution_applicants_received",
        "applicants_per_creditor",
        "ip_changed",
        "litigation_pending",
        "sector_enc",
        "bench_enc",
        "admission_year",
    ]


def get_ibc_outcome_feature_cols() -> list[str]:
    """
    Feature cols for the IBBI outcome classifier.
    Excludes favourable_outcome (that's the target).
    Excludes duration_days — unknown at assessment time for ongoing cases,
    and leaky for closed cases (duration is determined simultaneously with outcome).
    """
    return [
        "resolution_status_enc",
        "log_admitted_claim",
        "no_of_financial_creditors",
        "resolution_applicants_received",
        "applicants_per_creditor",
        "ip_changed",
        "litigation_pending",
        "sector_enc",
        "bench_enc",
        "admission_year",
    ]


# ── Entry point ───────────────────────────────────────────────────────────────
# main() is intentionally removed.
# Feature engineering is now called directly by the pipeline files:
#   src/pipelines/ibbi_pipeline.py  — active, uses ibbi_real.csv
#   src/pipelines/njdg_pipeline.py  — commented out, no real data yet