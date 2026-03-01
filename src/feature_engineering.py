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
    """Canonical list of model input features."""
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
    """Feature engineering for IBBI/IBC dataset."""
    if encoders is None:
        encoders = {}
    df = df.copy()

    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["admission_year"] = df["admission_date"].dt.year
    df["log_admitted_claim"] = np.log1p(df["admitted_claim_cr"])
    df["applicants_per_creditor"] = (
        df["resolution_applicants_received"] / df["no_of_financial_creditors"].clip(1)
    )

    ibc_cats = ["sector", "bench"]
    for col in ibc_cats:
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
    return [
        "sector_enc", "bench_enc", "admission_year",
        "log_admitted_claim", "no_of_financial_creditors",
        "resolution_applicants_received", "applicants_per_creditor",
        "ip_changed", "litigation_pending",
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # NJDG
    njdg = pd.read_csv(RAW_DIR / "njdg_synthetic.csv")
    njdg_feat, enc = engineer_njdg_features(njdg, fit=True)
    njdg_feat.to_csv(PROCESSED_DIR / "njdg_features.csv", index=False)
    joblib.dump(enc, MODELS_DIR / "njdg_encoders.pkl")
    print(f"[feature_engineering] NJDG features saved. Shape: {njdg_feat.shape}")

    # IBBI
    ibc = pd.read_csv(RAW_DIR / "ibbi_synthetic.csv")
    ibc_feat, ibc_enc = build_ibc_features(ibc, fit=True)
    ibc_feat.to_csv(PROCESSED_DIR / "ibc_features.csv", index=False)
    joblib.dump(ibc_enc, MODELS_DIR / "ibc_encoders.pkl")
    print(f"[feature_engineering] IBC features saved. Shape: {ibc_feat.shape}")


if __name__ == "__main__":
    main()