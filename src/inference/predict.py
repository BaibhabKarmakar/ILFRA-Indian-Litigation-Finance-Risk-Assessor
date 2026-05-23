"""
src/predict.py
--------------
Loads trained models and encoders, takes a case input dict,
and returns structured risk assessment output.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from cbr.cbr_engine import get_engine
from training.feature_engineering import (
    get_feature_cols,
    get_ibc_feature_cols,
    get_ibc_duration_feature_cols,
    get_ibc_outcome_feature_cols,
)
from cbr.cbr_explainer import summarise_precedents, blend_summary

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

def _run_cbr(case: dict, query_njdg: np.ndarray, query_ibc: np.ndarray,
             fc_njdg: list, fc_ibc: list, k: int = 5) -> dict:
    """Runs CBR retrieval and adaptation for a case query."""
    engine = get_engine()
    case_type = case.get("case_type", "")
    is_ibc = "IBC" in case_type or "Insolvency" in case_type

    if is_ibc and query_ibc is not None:
        similar = engine.retrieve_ibc(query_ibc, fc_ibc, k=k)
    else:
        similar = engine.retrieve_njdg(query_njdg, fc_njdg, k=k)

    adapted = engine.adapt(similar)
    return {"similar_cases": similar, "adapted": adapted}

def _load(name: str):
    """Internal helper to load a pickled model or encoder from the models directory."""
    p = MODELS_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Model file '{name}' not found. Run `python src/train.py` first."
        )
    return joblib.load(p)


def _load_optional(name: str):
    """Loads a model file if it exists, returns None otherwise."""
    p = MODELS_DIR / name
    return joblib.load(p) if p.exists() else None


def load_models():
    """
    Loads all available models and encoders.

    IBBI models (duration, outcome, realisation) are required.
    NJDG models are optional — loaded if present, None otherwise.
    predict_case() handles None models gracefully.
    """
    def _require(name: str):
        p = MODELS_DIR / name
        if not p.exists():
            raise FileNotFoundError(
                f"Required model '{name}' not found.\n"
                "Run: python src/train.py"
            )
        return joblib.load(p)

    return {
        # IBBI models — required
        "ibc_duration":     _require("ibc_duration_model.pkl"),
        "ibc_duration_q10": _require("ibc_duration_q10.pkl"),
        "ibc_duration_q90": _require("ibc_duration_q90.pkl"),
        "outcome":          _require("ibc_outcome_model.pkl"),
        "realisation":      _require("realisation_model.pkl"),
        "realisation_q10":  _require("realisation_q10.pkl"),
        "realisation_q90":  _require("realisation_q90.pkl"),
        "ibc_encoders":     _require("ibc_encoders.pkl"),

        # NJDG models — optional, loaded if available
        "duration":         _load_optional("duration_model.pkl"),
        "duration_q10":     _load_optional("duration_q10.pkl"),
        "duration_q90":     _load_optional("duration_q90.pkl"),
        "njdg_encoders":    _load_optional("njdg_encoders.pkl"),

        # Calibrated outcome (optional — used if available)
        "outcome_calibrated": _load_optional("outcome_calibrated.pkl"),
    }


# Court lookup table (must match feature_engineering.py)
COURT_HIERARCHY = {
    "District Court": 1, "Commercial Court": 2,
    "High Court (Original)": 3, "High Court (Appeal)": 3,
    "NCLT": 2, "NCLAT": 3, "Supreme Court": 4,
    "Consumer Forum (District)": 1, "Consumer Forum (State)": 2,
}

COURT_AVG_DURATION = {
    "District Court": 48, "Commercial Court": 24,
    "High Court (Original)": 36, "High Court (Appeal)": 42,
    "NCLT": 30, "NCLAT": 18, "Supreme Court": 60,
    "Consumer Forum (District)": 20, "Consumer Forum (State)": 28,
}

COURT_AVG_WIN_RATE = {
    "District Court": 0.45, "Commercial Court": 0.50,
    "High Court (Original)": 0.48, "High Court (Appeal)": 0.42,
    "NCLT": 0.55, "NCLAT": 0.50, "Supreme Court": 0.40,
    "Consumer Forum (District)": 0.60, "Consumer Forum (State)": 0.55,
}


def _encode_col(value: str, encoder, fallback=None):
    """
    Safely encodes a categorical string feature for inference.
    If the value was not seen during training, returns a fallback category to prevent crashes.
    """
    known = set(encoder.classes_)
    v = value if value in known else (fallback or encoder.classes_[0])
    return int(encoder.transform([v])[0])


def predict_case(case: dict, models: dict) -> dict:
    """
    Runs inference for a given legal case.

    All cases now go through the IBBI model pipeline since that's
    what's trained on real data. NJDG models are used for duration
    only if available (optional).

    Args:
        case: Dictionary with keys matching Streamlit form fields.
        models: Dictionary of loaded pre-trained models and encoders.

    Returns:
        Structured assessment dict with duration, outcome, realisation,
        risk score, recommendation, and CBR precedents.
    """
    ibc_enc = models["ibc_encoders"]
    claim   = float(case.get("claim_amount_lakhs", 10))
    court   = case.get("court", "NCLT")

    is_ibc   = "IBC" in case.get("case_type", "") or "Liquidation" in case.get("case_type", "")
    is_money = case.get("case_type", "") == "Money Recovery"

    num_creditors  = int(case.get("no_of_financial_creditors", 1))
    num_applicants = int(case.get("resolution_applicants_received", 1))

    _status_map = {
        "CIRP (IBC)":        "Ongoing",
        "Liquidation (IBC)": "Liquidation Order",
        "Money Recovery":    "Resolution Plan Approved",
    }
    resolution_status = _status_map.get(case.get("case_type", ""), "Ongoing")

    # ── Outcome prediction (IBC outcome model) ────────────────────────────────
    # Build IBC outcome feature row — excludes duration_days (unknown at assessment time)
    fc_outcome = get_ibc_outcome_feature_cols()
    outcome_row = {
        "resolution_status_enc": _encode_col(
            resolution_status, ibc_enc["ibc_resolution_status"]
        ),
        "log_admitted_claim":              np.log1p(claim / 100),
        "no_of_financial_creditors":       num_creditors,
        "resolution_applicants_received":  num_applicants,
        "applicants_per_creditor":         num_applicants / max(num_creditors, 1),
        "ip_changed":                      int(case.get("ip_changed", False)),
        "litigation_pending":              int(case.get("litigation_pending", False)),
        "sector_enc":                      _encode_col(
            case.get("sector", "Others"), ibc_enc["ibc_sector"]
        ),
        "bench_enc":      0,
        "admission_year": int(case.get("filing_year", 2021)),
    }
    X_outcome = pd.DataFrame([outcome_row])[fc_outcome]

    # Use calibrated model if available
    outcome_model = models.get("outcome_calibrated") or models["outcome"]
    p_favour = float(outcome_model.predict_proba(X_outcome)[0][1])

    # ── Duration prediction (IBC duration model) ──────────────────────────────
    fc_duration = get_ibc_duration_feature_cols()
    duration_row = {
        "resolution_status_enc":           outcome_row["resolution_status_enc"],
        "log_admitted_claim":              outcome_row["log_admitted_claim"],
        "no_of_financial_creditors":       num_creditors,
        "resolution_applicants_received":  num_applicants,
        "applicants_per_creditor":         outcome_row["applicants_per_creditor"],
        "ip_changed":                      outcome_row["ip_changed"],
        "litigation_pending":              outcome_row["litigation_pending"],
        "sector_enc":                      outcome_row["sector_enc"],
        "bench_enc":                       0,
        "admission_year":                  outcome_row["admission_year"],
    }
    X_duration = pd.DataFrame([duration_row])[fc_duration]

    # Duration in days from IBC model → convert to months for display
    dur_days_p50 = float(models["ibc_duration"].predict(X_duration)[0])
    dur_days_p10 = float(models["ibc_duration_q10"].predict(X_duration)[0])
    dur_days_p90 = float(models["ibc_duration_q90"].predict(X_duration)[0])

    dur_p50 = dur_days_p50 / 30
    dur_p10 = dur_days_p10 / 30
    dur_p90 = dur_days_p90 / 30

    # ── Realisation prediction (IBC / money recovery only) ────────────────────
    realisation = None
    if is_ibc or is_money:
        fc_real = get_ibc_feature_cols()
        real_row = {
            **outcome_row,
            "duration_days": dur_days_p50,
            "favourable_outcome": int(p_favour >= 0.5),
            "duration_days":      dur_days_p50,
        }
        X_real = pd.DataFrame([real_row])[fc_real]
        try:
            r_p50 = float(models["realisation"].predict(X_real)[0])
            r_p10 = float(models["realisation_q10"].predict(X_real)[0])
            r_p90 = float(models["realisation_q90"].predict(X_real)[0])
            realisation = {
                "p50": round(np.clip(r_p50, 0, 100), 1),
                "p10": round(np.clip(r_p10, 0, 100), 1),
                "p90": round(np.clip(r_p90, 0, 100), 1),
            }
        except Exception:
            pass

    # ── Risk score ────────────────────────────────────────────────────────────
    risk_score = round(
        (p_favour * 40)
        + (max(0, 1 - dur_p50 / 48) * 30)   # 48 months = IBC statutory outer limit
        + ((realisation["p50"] / 100 if realisation else 0.5) * 30),
        1,
    )

    # ── CBR ───────────────────────────────────────────────────────────────────
    try:
        fc_ibc_full = get_ibc_feature_cols()
        real_row_full = {
            **outcome_row,
            "favourable_outcome": int(p_favour >= 0.5),
            "duration_days":      dur_days_p50,
        }
        X_ibc_arr = pd.DataFrame([real_row_full])[fc_ibc_full]\
                      .fillna(0).values.astype(np.float32)[0]

        engine = get_engine()
        similar_cases = engine.retrieve_ibc(X_ibc_arr, fc_ibc_full, k=5)
        adapted       = engine.adapt(similar_cases)
        cbr_result    = {"similar_cases": similar_cases, "adapted": adapted}

    except Exception as e:
        cbr_result = {"similar_cases": [], "adapted": {}, "error": str(e)}

    return {
        "duration_months":  round(max(1, dur_p50), 1),
        "duration_low":     round(max(1, dur_p10), 1),
        "duration_high":    round(max(1, dur_p90), 1),
        "outcome_prob":     round(p_favour, 3),
        "outcome_label":    "Favourable" if p_favour >= 0.5 else "Unfavourable",
        "realisation_pct":  round(realisation["p50"], 1) if realisation else 0.0,
        "realisation_low":  round(realisation["p10"], 1) if realisation else 0.0,
        "realisation_high": round(realisation["p90"], 1) if realisation else 0.0,
        "risk_score":       risk_score,
        "recommendation":   _recommendation(p_favour, dur_p50, realisation),
        "data_source":      "IBC" if (is_ibc or is_money) else "NJDG",
        "cbr":              cbr_result,
    }


def _recommendation(p_fav: float, dur_months: float, realisation: dict | None) -> str:
    """
    Maps model outputs (probability of favourability and expected duration) 
    to a human-readable risk recommendation tier for litigation funding.
    """
    if p_fav >= 0.65 and dur_months <= 36:
        return "Strong Candidate"
    elif p_fav >= 0.50 and dur_months <= 60:
        return "Moderate Candidate — Further Due Diligence Advised"
    elif p_fav >= 0.40:
        return "Borderline — High Risk, Seek Senior Legal Opinion"
    else:
        return "Weak Candidate — Unfavourable Risk Profile"