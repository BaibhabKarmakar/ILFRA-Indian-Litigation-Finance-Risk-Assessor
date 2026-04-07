"""
src/predict.py
--------------
Loads trained models and encoders, takes a case input dict,
and returns structured risk assessment output.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.cbr_engine import get_engine
from src.feature_engineering import get_feature_cols, get_ibc_feature_cols
from src.cbr_explainer import summarise_precedents, blend_summary

MODELS_DIR = Path(__file__).parent.parent / "models"

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


def load_models():
    """
    Loads into memory all required models (duration, outcome, realisation)
    and categorical encoders from the models directory.
    """
    return {
        "duration": _load("duration_model.pkl"),
        "duration_q10": _load("duration_q10.pkl"),
        "duration_q90": _load("duration_q90.pkl"),
        "outcome": _load("outcome_model.pkl"),
        "realisation": _load("realisation_model.pkl"),
        "realisation_q10": _load("realisation_q10.pkl"),
        "realisation_q90": _load("realisation_q90.pkl"),
        "njdg_encoders": _load("njdg_encoders.pkl"),
        "ibc_encoders": _load("ibc_encoders.pkl"),
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

    Args:
        case: Dictionary with keys matching Streamlit form fields representing raw case properties.
        models: Dictionary of loaded pre-trained models and encoders.

    Returns:
        Structured assessment dictionary containing duration percentiles, outcome probability,
        financial realisation estimates, a composite risk score, and a final recommendation tier.
    """
    enc = models["njdg_encoders"]

    # Build feature row
    case_age = float(case.get("case_age_months", 0))
    adjournments = int(case.get("num_prior_adjournments", 0))
    claim = float(case.get("claim_amount_lakhs", 10))
    court = case.get("court", "District Court")

    row = {
        "case_type_enc": _encode_col(case.get("case_type", "Civil Suit"), enc["case_type"]),
        "court_enc": _encode_col(court, enc["court"]),
        "court_hierarchy": COURT_HIERARCHY.get(court, 1),
        "state_enc": _encode_col(case.get("state", "Delhi"), enc["state"]),
        "sector_enc": _encode_col(case.get("sector", "Others"), enc["sector"]),
        "filing_year": int(case.get("filing_year", 2022)),
        "filing_quarter": int(case.get("filing_quarter", 1)),
        "case_age_months": case_age,
        "log_claim_amount": np.log1p(claim),
        "claim_bucket_enc": 0,  # simplified; full pipeline would compute this
        "claimant_lawyer_win_rate": float(case.get("claimant_lawyer_win_rate", 0.5)),
        "respondent_is_govt": int(case.get("respondent_is_govt", False)),
        "respondent_is_psu": int(case.get("respondent_is_psu", False)),
        "num_prior_adjournments": adjournments,
        "adjournment_density": adjournments / max(case_age, 1),
        "has_interim_order": int(case.get("has_interim_order", False)),
        "represented_by_senior_counsel": int(case.get("represented_by_senior_counsel", False)),
        "court_avg_duration": COURT_AVG_DURATION.get(court, 36),
        "court_avg_win_rate": COURT_AVG_WIN_RATE.get(court, 0.45),
    }

    X = pd.DataFrame([row])

    # Duration predictions
    # Predicts the expected case duration along with 10th and 90th percentile bounds to represent uncertainty
    dur_p50 = float(models["duration"].predict(X)[0])
    dur_p10 = float(models["duration_q10"].predict(X)[0])
    dur_p90 = float(models["duration_q90"].predict(X)[0])

    # Outcome probability
    # Predicts the likelihood (0.0 to 1.0) of a favourable outcome for the claimant
    p_favour = float(models["outcome"].predict_proba(X)[0][1])

    # Realisation (IBC / money recovery only)
    is_ibc = "IBC" in case.get("case_type", "") or "Liquidation" in case.get("case_type", "")
    is_money = case.get("case_type", "") == "Money Recovery"

    realisation = None
    if is_ibc or is_money:
        ibc_enc = models["ibc_encoders"]

        # Map the user-visible case_type to a resolution_status bucket for inference.
        # At assessment time the final status is unknown, so we use a data-informed prior:
        # IBC cases are majority "Ongoing" (unknown outcome); money recovery suits are
        # treated as civil equivalents and default to "Resolution Plan Approved" as a
        # conservative mid-range anchor.  The model will still use p_favour and other
        # signals to refine the estimate.
        _status_map = {
            "CIRP (IBC)": "Ongoing",
            "Liquidation (IBC)": "Liquidation Order",
            "Money Recovery": "Resolution Plan Approved",
        }
        resolution_status = _status_map.get(case.get("case_type", ""), "Ongoing")

        num_creditors = int(case.get("no_of_financial_creditors", 1))
        num_applicants = int(case.get("resolution_applicants_received", 1))

        ibc_row = {
            # Strongest predictors — must be present
            "resolution_status_enc": _encode_col(
                resolution_status, ibc_enc["ibc_resolution_status"]
            ),
            "favourable_outcome": int(p_favour >= 0.5),   # use outcome model's estimate

            # Financial
            "log_admitted_claim": np.log1p(claim / 100),  # convert lakhs → crores approx
            "duration_days": dur_p50 * 30,                # convert predicted months → days

            # Creditor / applicant dynamics
            "no_of_financial_creditors": num_creditors,
            "resolution_applicants_received": num_applicants,
            "applicants_per_creditor": num_applicants / max(num_creditors, 1),

            # Process risk
            "ip_changed": int(case.get("ip_changed", False)),
            "litigation_pending": int(case.get("litigation_pending", False)),

            # Context
            "sector_enc": _encode_col(case.get("sector", "Others"), ibc_enc["ibc_sector"]),
            "bench_enc": 0,   # bench unknown at input time; defaults to first class
            "admission_year": int(case.get("filing_year", 2021)),
        }
        X_ibc = pd.DataFrame([ibc_row])
        try:
            r_p50 = float(models["realisation"].predict(X_ibc)[0])
            r_p10 = float(models["realisation_q10"].predict(X_ibc)[0])
            r_p90 = float(models["realisation_q90"].predict(X_ibc)[0])
            realisation = {
                "p50": round(np.clip(r_p50, 0, 100), 1),
                "p10": round(np.clip(r_p10, 0, 100), 1),
                "p90": round(np.clip(r_p90, 0, 100), 1),
            }
        except Exception:
            pass

    # Risk score (simple composite: 0–100)
    risk_score = round(
        (p_favour * 40)
        + (max(0, 1 - dur_p50 / 120) * 30)
        + ((realisation["p50"] / 100 if realisation else 0.5) * 30),
        1,
    )

    try:
        fc_njdg = get_feature_cols()
        fc_ibc  = get_ibc_feature_cols()

        # Convert DataFrames to numpy arrays for the CBR engine
        X_njdg_arr = X[fc_njdg].fillna(0).values.astype(np.float32)[0]

        # X_ibc only exists if this is an IBC/money recovery case
        X_ibc_arr = X_ibc[fc_ibc].fillna(0).values.astype(np.float32)[0] \
                    if (is_ibc or is_money) else None

        engine = get_engine()

        if is_ibc or is_money:
            similar_cases = engine.retrieve_ibc(X_ibc_arr, fc_ibc, k=5)
        else:
            similar_cases = engine.retrieve_njdg(X_njdg_arr, fc_njdg, k=5)

        adapted = engine.adapt(similar_cases)
        cbr_result = {"similar_cases": similar_cases, "adapted": adapted}

    except Exception as e:
        # CBR failure must never crash the main prediction
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