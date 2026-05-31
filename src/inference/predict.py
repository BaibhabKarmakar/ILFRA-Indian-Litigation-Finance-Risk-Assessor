"""
Inference module for ILFRA.
Loads models, encoders, runs predictions, and retrieves CBR precedents.
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
    """Loads a pkl file from the models directory."""
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
    Loads all models and encoders. 
    IBBI is required; NJDG models/explainers are optional and fall back to None.
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

        # SHAP explainers — optional, None if not yet generated
        "shap_duration":     _load_optional("ibc_duration_shap_explainer.pkl"),
        "shap_outcome":      _load_optional("ibc_outcome_shap_explainer.pkl"),
        "shap_realisation":  _load_optional("realisation_shap_explainer.pkl"),
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
    """Encodes categorical string features. Falls back to a known class if unseen to avoid crashes."""
    known = set(encoder.classes_)
    v = value if value in known else (fallback or encoder.classes_[0])
    return int(encoder.transform([v])[0])

def _compute_shap(explainer, X_df: pd.DataFrame, is_classifier: bool = False) -> dict | None:
    """Computes SHAP values for a single case, sorted by impact. Returns None on failure."""
    if explainer is None:
        return None
    try:
        sv = explainer(X_df)
        values = sv.values[0]           # single row → shape (features,) or (features, 2)
        if values.ndim == 2:
            values = values[:, 1]       # class-1 slice for binary classifier
        feature_names = X_df.columns.tolist()
        shap_map = dict(zip(feature_names, values.tolist()))
        return dict(sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True))
    except Exception:
        return None
    
def predict_case(case: dict, models: dict) -> dict:
    """
    Main inference entry point. 
    Processes case details, runs LightGBM predictions, computes SHAP, and retrieves CBR precedents.
    """
    ibc_enc = models["ibc_encoders"]
    claim   = float(case.get("claim_amount_lakhs", 10))
    court   = case.get("court", "NCLT")

    is_ibc   = "IBC" in case.get("case_type", "") or "Liquidation" in case.get("case_type", "")
    is_money = case.get("case_type", "") == "Money Recovery"

    _status_map = {
        "CIRP (IBC)":        "Ongoing",
        "Liquidation (IBC)": "Liquidation Order",
        "Money Recovery":    "Resolution Plan Approved",
    }
    resolution_status = _status_map.get(case.get("case_type", ""), "Ongoing")

    # 1. Outcome prediction (LightGBM classifier)
    # Features for the classifier (excludes duration as it is unknown at assessment)
    fc_outcome = get_ibc_outcome_feature_cols()
    outcome_row = {
        "log_admitted_claim":         np.log1p(claim / 100),
        "log_liquidation_value":      0.0,
        "claim_to_liquidation_ratio": 1.0,
        "is_large_case":              int(claim > 50000),
        "admission_year":             int(case.get("filing_year", 2021)),
    }
    X_outcome = pd.DataFrame([outcome_row])[fc_outcome]

    # Predict using calibrated outcome model if available
    outcome_model = models["outcome"]
    p_favour = float(outcome_model.predict_proba(X_outcome)[0][1])

    # 2. Duration prediction
    fc_duration = get_ibc_duration_feature_cols()
    duration_row = {
        "log_admitted_claim":         outcome_row["log_admitted_claim"],
        "log_liquidation_value":      outcome_row["log_liquidation_value"],
        "claim_to_liquidation_ratio": outcome_row["claim_to_liquidation_ratio"],
        "is_large_case":              outcome_row["is_large_case"],
        "admission_year":             outcome_row["admission_year"],
    }
    X_duration = pd.DataFrame([duration_row])[fc_duration]

    # Target is in days, convert to months for display
    dur_days_p50 = float(models["ibc_duration"].predict(X_duration)[0])
    dur_days_p10 = float(models["ibc_duration_q10"].predict(X_duration)[0])
    dur_days_p90 = float(models["ibc_duration_q90"].predict(X_duration)[0])

    dur_p50 = dur_days_p50 / 30
    dur_p10 = dur_days_p10 / 30
    dur_p90 = dur_days_p90 / 30

    # 3. Realisation prediction (only for IBC or Money Recovery)
    realisation = None
    if is_ibc or is_money:
        fc_real = get_ibc_feature_cols()
        real_row = {
            **outcome_row,
            "duration_days":      dur_days_p50,
            "favourable_outcome": int(p_favour >= 0.5),
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

    # 4. Composite Risk Score
    # Combines outcome prob (40%), duration (30%), and realisation (30%)
    risk_score = round(
        (p_favour * 40)
        + (max(0, 1 - dur_p50 / 48) * 30)   # 48 months is the statutory IBC outer limit
        + ((realisation["p50"] / 100 if realisation else 0.5) * 30),
        1,
    )

    # 5. Retrieve CBR precedents
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

    # 6. Compute SHAP explanations
    shap_outcome      = _compute_shap(models.get("shap_outcome"),      X_outcome,  is_classifier=True)
    shap_duration     = _compute_shap(models.get("shap_duration"),     X_duration, is_classifier=False)
    shap_realisation  = None
    if realisation is not None:
        shap_realisation = _compute_shap(models.get("shap_realisation"), X_real, is_classifier=False)

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
        "shap": {
            "outcome":      shap_outcome,
            "duration":     shap_duration,
            "realisation":  shap_realisation,
        },
    }


def _recommendation(p_fav: float, dur_months: float, realisation: dict | None) -> str:
    """Classifies risk profile based on expected duration and win rate."""
    if p_fav >= 0.65 and dur_months <= 36:
        return "Strong Candidate"
    elif p_fav >= 0.50 and dur_months <= 60:
        return "Moderate Candidate — Further Due Diligence Advised"
    elif p_fav >= 0.40:
        return "Borderline — High Risk, Seek Senior Legal Opinion"
    else:
        return "Weak Candidate — Unfavourable Risk Profile"