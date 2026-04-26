"""
src/predict.py — Runs inference for a single case.
Loads trained models, computes ML predictions, CBR retrieval,
and per-case SHAP values for explainability.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# ── Court lookup tables ───────────────────────────────────────────────────────

COURT_HIERARCHY = {
    "Supreme Court": 5, "High Court": 4, "Commercial Court": 3,
    "City Civil Court": 2, "District Court": 1, "Magistrate Court": 1,
}

COURT_AVG_DURATION = {
    "Supreme Court": 84, "High Court": 60, "Commercial Court": 30,
    "City Civil Court": 36, "District Court": 42, "Magistrate Court": 24,
}

COURT_AVG_WIN_RATE = {
    "Supreme Court": 0.55, "High Court": 0.50, "Commercial Court": 0.52,
    "City Civil Court": 0.45, "District Court": 0.45, "Magistrate Court": 0.42,
}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load(filename: str):
    path = MODELS_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None

def load_models() -> dict:
    """
    Loads all trained model artefacts.
    The calibrated outcome model is preferred over the raw one.
    SHAP explainers are loaded if available — their absence never
    blocks inference, SHAP output is simply omitted.
    """
    models = {
        "duration":       _load("duration_model.pkl"),
        "duration_q10":   _load("duration_q10.pkl"),
        "duration_q90":   _load("duration_q90.pkl"),
        # Prefer calibrated outcome model — falls back to raw if absent
        "outcome":        _load("outcome_calibrated.pkl") or _load("outcome_model.pkl"),
        "realisation":    _load("realisation_model.pkl"),
        "realisation_q10": _load("realisation_q10.pkl"),
        "realisation_q90": _load("realisation_q90.pkl"),
        "njdg_encoders":  _load("njdg_encoders.pkl"),
        "ibc_encoders":   _load("ibc_encoders.pkl"),
        # SHAP explainers — optional, loaded if available
        "shap_duration":     _load("duration_shap_explainer.pkl"),
        "shap_outcome":      _load("outcome_shap_explainer.pkl"),
        "shap_realisation":  _load("realisation_shap_explainer.pkl"),
    }
    return models


# ── Encoder helpers ───────────────────────────────────────────────────────────

def _encode_col(value, encoder) -> int:
    if encoder is None:
        return 0
    if hasattr(encoder, "transform"):
        try:
            return int(encoder.transform([value])[0])
        except Exception:
            return 0
    if isinstance(encoder, dict):
        return int(encoder.get(str(value), 0))
    return 0


# ── SHAP inference ────────────────────────────────────────────────────────────

def _compute_shap(explainer, X_row: pd.DataFrame,
                  feature_cols: list, model_type: str) -> dict | None:
    """
    Computes SHAP values for a single input row.

    Parameters
    ----------
    explainer   : loaded shap.TreeExplainer object
    X_row       : single-row DataFrame with model features
    feature_cols: list of feature names
    model_type  : "classifier" or "regressor" — controls array extraction

    Returns
    -------
    dict mapping feature_name → shap_value, sorted by absolute magnitude.
    Returns None if explainer is unavailable or computation fails.
    """
    if explainer is None:
        return None

    try:
        raw = explainer.shap_values(X_row.values)

        # For classifiers, shap_values returns [class_0_array, class_1_array]
        # We take class_1 (favourable outcome) for interpretability
        if isinstance(raw, list):
            shap_vals = raw[1][0]   # class_1, first (only) row
        else:
            shap_vals = raw[0]      # single array, first (only) row

        # Build feature → shap_value mapping
        shap_map = {
            feat: float(val)
            for feat, val in zip(feature_cols, shap_vals)
        }

        # Sort by absolute magnitude — most impactful features first
        shap_map = dict(
            sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        return shap_map

    except Exception as e:
        print(f"[predict] SHAP computation failed ({model_type}): {e}")
        return None


# ── Recommendation tier ───────────────────────────────────────────────────────

def _recommendation(p_fav: float, dur_months: float,
                     realisation: dict | None) -> str:
    if p_fav >= 0.65 and dur_months <= 36:
        return "Strong Candidate"
    elif p_fav >= 0.50 and dur_months <= 60:
        return "Moderate Candidate"
    elif p_fav >= 0.40:
        return "Borderline — High Risk"
    return "Weak Candidate"


# ── Core prediction function ──────────────────────────────────────────────────

def predict_case(case: dict, models: dict) -> dict:
    """
    Runs inference for a given legal case.

    Args:
        case:   Dictionary with keys matching Streamlit form fields.
        models: Dictionary of loaded pre-trained models and encoders.

    Returns:
        Structured assessment dict containing duration percentiles,
        outcome probability, realisation estimates, composite risk score,
        recommendation tier, CBR similar cases, and per-case SHAP values.
    """
    from src.feature_engineering import get_feature_cols, get_ibc_feature_cols

    enc = models["njdg_encoders"]

    # ── Build NJDG feature row ────────────────────────────────────────────────
    case_age     = float(case.get("case_age_months", 0))
    adjournments = int(case.get("num_prior_adjournments", 0))
    claim        = float(case.get("claim_amount_lakhs", 10))
    court        = case.get("court", "District Court")

    row = {
        "case_type_enc":   _encode_col(case.get("case_type", "Civil Suit"), enc["case_type"]),
        "court_enc":       _encode_col(court, enc["court"]),
        "court_hierarchy": COURT_HIERARCHY.get(court, 1),
        "state_enc":       _encode_col(case.get("state", "Delhi"), enc["state"]),
        "sector_enc":      _encode_col(case.get("sector", "Others"), enc["sector"]),
        "filing_year":     int(case.get("filing_year", 2022)),
        "filing_quarter":  int(case.get("filing_quarter", 1)),
        "case_age_months": case_age,
        "log_claim_amount":  np.log1p(claim),
        "claim_bucket_enc":  0,
        "claimant_lawyer_win_rate":      float(case.get("claimant_lawyer_win_rate", 0.5)),
        "respondent_is_govt":            int(case.get("respondent_is_govt", False)),
        "respondent_is_psu":             int(case.get("respondent_is_psu", False)),
        "num_prior_adjournments":        adjournments,
        "adjournment_density":           adjournments / max(case_age, 1),
        "has_interim_order":             int(case.get("has_interim_order", False)),
        "represented_by_senior_counsel": int(case.get("represented_by_senior_counsel", False)),
        "court_avg_duration": COURT_AVG_DURATION.get(court, 36),
        "court_avg_win_rate": COURT_AVG_WIN_RATE.get(court, 0.45),
    }

    fc_njdg = get_feature_cols()
    X = pd.DataFrame([row])[fc_njdg].fillna(0)

    # ── Duration predictions ──────────────────────────────────────────────────
    dur_p50 = float(models["duration"].predict(X)[0])
    dur_p10 = float(models["duration_q10"].predict(X)[0])
    dur_p90 = float(models["duration_q90"].predict(X)[0])

    # ── Outcome probability ───────────────────────────────────────────────────
    p_favour = float(models["outcome"].predict_proba(X)[0][1])

    # ── Realisation (IBC / money recovery only) ───────────────────────────────
    is_ibc   = "IBC" in case.get("case_type", "") or "Liquidation" in case.get("case_type", "")
    is_money = case.get("case_type", "") == "Money Recovery"

    realisation = None
    X_ibc = None
    fc_ibc = get_ibc_feature_cols()

    if is_ibc or is_money:
        ibc_enc = models["ibc_encoders"]

        _status_map = {
            "CIRP (IBC)":        "Ongoing",
            "Liquidation (IBC)": "Liquidation Order",
            "Money Recovery":    "Resolution Plan Approved",
        }
        resolution_status = _status_map.get(case.get("case_type", ""), "Ongoing")

        num_creditors  = int(case.get("no_of_financial_creditors", 1))
        num_applicants = int(case.get("resolution_applicants_received", 1))

        ibc_row = {
            "resolution_status_enc": _encode_col(
                resolution_status, ibc_enc["ibc_resolution_status"]
            ),
            "favourable_outcome":    int(p_favour >= 0.5),
            "log_admitted_claim":    np.log1p(claim / 100),
            "duration_days":         dur_p50 * 30,
            "no_of_financial_creditors":      num_creditors,
            "resolution_applicants_received": num_applicants,
            "applicants_per_creditor":        num_applicants / max(num_creditors, 1),
            "ip_changed":            int(case.get("ip_changed", False)),
            "litigation_pending":    int(case.get("litigation_pending", False)),
            "sector_enc":            _encode_col(case.get("sector", "Others"),
                                                  ibc_enc["ibc_sector"]),
            "bench_enc":             0,
            "admission_year":        int(case.get("filing_year", 2021)),
        }
        X_ibc = pd.DataFrame([ibc_row])[fc_ibc].fillna(0)

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

    # ── Composite risk score ──────────────────────────────────────────────────
    risk_score = round(
        (p_favour * 40)
        + (max(0, 1 - dur_p50 / 120) * 30)
        + ((realisation["p50"] / 100 if realisation else 0.5) * 30),
        1,
    )

    # ── SHAP per-case values ──────────────────────────────────────────────────
    # Each SHAP value tells you how much that feature pushed the prediction
    # up (positive) or down (negative) from the model's base rate.
    # Values are in the model's output units:
    #   duration  → months
    #   outcome   → log-odds contribution (positive = increases P(favourable))
    #   realisation → percentage points
    shap_results = {
        "duration": _compute_shap(
            models.get("shap_duration"), X, fc_njdg, "regressor"
        ),
        "outcome": _compute_shap(
            models.get("shap_outcome"), X, fc_njdg, "classifier"
        ),
        "realisation": _compute_shap(
            models.get("shap_realisation"), X_ibc, fc_ibc, "regressor"
        ) if X_ibc is not None else None,
    }

    # ── CBR similar precedent retrieval ──────────────────────────────────────
    try:
        from src.cbr_engine import get_engine
        engine = get_engine()

        X_njdg_arr = X[fc_njdg].fillna(0).values.astype(np.float32)[0]
        X_ibc_arr  = X_ibc[fc_ibc].fillna(0).values.astype(np.float32)[0] \
                     if X_ibc is not None else None

        if is_ibc or is_money:
            similar_cases = engine.retrieve_ibc(X_ibc_arr, fc_ibc, k=5)
        else:
            similar_cases = engine.retrieve_njdg(X_njdg_arr, fc_njdg, k=5)

        cbr_result = {
            "similar_cases": similar_cases,
            "adapted":       engine.adapt(similar_cases),
        }
    except Exception as e:
        cbr_result = {"similar_cases": [], "adapted": {}, "error": str(e)}

    # ── Assemble final result ─────────────────────────────────────────────────
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
        "shap":             shap_results,
        "cbr":              cbr_result,
    }