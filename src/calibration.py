"""
src/calibration.py
------------------
Calibrates the outcome classifier's predicted probabilities using
isotonic regression so that "P(favourable) = 0.7" genuinely means
~70% of such cases have a favourable outcome.

Run standalone after train.py:
    python src/calibration.py

Or called automatically from the updated train.py main().

Outputs:
    models/outcome_calibrated.pkl   — CalibratedClassifierCV wrapper
    models/calibration_curve.csv    — for reliability diagram in Streamlit
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
SEED = 42


def calibrate_outcome_model():
    """
    Loads the raw trained outcome model and fits an isotonic calibration
    layer on a held-out calibration set (20% of NJDG data).

    Why a separate calibration set and not the CV folds?
    Because the model already saw those folds during early-stopping.
    Using them for calibration would overfit the calibration layer.
    A fresh 20% hold-out is the cleanest approach.
    """
    from src.feature_engineering import get_feature_cols

    njdg = pd.read_csv(PROCESSED_DIR / "njdg_features.csv")
    fc = get_feature_cols()
    X = njdg[fc].fillna(0).values
    y = njdg["favourable_outcome"].values

    # Split: 60% train (used by train.py), 20% calibration, 20% test
    # We only need the calibration split here — train.py already handled the rest
    _, X_cal, _, y_cal = train_test_split(
        X, y, test_size=0.20, random_state=SEED + 1, stratify=y
    )

    # Load the raw trained model
    raw_model_path = MODELS_DIR / "outcome_model.pkl"
    if not raw_model_path.exists():
        raise FileNotFoundError(
            "outcome_model.pkl not found. Run train.py before calibration.py"
        )
    raw_model = joblib.load(raw_model_path)

    # Fit isotonic calibration on the calibration set
    # cv="prefit" tells sklearn the model is already trained — don't refit it
    calibrated = CalibratedClassifierCV(raw_model, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    # Save calibrated model
    joblib.dump(calibrated, MODELS_DIR / "outcome_calibrated.pkl")
    print(f"[calibration] Calibrated model saved.")

    # Generate calibration curve data for the reliability diagram in Streamlit
    raw_proba      = raw_model.predict_proba(X_cal)[:, 1]
    cal_proba      = calibrated.predict_proba(X_cal)[:, 1]

    frac_pos_raw, mean_pred_raw = calibration_curve(y_cal, raw_proba, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_cal, cal_proba, n_bins=10)

    # Save separately — sklearn drops empty bins independently for each curve
    # so the two arrays are not guaranteed to have the same length
    pd.DataFrame({
        "mean_predicted":    mean_pred_raw,
        "fraction_positive": frac_pos_raw,
    }).to_csv(MODELS_DIR / "calibration_curve_raw.csv", index=False)

    pd.DataFrame({
        "mean_predicted":    mean_pred_cal,
        "fraction_positive": frac_pos_cal,
    }).to_csv(MODELS_DIR / "calibration_curve_cal.csv", index=False)

    # Print summary: how far off was the raw model?
    ece_raw = _expected_calibration_error(y_cal, raw_proba)
    ece_cal = _expected_calibration_error(y_cal, cal_proba)
    print(f"[calibration] ECE before: {ece_raw:.4f}  →  after: {ece_cal:.4f}")
    print(f"[calibration] Improvement: {(ece_raw - ece_cal) / ece_raw * 100:.1f}%")

    return calibrated


def _expected_calibration_error(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) — the weighted average
    gap between predicted probability and actual frequency across bins.
    ECE near 0 means the model is well-calibrated.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()     # actual positive rate in bin
        bin_conf = y_prob[mask].mean()     # average predicted probability in bin
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


if __name__ == "__main__":
    calibrate_outcome_model()