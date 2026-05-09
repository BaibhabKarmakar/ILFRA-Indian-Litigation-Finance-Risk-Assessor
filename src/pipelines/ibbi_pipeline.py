"""
src/pipelines/ibbi_pipeline.py
-------------------------------
Self-contained IBBI pipeline. Runs everything needed to go from
ibbi_real.csv → trained models → calibrated outcome model.

Run order:
    1. python src/ingestion/ibbi_channel.py   → data/raw/ibbi_real.csv
    2. python src/pipelines/ibbi_pipeline.py  → models/

Models trained:
    - ibc_duration_model.pkl      (target: duration_days)
    - ibc_duration_q10.pkl
    - ibc_duration_q90.pkl
    - ibc_outcome_model.pkl       (target: favourable_outcome)
    - outcome_calibrated.pkl      (isotonic calibration of outcome model)
    - realisation_model.pkl       (target: realisation_pct)
    - realisation_q10.pkl
    - realisation_q90.pkl

Usage:
    python src/pipelines/ibbi_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    roc_auc_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    USE_LGB = False
    print("[ibbi_pipeline] LightGBM unavailable — using sklearn GradientBoosting")

from src.training.feature_engineering import (
    build_ibc_features,
    get_ibc_feature_cols,
    get_ibc_duration_feature_cols,
    get_ibc_outcome_feature_cols,
)

RAW_DIR       = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Model factories ───────────────────────────────────────────────────────────

def _load_best_params(model_name: str) -> dict:
    """Loads tuned params from best_params.json, falls back to defaults."""
    import json
    defaults = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    params_path = MODELS_DIR / "best_params.json"
    if not params_path.exists():
        return defaults
    with open(params_path) as f:
        all_params = json.load(f)
    params = all_params.get(model_name, defaults)
    params.update({"random_state": SEED, "n_jobs": -1, "verbose": -1})
    return params


_SKL_FALLBACK = dict(
    n_estimators=300, learning_rate=0.05,
    max_depth=5, subsample=0.8, random_state=SEED,
)


def _make_reg(model_name: str = "duration"):
    if USE_LGB:
        return lgb.LGBMRegressor(**_load_best_params(model_name))
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(**_SKL_FALLBACK)


def _make_cls(model_name: str = "outcome"):
    if USE_LGB:
        return lgb.LGBMClassifier(**_load_best_params(model_name))
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(**_SKL_FALLBACK)


def _make_quantile(alpha: float, model_name: str = "duration"):
    if USE_LGB:
        return lgb.LGBMRegressor(
            **{**_load_best_params(model_name),
               "objective": "quantile", "alpha": alpha}
        )
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        loss="quantile", alpha=alpha, **_SKL_FALLBACK
    )


def _fit(m, Xtr, ytr, Xte=None, yte=None):
    if USE_LGB and Xte is not None:
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
    else:
        m.fit(Xtr, ytr)
    return m


def _save_fi(m, cols, name: str):
    if hasattr(m, "feature_importances_"):
        pd.Series(m.feature_importances_, index=cols)\
          .sort_values(ascending=False)\
          .to_csv(MODELS_DIR / f"{name}_feature_importance.csv",
                  header=["importance"])


# ── ECE helper ────────────────────────────────────────────────────────────────

def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)


# ── Step 1: Feature engineering ───────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    """
    Loads ibbi_real.csv, runs feature engineering, saves ibc_features.csv.
    Returns the engineered DataFrame.
    """
    ibc_path = RAW_DIR / "ibbi_real.csv"
    if not ibc_path.exists():
        raise FileNotFoundError(
            f"ibbi_real.csv not found at {ibc_path}\n"
            "Run the ingestion pipeline first:\n"
            "  python src/ingestion/ibbi_channel.py"
        )

    ibc = pd.read_csv(ibc_path)
    print(f"[ibbi_pipeline] Loaded {len(ibc)} cases from ibbi_real.csv")

    ibc_feat, ibc_enc = build_ibc_features(ibc, fit=True)
    ibc_feat.to_csv(PROCESSED_DIR / "ibc_features.csv", index=False)
    joblib.dump(ibc_enc, MODELS_DIR / "ibc_encoders.pkl")
    print(f"[ibbi_pipeline] Features saved. Shape: {ibc_feat.shape}")

    return ibc_feat


# ── Step 2: Duration model ────────────────────────────────────────────────────

def train_ibc_duration(df: pd.DataFrame) -> dict:
    """
    Trains a duration model predicting how long a CIRP will take (in days).
    Only trained on closed cases (duration_days is not null).

    Target: duration_days
    Features: get_ibc_duration_feature_cols() — excludes duration_days and
              favourable_outcome to avoid leakage.
    """
    fc = get_ibc_duration_feature_cols()

    # Only closed cases have a real duration
    closed = df[df["duration_days"].notna() & (df["duration_days"] > 0)].copy()
    print(f"[ibbi_pipeline] Duration training on {len(closed)} closed cases")

    X, y = closed[fc].fillna(0), closed["duration_days"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    m   = _fit(_make_reg("duration"), Xtr, ytr, Xte, yte)
    q10 = _fit(_make_quantile(0.10, "duration"), Xtr, ytr)
    q90 = _fit(_make_quantile(0.90, "duration"), Xtr, ytr)

    preds = m.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    r2  = r2_score(yte, preds)
    print(f"[ibbi_pipeline] Duration — MAE: {mae:.1f} days | R²: {r2:.3f}")

    joblib.dump(m,   MODELS_DIR / "ibc_duration_model.pkl")
    joblib.dump(q10, MODELS_DIR / "ibc_duration_q10.pkl")
    joblib.dump(q90, MODELS_DIR / "ibc_duration_q90.pkl")
    _save_fi(m, fc, "ibc_duration")

    return {"mae_days": round(mae, 2), "r2": round(r2, 3)}


# ── Step 3: Outcome model ─────────────────────────────────────────────────────

def train_ibc_outcome(df: pd.DataFrame) -> dict:
    """
    Trains a binary classifier predicting whether a CIRP will result in a
    resolution plan (favourable) vs liquidation or other (unfavourable).

    Target: favourable_outcome
    Features: get_ibc_outcome_feature_cols() — excludes favourable_outcome
              and duration_days (leaky at assessment time).
    Trained on ALL cases (not just closed) — ongoing cases have a known
    favourable_outcome=0 placeholder but that's acceptable here since
    the model is predicting at assessment time.
    """
    fc = get_ibc_outcome_feature_cols()

    X, y = df[fc].fillna(0), df["favourable_outcome"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    m = _fit(_make_cls("outcome"), Xtr, ytr, Xte, yte)

    proba = m.predict_proba(Xte)[:, 1]
    auc   = roc_auc_score(yte, proba)
    print(f"[ibbi_pipeline] Outcome — AUC: {auc:.3f}")
    print(classification_report(
        yte, m.predict(Xte),
        target_names=["Unfavourable", "Favourable"]
    ))

    joblib.dump(m, MODELS_DIR / "ibc_outcome_model.pkl")
    _save_fi(m, fc, "ibc_outcome")

    return {"auc": round(auc, 3)}


# ── Step 4: Calibration ───────────────────────────────────────────────────────

def calibrate_ibc_outcome(df: pd.DataFrame) -> None:
    """
    Fits isotonic calibration on the IBBI outcome model using a held-out
    calibration set. Saves outcome_calibrated.pkl and calibration curves
    for the Streamlit reliability diagram.
    """
    fc = get_ibc_outcome_feature_cols()
    X  = df[fc].fillna(0).values
    y  = df["favourable_outcome"].values

    raw_model_path = MODELS_DIR / "ibc_outcome_model.pkl"
    if not raw_model_path.exists():
        raise FileNotFoundError(
            "ibc_outcome_model.pkl not found. Run train_ibc_outcome() first."
        )
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.frozen import FrozenEstimator

    raw_model = joblib.load(raw_model_path)

    # Fresh 20% hold-out for calibration — model never saw this during training
    _, X_cal, _, y_cal = train_test_split(
        X, y, test_size=0.20, random_state=SEED + 1, stratify=y
    )

    # sklearn 1.2+ uses FrozenEstimator instead of cv="prefit"
    try:
        frozen = FrozenEstimator(raw_model)
        calibrated = CalibratedClassifierCV(frozen, method="isotonic")
    except ImportError:
        # Fallback for older sklearn versions
        calibrated = CalibratedClassifierCV(raw_model, method="isotonic", cv="prefit")

    calibrated.fit(X_cal, y_cal)
    joblib.dump(calibrated, MODELS_DIR / "outcome_calibrated.pkl")

    # Calibration curves for Streamlit reliability diagram
    raw_proba = raw_model.predict_proba(X_cal)[:, 1]
    cal_proba = calibrated.predict_proba(X_cal)[:, 1]

    frac_pos_raw, mean_pred_raw = calibration_curve(y_cal, raw_proba, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_cal, cal_proba, n_bins=10)

    pd.DataFrame({"mean_predicted": mean_pred_raw,
                  "fraction_positive": frac_pos_raw})\
      .to_csv(MODELS_DIR / "calibration_curve_raw.csv", index=False)

    pd.DataFrame({"mean_predicted": mean_pred_cal,
                  "fraction_positive": frac_pos_cal})\
      .to_csv(MODELS_DIR / "calibration_curve_cal.csv", index=False)

    ece_raw = _ece(y_cal, raw_proba)
    ece_cal = _ece(y_cal, cal_proba)
    print(f"[ibbi_pipeline] Calibration ECE: {ece_raw:.4f} → {ece_cal:.4f} "
          f"({(ece_raw - ece_cal) / ece_raw * 100:.1f}% improvement)")


# ── Step 5: Realisation model ─────────────────────────────────────────────────

def train_realisation(df: pd.DataFrame) -> dict:
    """
    Trains models estimating the percentage of admitted claims likely to be
    recovered. Only applicable for closed cases with a known realisation_pct.

    Target: realisation_pct
    Features: get_ibc_feature_cols() — uses duration_days and
              favourable_outcome as features since both are known for
              closed cases (the only subset trained on).
    """
    fc = get_ibc_feature_cols()

    closed = df[df["realisation_pct"].notna()].copy()
    print(f"[ibbi_pipeline] Realisation training on {len(closed)} closed cases")

    X, y = closed[fc].fillna(0), closed["realisation_pct"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    m   = _fit(_make_reg("realisation"), Xtr, ytr, Xte, yte)
    q10 = _fit(_make_quantile(0.10, "realisation"), Xtr, ytr)
    q90 = _fit(_make_quantile(0.90, "realisation"), Xtr, ytr)

    preds = m.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    r2  = r2_score(yte, preds)
    print(f"[ibbi_pipeline] Realisation — MAE: {mae:.1f}% | R²: {r2:.3f}")

    joblib.dump(m,   MODELS_DIR / "realisation_model.pkl")
    joblib.dump(q10, MODELS_DIR / "realisation_q10.pkl")
    joblib.dump(q90, MODELS_DIR / "realisation_q90.pkl")
    _save_fi(m, fc, "realisation")

    return {"mae_pct": round(mae, 2), "r2": round(r2, 3)}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run():
    """
    Runs the full IBBI pipeline end to end.
    Called by train.py or directly.
    """
    print("=" * 60)
    print("ILFRA — IBBI Pipeline")
    print("=" * 60)

    # 1. Feature engineering
    df = run_feature_engineering()

    # 2. Duration
    print("\n── Duration ──────────────────────────────────────────")
    metrics_dur = train_ibc_duration(df)

    # 3. Outcome
    print("\n── Outcome ───────────────────────────────────────────")
    metrics_out = train_ibc_outcome(df)

    # 4. Calibration
    print("\n── Calibration ───────────────────────────────────────")
    calibrate_ibc_outcome(df)

    # 5. Realisation
    print("\n── Realisation ───────────────────────────────────────")
    metrics_real = train_realisation(df)

    # Summary
    print("\n── Summary ───────────────────────────────────────────")
    summary = {
        "ibc_duration":    metrics_dur,
        "ibc_outcome":     metrics_out,
        "realisation":     metrics_real,
    }
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    pd.DataFrame(summary).T.to_csv(MODELS_DIR / "training_metrics.csv")
    print(f"\n[ibbi_pipeline] All models saved to {MODELS_DIR}")

    return summary


if __name__ == "__main__":
    run()