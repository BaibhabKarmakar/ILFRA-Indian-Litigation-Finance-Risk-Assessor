"""
src/train.py — Trains duration, outcome, and realisation models.
Uses LightGBM if available, falls back to sklearn GradientBoosting.
After training, computes SHAP explainers and global SHAP values for
model interpretability.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    USE_LGB = False
    print("[train] LightGBM unavailable — using sklearn GradientBoosting")

try:
    import shap
    USE_SHAP = True
except ImportError:
    USE_SHAP = False
    print("[train] SHAP unavailable — skipping explainer computation. "
          "Install with: pip install shap")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from src.feature_engineering import get_feature_cols, get_ibc_feature_cols


# ── Hyperparameter loading ────────────────────────────────────────────────────

def _load_best_params(model_name: str) -> dict:
    """
    Loads tuned hyperparameters from best_params.json.
    Falls back to sensible defaults if tune.py hasn't been run yet.
    """
    params_path = MODELS_DIR / "best_params.json"
    defaults = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    if not params_path.exists():
        print(f"[train] best_params.json not found — using defaults for {model_name}. "
              "Run src/tune.py first for optimised parameters.")
        return defaults
    with open(params_path) as f:
        all_params = json.load(f)
    params = all_params.get(model_name, defaults)
    params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
    return params


# ── Model factories ───────────────────────────────────────────────────────────

_SKL_FALLBACK = dict(
    n_estimators=300, learning_rate=0.05,
    max_depth=5, subsample=0.8, random_state=42,
)

def make_reg(model_name: str = "duration", **kw):
    if USE_LGB:
        return lgb.LGBMRegressor(**{**_load_best_params(model_name), **kw})
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(**{**_SKL_FALLBACK, **kw})

def make_cls(**kw):
    if USE_LGB:
        return lgb.LGBMClassifier(**{**_load_best_params("outcome"), **kw})
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(**{**_SKL_FALLBACK, **kw})

def make_quantile(alpha: float, model_name: str = "duration"):
    if USE_LGB:
        return lgb.LGBMRegressor(
            **{**_load_best_params(model_name), "objective": "quantile", "alpha": alpha}
        )
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, **_SKL_FALLBACK)

def fit(m, Xtr, ytr, Xte=None, yte=None):
    if USE_LGB and Xte is not None:
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    else:
        m.fit(Xtr, ytr)
    return m

def get_fi(m, cols):
    if hasattr(m, "feature_importances_"):
        return pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    return pd.Series(dtype=float)


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_and_save_shap(model, X_train: np.ndarray, X_test: np.ndarray,
                           feature_cols: list, model_name: str) -> None:
    """
    Computes SHAP values using TreeExplainer and saves:
      - models/{model_name}_shap_explainer.pkl   — the explainer object for inference
      - models/{model_name}_shap_values.csv      — mean absolute SHAP per feature
                                                   (global feature importance)

    Parameters
    ----------
    model       : trained LightGBM or sklearn tree model
    X_train     : training features (numpy array) — used to build the explainer
    X_test      : held-out test features — used to compute global SHAP summary
    feature_cols: list of feature names matching columns of X_train / X_test
    model_name  : one of "duration", "outcome", "realisation"
    """
    if not USE_SHAP:
        return

    print(f"[shap] Computing SHAP values for {model_name}...")

    try:
        # TreeExplainer is the correct explainer for LightGBM and sklearn
        # tree ensembles — it uses the exact Tree SHAP algorithm (O(TLD²))
        # rather than the slower sampling-based KernelExplainer.
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values on the test set for the global summary.
        # For classifiers, shap_values returns a list [class_0, class_1] —
        # we take index 1 (positive class) for the outcome model.
        raw_shap = explainer.shap_values(X_test)

        if isinstance(raw_shap, list):
            # Binary classifier: take positive class SHAP values
            shap_values = raw_shap[1]
        else:
            shap_values = raw_shap

        # Global importance = mean absolute SHAP value per feature
        # This is more reliable than LightGBM's gain-based importance
        # because it directly measures each feature's contribution to
        # the output in the same units as the prediction.
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = (
            pd.Series(mean_abs_shap, index=feature_cols)
            .sort_values(ascending=False)
            .rename("mean_abs_shap")
        )
        shap_df.to_csv(MODELS_DIR / f"{model_name}_shap_values.csv", header=True)

        # Save explainer for use at inference time in predict.py
        joblib.dump(explainer, MODELS_DIR / f"{model_name}_shap_explainer.pkl")

        print(f"[shap] {model_name} — explainer saved, "
              f"top feature: {shap_df.index[0]} ({shap_df.iloc[0]:.4f})")

    except Exception as e:
        # SHAP failure must never block training
        print(f"[shap] Warning: SHAP computation failed for {model_name}: {e}")


# ── Duration ──────────────────────────────────────────────────────────────────

def train_duration(df: pd.DataFrame) -> dict:
    """
    Trains the litigation duration model suite.
    Includes a core regressor for expected time, and two quantile regressors
    to establish best-case (q10) and worst-case (q90) timelines.
    """
    fc = get_feature_cols()
    X, y = df[fc].fillna(0), df["duration_months"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    m   = fit(make_reg("duration"),        Xtr, ytr, Xte, yte)
    q10 = fit(make_quantile(0.10, "duration"), Xtr, ytr)
    q90 = fit(make_quantile(0.90, "duration"), Xtr, ytr)

    preds = m.predict(Xte)
    mae, r2 = mean_absolute_error(yte, preds), r2_score(yte, preds)
    print(f"[train] Duration — MAE: {mae:.1f} mo | R²: {r2:.3f}")

    joblib.dump(m,   MODELS_DIR / "duration_model.pkl")
    joblib.dump(q10, MODELS_DIR / "duration_q10.pkl")
    joblib.dump(q90, MODELS_DIR / "duration_q90.pkl")

    # Legacy LightGBM feature importance (kept for backward compatibility)
    get_fi(m, fc).to_csv(MODELS_DIR / "duration_feature_importance.csv",
                          header=["importance"])

    # SHAP — trained on Xtr, evaluated on Xte
    compute_and_save_shap(m, Xtr.values, Xte.values, fc, "duration")

    return {"mae_months": round(mae, 2), "r2": round(r2, 3)}


# ── Outcome ───────────────────────────────────────────────────────────────────

def train_outcome(df: pd.DataFrame) -> dict:
    """
    Trains a binary classifier predicting the probability of a favourable
    outcome for the claimant.
    """
    fc = get_feature_cols()
    X, y = df[fc].fillna(0), df["favourable_outcome"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    m = fit(make_cls(), Xtr, ytr, Xte, yte)

    proba = m.predict_proba(Xte)[:, 1]
    auc   = roc_auc_score(yte, proba)
    print(f"[train] Outcome — AUC: {auc:.3f}")
    print(classification_report(yte, m.predict(Xte),
                                 target_names=["Unfavourable", "Favourable"]))

    joblib.dump(m, MODELS_DIR / "outcome_model.pkl")

    get_fi(m, fc).to_csv(MODELS_DIR / "outcome_feature_importance.csv",
                          header=["importance"])

    # SHAP — for classifiers TreeExplainer returns per-class arrays;
    # compute_and_save_shap handles the list→array extraction internally
    compute_and_save_shap(m, Xtr.values, Xte.values, fc, "outcome")

    return {"auc": round(auc, 3)}


# ── Realisation ───────────────────────────────────────────────────────────────

def train_realisation(df: pd.DataFrame) -> dict:
    """
    Trains models estimating the percentage of the claim value likely to be
    recovered. Only applicable for IBC/money recovery cases.
    """
    fc = get_ibc_feature_cols()
    df = df[df["realisation_pct"].notna()].copy()
    X, y = df[fc].fillna(0), df["realisation_pct"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    m   = fit(make_reg("realisation"),        Xtr, ytr, Xte, yte)
    q10 = fit(make_quantile(0.10, "realisation"), Xtr, ytr)
    q90 = fit(make_quantile(0.90, "realisation"), Xtr, ytr)

    preds = m.predict(Xte)
    mae, r2 = mean_absolute_error(yte, preds), r2_score(yte, preds)
    print(f"[train] Realisation — MAE: {mae:.1f}% | R²: {r2:.3f}")

    joblib.dump(m,   MODELS_DIR / "realisation_model.pkl")
    joblib.dump(q10, MODELS_DIR / "realisation_q10.pkl")
    joblib.dump(q90, MODELS_DIR / "realisation_q90.pkl")

    get_fi(m, fc).to_csv(MODELS_DIR / "realisation_feature_importance.csv",
                          header=["importance"])

    compute_and_save_shap(m, Xtr.values, Xte.values, fc, "realisation")

    return {"mae_pct": round(mae, 2), "r2": round(r2, 3)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LitFin Risk Assessor — Model Training")
    print("=" * 60)

    njdg = pd.read_csv(PROCESSED_DIR / "njdg_features.csv")
    ibc  = pd.read_csv(PROCESSED_DIR / "ibc_features.csv")

    results = {}
    results["duration"]    = train_duration(njdg)
    results["outcome"]     = train_outcome(njdg)
    results["realisation"] = train_realisation(ibc)

    print("\n── Summary ──────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:15s}: {v}")

    pd.DataFrame(results).T.to_csv(MODELS_DIR / "training_metrics.csv")
    print(f"\n[train] Models saved to {MODELS_DIR}")

    # Calibrate outcome model probabilities
    from src.calibration import calibrate_outcome_model
    print("\n[train] Calibrating outcome model probabilities...")
    calibrate_outcome_model()


if __name__ == "__main__":
    main()