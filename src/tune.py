"""
src/tune.py
-----------
Hyperparameter tuning for all three ILFRA models using Optuna + cross-validation.

Run with:
    python src/tune.py

Outputs:
    models/best_params.json   — best hyperparameters per model
    models/cv_results.csv     — per-trial scores for diagnostics
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial noise

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 40      # Optuna trials per model — increase to 80 for real data
N_FOLDS  = 5       # CV folds
SEED     = 42

from src.feature_engineering import get_feature_cols, get_ibc_feature_cols


# ── Search spaces ─────────────────────────────────────────────────────────────

def _lgb_space(trial: optuna.Trial) -> dict:
    """
    Shared LightGBM search space used for all three models.
    Covers the parameters that most affect bias/variance tradeoff on small datasets.
    """
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


# ── Objective functions ───────────────────────────────────────────────────────

def _objective_outcome(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Optuna objective for the binary outcome classifier.
    Metric: mean AUC across 5 stratified folds (higher = better).
    Stratified folds preserve the class ratio in each fold.
    """
    import lightgbm as lgb
    params = _lgb_space(trial)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(0)])
        proba = m.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, proba))
    return float(np.mean(aucs))


def _objective_duration(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Optuna objective for the duration regressor.
    Metric: negative mean MAE across 5 folds (Optuna maximises, so negate MAE).
    """
    import lightgbm as lgb
    params = _lgb_space(trial)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    maes = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(0)])
        maes.append(mean_absolute_error(y_val, m.predict(X_val)))
    return -float(np.mean(maes))   # negative because Optuna maximises


def _objective_realisation(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Optuna objective for the realisation regressor.
    Same as duration: negative MAE, 5-fold CV.
    """
    import lightgbm as lgb
    params = _lgb_space(trial)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    maes = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(0)])
        maes.append(mean_absolute_error(y_val, m.predict(X_val)))
    return -float(np.mean(maes))


# ── Tuning runner ─────────────────────────────────────────────────────────────

def tune_model(name: str, objective_fn, X: np.ndarray, y: np.ndarray,
               n_trials: int = N_TRIALS) -> dict:
    """
    Runs an Optuna study for one model, prints progress, returns best params.
    """
    print(f"\n[tune] {name} — running {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best = study.best_params
    score = study.best_value
    print(f"[tune] {name} best score: {score:.4f}")
    print(f"[tune] {name} best params: {best}")
    return best, study


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ILFRA — Hyperparameter Tuning")
    print("=" * 60)

    njdg = pd.read_csv(PROCESSED_DIR / "njdg_features.csv")
    ibc  = pd.read_csv(PROCESSED_DIR / "ibc_features.csv")

    fc_njdg = get_feature_cols()
    fc_ibc  = get_ibc_feature_cols()

    X_njdg = njdg[fc_njdg].fillna(0).values
    X_ibc  = ibc[fc_ibc].fillna(0).values

    best_params = {}
    all_trials  = []

    # 1. Outcome classifier
    y_outcome = njdg["favourable_outcome"].values
    params_outcome, study_outcome = tune_model(
        "Outcome classifier", _objective_outcome, X_njdg, y_outcome
    )
    best_params["outcome"] = params_outcome
    for t in study_outcome.trials:
        all_trials.append({"model": "outcome", "trial": t.number,
                           "score": t.value, **t.params})

    # 2. Duration regressor
    y_duration = njdg["duration_months"].values
    params_duration, study_duration = tune_model(
        "Duration regressor", _objective_duration, X_njdg, y_duration
    )
    best_params["duration"] = params_duration
    for t in study_duration.trials:
        all_trials.append({"model": "duration", "trial": t.number,
                           "score": t.value, **t.params})

    # 3. Realisation regressor  (IBC only — filter to rows with realisation_pct)
    ibc_real = ibc[ibc["realisation_pct"].notna()].copy()
    X_real = ibc_real[fc_ibc].fillna(0).values
    y_real = ibc_real["realisation_pct"].values
    params_real, study_real = tune_model(
        "Realisation regressor", _objective_realisation, X_real, y_real
    )
    best_params["realisation"] = params_real
    for t in study_real.trials:
        all_trials.append({"model": "realisation", "trial": t.number,
                           "score": t.value, **t.params})

    # Save best params
    out_path = MODELS_DIR / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[tune] Best params saved to {out_path}")

    # Save full trial history for diagnostics
    pd.DataFrame(all_trials).to_csv(MODELS_DIR / "cv_results.csv", index=False)
    print(f"[tune] Trial history saved to {MODELS_DIR / 'cv_results.csv'}")


if __name__ == "__main__":
    main()