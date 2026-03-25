# ILFRA — Indian Litigation Finance Risk Assessor

**ILFRA** (LitFin Risk Assessor) is a machine learning-based advisory tool that evaluates risks associated with civil and commercial litigation in India. It is designed to assist litigation funders, lawyers, and businesses by predicting key case trajectories based on historical patterns extracted from public Indian government sources such as the National Judicial Data Grid (NJDG), IBBI CIRP Data, and eCourts judgments.

---

## What Does It Predict?

The tool evaluates a case and generates three main predictive outputs:

1. **Expected Duration** — The probable length of time the case proceedings will take, complete with confidence intervals (Optimistic, Median, and Pessimistic timelines).
2. **Probability of Favourable Outcome** — A calibrated percentage likelihood of a positive result (e.g., winning the lawsuit or reaching a settlement).
3. **Recovery / Realisation Range** — The expected percentage of the claim amount that might be recovered (specifically relevant for Money Recovery and IBC/Insolvency cases).

---

## System Architecture

The project is structured into the following key modules:

- **`src/data_ingestion.py`** — Fetches raw data from NJDG, IBBI, and eCourts, or synthesises realistic mock data (~5,000 records) to emulate civil and commercial litigations in India for development and testing.
- **`src/feature_engineering.py`** — Transforms raw data into numerical and categorical features suitable for ML modelling, with separate pipelines for NJDG and IBC data.
- **`src/tune.py`** — Runs Optuna-based hyperparameter search with 5-fold cross-validation across all three model families. Outputs `models/best_params.json` consumed by `train.py`.
- **`src/train.py`** — Trains three model families sequentially using tuned hyperparameters: LightGBM Regressor (Duration), LightGBM Classifier (Outcome), and LightGBM Regressor (Realisation). Automatically triggers calibration on completion.
- **`src/calibration.py`** — Wraps the trained outcome classifier with isotonic regression calibration so that predicted probabilities are statistically reliable. Outputs `models/outcome_calibrated.pkl` and calibration curve CSVs.
- **`app/streamlit_app.py`** — The Streamlit frontend. Processes user inputs (Case Age, Lawyer Win Rate, Adjournments, Court Type, Sector, Interim Order status, etc.) and outputs a full risk assessment dashboard with model insights and a calibration reliability diagram.

---

## ML Pipeline Details

### Hyperparameter Tuning with Cross-Validation

All three models are tuned using **Optuna** with a Tree-structured Parzen Estimator (TPE) sampler — a Bayesian optimisation strategy that learns which hyperparameter regions score well and samples intelligently from them, converging significantly faster than grid or random search.

**How it works:**

- `src/tune.py` runs 40 Optuna trials per model (configurable via `N_TRIALS`)
- Each trial is evaluated using **5-fold cross-validation** rather than a single train/test split, making the metric estimates more stable on a dataset of ~5,000 rows
- The outcome classifier is scored on mean AUC across folds (stratified to preserve class balance); the duration and realisation regressors are scored on mean MAE
- Best parameters are saved to `models/best_params.json` and automatically loaded by `train.py`
- If `tune.py` has not been run, `train.py` falls back to sensible hardcoded defaults so the pipeline remains runnable out of the box

**Parameters tuned per model:**

| Parameter | Search range |
|---|---|
| `n_estimators` | 100 – 600 |
| `learning_rate` | 0.01 – 0.15 (log scale) |
| `num_leaves` | 15 – 127 |
| `min_child_samples` | 10 – 60 |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `reg_alpha` | 1e-4 – 10.0 (log scale) |
| `reg_lambda` | 1e-4 – 10.0 (log scale) |

### Confidence Calibration for Reliable Probabilities

Raw LightGBM classifier outputs are not true probabilities — a score of 0.72 does not mean 72% of such cases have a favourable outcome. `src/calibration.py` corrects this using **isotonic regression calibration** via scikit-learn's `CalibratedClassifierCV`.

**How it works:**

- A held-out calibration set (20% of NJDG data, never seen during training or tuning) is used to fit the isotonic layer on top of the frozen trained classifier
- `cv="prefit"` tells sklearn the base model is already trained — only the calibration mapping is fitted
- The calibrated model is saved as `models/outcome_calibrated.pkl` and used by `predict.py` automatically (falls back to the raw model if the calibrated file is absent)
- **Expected Calibration Error (ECE)** is computed before and after calibration and printed to the console during training, giving a concrete measure of improvement
- Calibration curves are saved to `models/calibration_curve_raw.csv` and `models/calibration_curve_cal.csv` for the reliability diagram rendered in the Model Insights tab

Isotonic regression is preferred over Platt scaling here because it makes fewer assumptions about the shape of the miscalibration — it only requires that the calibrated probabilities are monotonically increasing with the raw scores, which is well-suited to LightGBM's output distribution.

---

## How to Run the Tool

Follow these steps sequentially inside the project directory.

### 1. Set Up the Environment

```bash
pip install -r requirements.txt
```

### 2. Generate Raw Data

Generates `data/raw/` CSV files used as the training dataset:

```bash
python src/data_ingestion.py
```

### 3. Engineer Features

Transforms raw datasets into ML-ready feature matrices:

```bash
python src/feature_engineering.py
```

### 4. Tune Hyperparameters *(recommended, ~5–10 min)*

Runs Optuna search with 5-fold CV and saves best parameters to `models/best_params.json`:

```bash
python src/tune.py
```

> This step is optional but strongly recommended before training on real data. If skipped, `train.py` uses built-in defaults.

### 5. Train the ML Models

Trains all three model families using tuned parameters and runs calibration automatically on completion:

```bash
python src/train.py
```

This produces the following artefacts in `models/`:

```
duration_model.pkl          duration_q10.pkl         duration_q90.pkl
outcome_model.pkl           outcome_calibrated.pkl
realisation_model.pkl       realisation_q10.pkl      realisation_q90.pkl
training_metrics.csv        best_params.json
calibration_curve_raw.csv   calibration_curve_cal.csv
*_feature_importance.csv
```

### 6. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. Navigate through the **Case Assessment**, **Model Insights**, and **How It Works** tabs to interact with the predictions and inspect model behaviour including the calibration reliability diagram.

---

## Data Sources

| Source | Portal | Used for |
|---|---|---|
| NJDG | `njdg.ecourts.gov.in` | Duration and outcome models |
| IBBI CIRP | `ibbi.gov.in` | Realisation model |
| eCourts | `ecourts.gov.in` | Judgment outcome labels |

> When real government exports are not available, the pipeline automatically falls back to synthetic data generators that mirror real-world Indian litigation distributions.

---

## Ethical Disclaimer

ILFRA is an **advisory tool only**. Its predictions are based on statistical patterns and carry inherent uncertainty. They should not be treated as legal advice or as a guarantee of case outcome. All funding and legal decisions must involve qualified legal professionals.