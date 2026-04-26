# ILFRA — Indian Litigation Finance Risk Assessor

**ILFRA** (LitFin Risk Assessor) is a machine learning-based advisory tool that evaluates risks associated with civil and commercial litigation in India. It is designed to assist litigation funders, lawyers, and businesses by predicting key case trajectories based on historical patterns extracted from public Indian government sources such as the National Judicial Data Grid (NJDG), IBBI CIRP Data, and eCourts judgments.

---

## What Does It Predict?

The tool evaluates a case and generates three main predictive outputs:

1. **Expected Duration** — The probable length of time the case proceedings will take, complete with confidence intervals (Optimistic, Median, and Pessimistic timelines).
2. **Probability of Favourable Outcome** — A calibrated percentage likelihood of a positive result (e.g., winning the lawsuit or reaching a settlement).
3. **Recovery / Realisation Range** — The expected percentage of the claim amount that might be recovered (specifically relevant for Money Recovery and IBC/Insolvency cases).
4. **Similar Precedents** — The K most similar historical cases retrieved from the case base, with similarity-weighted outcome estimates and a natural language precedent summary.
5. **SHAP Feature Contributions** — Per-case explanation of which features drove each prediction up or down, rendered as waterfall charts in the dashboard.

---

## System Architecture

The project is structured into the following key modules:

- **`src/data_ingestion.py`** — Fetches raw data from NJDG, IBBI, and eCourts, or synthesises realistic mock data (~5,000 records) to emulate civil and commercial litigations in India for development and testing.
- **`src/feature_engineering.py`** — Transforms raw data into numerical and categorical features suitable for ML modelling, with separate pipelines for NJDG and IBC data.
- **`src/ingestion/ibbi_channel.py`** — Orchestrates the full IBBI real data ingestion pipeline. Scans the source folder, routes each file to the correct parser, deduplicates across quarters, derives features, validates, and saves `data/raw/ibbi_real.csv`.
- **`src/parsers/ibbi_excel.py`** — Parses IBBI quarterly newsletter `.xlsx` files. Extracts resolution plan and closed liquidation tables, handling IBBI's inconsistent table numbering across quarterly releases via content-based sheet detection.
- **`src/parsers/ibbi_pdf.py`** — Parses IBBI quarterly newsletter `.pdf` files using `pdfplumber`, handling multi-page tables with repeated headers.
- **`src/genai/genai_utils.py`** — Deterministic utility functions for column name mapping (fuzzy string matching via `rapidfuzz`) and company name normalisation (regex-based).
- **`src/tune.py`** — Runs Optuna-based hyperparameter search with 5-fold cross-validation across all three model families. Outputs `models/best_params.json` consumed by `train.py`.
- **`src/train.py`** — Trains three model families sequentially using tuned hyperparameters. After each model trains, computes SHAP values using `shap.TreeExplainer` and saves the explainer objects and global SHAP summaries to `models/`. Automatically triggers calibration on completion.
- **`src/calibration.py`** — Wraps the trained outcome classifier with isotonic regression calibration so that predicted probabilities are statistically reliable. Outputs `models/outcome_calibrated.pkl` and calibration curve CSVs.
- **`src/cbr_case_base.py`** — Builds and serialises the searchable CBR case base from processed NJDG and IBC feature data. Run once after feature engineering. Outputs `models/cbr_case_base.pkl`.
- **`src/cbr_engine.py`** — Core CBR retrieval and adaptation engine. Computes weighted cosine similarity between a new case query and every historical case, retrieves the K most similar, and derives similarity-weighted outcome estimates.
- **`src/cbr_explainer.py`** — Converts retrieved precedents into natural language summaries and a blended ML + CBR interpretation for display in the Streamlit UI.
- **`src/predict.py`** — Loads trained models and SHAP explainers, runs inference for a single case, computes per-case SHAP values for all three models, and returns the full structured result including SHAP contributions under a `"shap"` key.
- **`app/streamlit_app.py`** — The Streamlit frontend. Tab 1 shows ML predictions with per-case SHAP waterfall charts. Tab 2 shows global SHAP summary plots (replacing LightGBM importance) and the calibration reliability diagram. Tab 3 covers architecture and ethical guardrails.
- **`check_ibbi_files.py`** — Maintenance utility. Run before adding new quarterly IBBI files to verify that the parser can locate the resolution and liquidation tables correctly.

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

### Case-Based Reasoning (CBR)

ILFRA augments its ML predictions with a **Case-Based Reasoning** engine inspired by how legal practitioners actually reason — through precedent. Rather than relying solely on statistical patterns, CBR retrieves the most similar historical cases and adapts their known outcomes to inform the current assessment.

The engine follows the classical **4R CBR cycle**:

1. **Retrieve** — Given a new case query, compute weighted cosine similarity against every case in the case base and return the K most similar (default K = 5).
2. **Reuse** — Derive adapted outcome estimates (duration, win probability, realisation %) using similarity-weighted averaging, so closer precedents contribute more than distant ones.
3. **Revise** — Blend the CBR-adapted estimates with the ML model predictions. When the two sources agree, confidence is high; when they diverge, the discrepancy is surfaced explicitly as a risk flag.
4. **Retain** — The case base persists across sessions and can be updated with new resolved cases as they become available.

**Similarity metric — weighted cosine similarity:**

Raw euclidean distance on mixed features is misleading because features like court type (encoded 0–5) and claim amount (potentially thousands of lakhs) have incompatible scales and different semantic importance. ILFRA uses **weighted cosine similarity** where each feature dimension is multiplied by a domain importance weight derived from LightGBM feature importance rankings before the similarity is computed.

**Feature weights (NJDG case base):**

| Feature | Weight |
|---|---|
| `case_type_enc` | 3.0 |
| `court_enc` | 2.5 |
| `claimant_lawyer_win_rate` | 2.0 |
| `court_hierarchy` | 2.0 |
| `log_claim_amount` | 1.8 |
| `court_avg_duration` | 1.5 |
| `court_avg_win_rate` | 1.5 |
| `adjournment_density` | 1.5 |
| `has_interim_order` | 1.5 |
| `respondent_is_govt` | 1.3 |
| Other features | 0.8 – 1.2 |

### SHAP Explainability

ILFRA uses **SHAP (SHapley Additive exPlanations)** to make every prediction interpretable at the individual case level. Rather than relying on global feature importance averages, SHAP computes the exact contribution of each feature to the prediction for a specific case — answering not just "what features matter in general" but "why did this particular case get this particular score."

**How it works at training time (`train.py`):**

- After each model trains, `shap.TreeExplainer` is fitted on the trained model
- `TreeExplainer` uses the exact Tree SHAP algorithm — it is the correct choice for LightGBM models and is orders of magnitude faster than the sampling-based `KernelExplainer`
- Global SHAP importance (mean absolute SHAP value per feature across the test set) is saved to `models/{model_name}_shap_values.csv`
- The explainer object itself is saved to `models/{model_name}_shap_explainer.pkl` for use at inference time

**How it works at inference time (`predict.py`):**

- When a case is assessed, the loaded SHAP explainer computes per-case SHAP values for the single input row
- For the binary outcome classifier, SHAP values for the positive class (favourable outcome) are extracted
- Results are returned in `result["shap"]` with sub-keys `"duration"`, `"outcome"`, and `"realisation"`, each containing a dict of `feature → shap_value` sorted by absolute magnitude

**What the UI shows:**

- **Tab 1 (Case Assessment)** — After the prediction KPI cards, a SHAP waterfall chart per model shows which features pushed this specific case's prediction up (red) or down (blue) from the model's baseline. The magnitude of each bar is in the model's output units: months for duration, log-odds contribution for outcome, and percentage points for realisation.
- **Tab 2 (Model Insights)** — Global SHAP summary bars replace LightGBM's native gain-based importance charts. SHAP-based global importance is more reliable because it is measured in prediction units and correctly accounts for feature interactions.

**Why SHAP over LightGBM feature importance:**

LightGBM's built-in importance measures how often a feature is used in splits (frequency) or the total gain from splits using that feature (gain). Both can be misleading — a feature used in many low-value splits looks important by frequency, and correlated features split the true importance between them. SHAP importance is measured directly in prediction units, is consistent regardless of feature correlation, and supports per-case explanations that gain-based importance fundamentally cannot provide.

### Real IBBI Data Pipeline

ILFRA now ingests real IBBI quarterly newsletter Excel files directly from `data/raw/ibbi/` instead of relying solely on synthetic data. The pipeline currently processes **1,332 real CIRP cases** spanning 12 quarters (Q1 2023 – Q4 2025), covering 702 resolution and 630 liquidation outcomes with an average creditor realisation of 20.7% — consistent with IBBI's own published aggregate figures.

**Key design decisions:**

- **Content-based sheet detection** — IBBI renumbers its tables almost every quarter. Rather than maintaining a hardcoded table number map, the parser scans every sheet's title row for domain-specific keywords and selects the correct sheet regardless of its number.
- **Plugin architecture** — `ibbi_channel.py` is a format-agnostic orchestrator. Adding support for a new file format requires only creating a new parser file and registering its extension in the `PARSERS` dict.
- **Deterministic column mapping** — Fuzzy string matching via `rapidfuzz` maps raw headers to a canonical internal vocabulary without any API calls or LLM dependency.
- **String date handling** — Newer IBBI files store dates as DD.MM.YYYY strings. The pipeline detects and parses both formats, with a sanity check that drops dates before 2016.
- **Quarterly deduplication** — Each quarterly file includes a "Part A: Prior Period" section repeating earlier cases. The pipeline deduplicates on `company_name + cirp_start_date`, keeping the most recent record.

---

## How to Run the Tool

Follow these steps sequentially inside the project directory.

### 1. Set Up the Environment

```bash
pip install -r requirements.txt
```

### 2. Ingest Real IBBI Data *(if you have quarterly files)*

Place IBBI quarterly `.xlsx` files into `data/raw/ibbi/`, verify they are parseable, then run:

```bash
python check_ibbi_files.py
python src/ingestion/ibbi_channel.py
```

### 3. Generate Synthetic Data *(if no real files available)*

```bash
python src/data_ingestion.py
```

### 4. Engineer Features

```bash
python src/feature_engineering.py
```

### 5. Build the CBR Case Base

```bash
python src/cbr_case_base.py
```

### 6. Tune Hyperparameters *(recommended, ~5–10 min)*

```bash
python src/tune.py
```

> Optional but strongly recommended before training on real data.

### 7. Train the ML Models

Trains all three model families, computes SHAP explainers, and runs calibration automatically:

```bash
python src/train.py
```

This produces the following artefacts in `models/`:

```
duration_model.pkl              duration_q10.pkl              duration_q90.pkl
outcome_model.pkl               outcome_calibrated.pkl
realisation_model.pkl           realisation_q10.pkl           realisation_q90.pkl
training_metrics.csv            best_params.json
calibration_curve_raw.csv       calibration_curve_cal.csv
cbr_case_base.pkl
duration_shap_explainer.pkl     duration_shap_values.csv
outcome_shap_explainer.pkl      outcome_shap_values.csv
realisation_shap_explainer.pkl  realisation_shap_values.csv
*_feature_importance.csv
```

### 8. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. Navigate through the **Case Assessment**, **Model Insights**, and **How It Works** tabs to interact with predictions, inspect SHAP feature contributions, review the calibration reliability diagram, and explore similar precedent cases.

---

## Adding New IBBI Quarterly Files

1. Drop the new `.xlsx` file into `data/raw/ibbi/`
2. Run `python check_ibbi_files.py` to verify the parser finds the correct sheets
3. If flagged, inspect with `debug_xlsx.py` and update keywords in `src/parsers/ibbi_excel.py`
4. Re-run the full pipeline from Step 2 above

---

## Data Sources

| Source | Portal | Used for |
|---|---|---|
| NJDG | `njdg.ecourts.gov.in` | Duration and outcome models, NJDG CBR case base |
| IBBI CIRP | `ibbi.gov.in` | Realisation model, IBC CBR case base (1,332 real cases) |
| eCourts | `ecourts.gov.in` | Judgment outcome labels |

> When real IBBI government exports are not available, the pipeline automatically falls back to synthetic data generators that mirror real-world Indian litigation distributions.

---

## Ethical Disclaimer

ILFRA is an **advisory tool only**. Its predictions are based on statistical patterns and retrieved precedents, and carry inherent uncertainty. They should not be treated as legal advice or as a guarantee of case outcome. SHAP values explain model behaviour but do not imply causal relationships between features and case outcomes. All funding and legal decisions must involve qualified legal professionals.