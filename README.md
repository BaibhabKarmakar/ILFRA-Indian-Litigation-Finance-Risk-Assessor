# ILFRA — Indian Litigation Finance Risk Assessor

**ILFRA** (LitFin Risk Assessor) is a machine learning-based advisory tool that evaluates risks associated with civil and commercial litigation in India. It is designed to assist litigation funders, lawyers, and businesses by predicting key case trajectories based on historical patterns extracted from public Indian government sources such as the National Judicial Data Grid (NJDG), IBBI CIRP Data, and eCourts judgments.

---

## What Does It Predict?

The tool evaluates a case and generates three main predictive outputs:

1. **Expected Duration** — The probable length of time the case proceedings will take, complete with confidence intervals (Optimistic, Median, and Pessimistic timelines).
2. **Probability of Favourable Outcome** — A calibrated percentage likelihood of a positive result (e.g., winning the lawsuit or reaching a settlement).
3. **Recovery / Realisation Range** — The expected percentage of the claim amount that might be recovered (specifically relevant for Money Recovery and IBC/Insolvency cases).
4. **Similar Precedents** — The K most similar historical cases retrieved from the case base, with similarity-weighted outcome estimates and a natural language precedent summary.

---

## System Architecture

The project is structured into the following key modules:

- **`src/data_ingestion.py`** — Fetches raw data from NJDG, IBBI, and eCourts, or synthesises realistic mock data (~5,000 records) to emulate civil and commercial litigations in India for development and testing.
- **`src/feature_engineering.py`** — Transforms raw data into numerical and categorical features suitable for ML modelling, with separate pipelines for NJDG and IBC data.
- **`src/ingestion/ibbi_channel.py`** — Orchestrates the full IBBI real data ingestion pipeline. Scans the source folder, routes each file to the correct parser, deduplicates across quarters, derives features, validates, and saves `data/raw/ibbi_real.csv`.
- **`src/parsers/ibbi_excel.py`** — Parses IBBI quarterly newsletter `.xlsx` files. Extracts Table 8/5/9 (Resolution Plans) and Table 14/11/10/9/15 (Closed Liquidations), handling IBBI's inconsistent table numbering across quarterly releases via content-based sheet detection.
- **`src/parsers/ibbi_pdf.py`** — Parses IBBI quarterly newsletter `.pdf` files using `pdfplumber`, handling multi-page tables with repeated headers.
- **`src/genai/genai_utils.py`** — Deterministic utility functions for column name mapping (fuzzy string matching via `rapidfuzz`) and company name normalisation (regex-based). Replaced a planned LLM-based approach with a free, deterministic alternative.
- **`src/tune.py`** — Runs Optuna-based hyperparameter search with 5-fold cross-validation across all three model families. Outputs `models/best_params.json` consumed by `train.py`.
- **`src/train.py`** — Trains three model families sequentially using tuned hyperparameters: LightGBM Regressor (Duration), LightGBM Classifier (Outcome), and LightGBM Regressor (Realisation). Automatically triggers calibration on completion.
- **`src/calibration.py`** — Wraps the trained outcome classifier with isotonic regression calibration so that predicted probabilities are statistically reliable. Outputs `models/outcome_calibrated.pkl` and calibration curve CSVs.
- **`src/cbr_case_base.py`** — Builds and serialises the searchable CBR case base from processed NJDG and IBC feature data. Run once after feature engineering. Outputs `models/cbr_case_base.pkl`.
- **`src/cbr_engine.py`** — Core CBR retrieval and adaptation engine. Computes weighted cosine similarity between a new case query and every historical case, retrieves the K most similar, and derives similarity-weighted outcome estimates.
- **`src/cbr_explainer.py`** — Converts retrieved precedents into natural language summaries and a blended ML + CBR interpretation for display in the Streamlit UI.
- **`app/streamlit_app.py`** — The Streamlit frontend. Processes user inputs and outputs a full risk assessment dashboard including ML predictions, a calibration reliability diagram, and a Similar Precedents panel.
- **`check_ibbi_files.py`** — Maintenance utility. Run before adding new quarterly IBBI files to verify that the parser can locate the resolution and liquidation tables correctly, and flag any files that need attention before the pipeline is run.

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

Raw euclidean distance on mixed features is misleading because features like court type (encoded 0–5) and claim amount (potentially thousands of lakhs) have incompatible scales and different semantic importance. ILFRA uses **weighted cosine similarity** where each feature dimension is multiplied by a domain importance weight derived from LightGBM feature importance rankings before the similarity is computed. This means a mismatch on `case_type_enc` (weight 3.0) penalises similarity far more than a mismatch on `filing_quarter` (weight 0.8).

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

**What the UI shows:**

- A natural language precedent summary ("5 similar cases found — 4 resolved favourably, average duration 31 months")
- A blended ML + CBR commentary that explicitly flags agreements and divergences between the two approaches
- Individual precedent cards showing similarity score, duration, outcome, and recovery % for each retrieved case

**Why CBR matters for litigation finance:**

ML models produce a number. CBR produces an explanation. A litigation funder evaluating a ₹50Cr commercial dispute can point to five specific precedent cases from the same court and case type that resolved in a comparable timeframe — that is auditable, defensible, and legally meaningful in a way that a gradient boosting score alone is not. When ML and CBR estimates agree, that convergence is a strong confidence signal. When they diverge, it flags genuine uncertainty that a single model score would have hidden entirely.

### Real IBBI Data Pipeline

ILFRA now ingests real IBBI quarterly newsletter Excel files directly from `data/raw/ibbi/` instead of relying solely on synthetic data. The pipeline currently processes **1,332 real CIRP cases** spanning 12 quarters (Q1 2023 – Q4 2025), covering 702 resolution and 630 liquidation outcomes with an average creditor realisation of 20.7% — consistent with IBBI's own published aggregate figures.

**Key design decisions:**

- **Content-based sheet detection** — IBBI renumbers its tables almost every quarter (the resolution data has appeared in Table 5, Table 8, and Table 9 across different releases; liquidation data in Table 9, 10, 11, 14, and 15). Rather than maintaining a hardcoded table number map, the parser scans every sheet's title row for domain-specific keywords ("cirps yielding", "closed liquidations") and selects the correct sheet regardless of its number.
- **Plugin architecture** — `ibbi_channel.py` is a format-agnostic orchestrator. Adding support for a new file format in the future requires only creating a new parser file and registering its extension in the `PARSERS` dict — no other file changes.
- **Deterministic column mapping** — Column names vary across quarterly releases (e.g., "Total Admitted Claims" vs "Amount of Claims Admitted"). Fuzzy string matching via `rapidfuzz` maps raw headers to a canonical internal vocabulary without any API calls or LLM dependency.
- **String date handling** — Newer IBBI files store dates as DD.MM.YYYY strings rather than datetime objects. The pipeline detects and parses both formats, with a sanity check that drops any dates before 2016 (when IBC was enacted).
- **Quarterly deduplication** — Each quarterly file includes a "Part A: Prior Period" section repeating cases from earlier quarters. The pipeline deduplicates on `company_name + cirp_start_date`, keeping the most recent record.

---

## How to Run the Tool

Follow these steps sequentially inside the project directory.

### 1. Set Up the Environment

```bash
pip install -r requirements.txt
```

### 2. Ingest Real IBBI Data *(if you have quarterly files)*

Place IBBI quarterly `.xlsx` files into `data/raw/ibbi/`, then run the checker to verify all files are parseable:

```bash
python check_ibbi_files.py
```

If the checker passes, run the ingestion pipeline:

```bash
python src/ingestion/ibbi_channel.py
```

This produces `data/raw/ibbi_real.csv` with cleaned, deduplicated, feature-enriched CIRP case data.

### 3. Generate Synthetic Data *(if no real files available)*

Generates `data/raw/` CSV files used as the training dataset:

```bash
python src/data_ingestion.py
```

### 4. Engineer Features

Transforms raw datasets into ML-ready feature matrices:

```bash
python src/feature_engineering.py
```

### 5. Build the CBR Case Base

Serialises the processed feature data into the searchable case base used by the CBR engine at inference time. Only needs to be re-run if the underlying processed data changes:

```bash
python src/cbr_case_base.py
```

### 6. Tune Hyperparameters *(recommended, ~5–10 min)*

Runs Optuna search with 5-fold CV and saves best parameters to `models/best_params.json`:

```bash
python src/tune.py
```

> This step is optional but strongly recommended before training on real data. If skipped, `train.py` uses built-in defaults.

### 7. Train the ML Models

Trains all three model families using tuned parameters and runs calibration automatically on completion:

```bash
python src/train.py
```

This produces the following artefacts in `models/`:

```
duration_model.pkl            duration_q10.pkl           duration_q90.pkl
outcome_model.pkl             outcome_calibrated.pkl
realisation_model.pkl         realisation_q10.pkl        realisation_q90.pkl
training_metrics.csv          best_params.json
calibration_curve_raw.csv     calibration_curve_cal.csv
cbr_case_base.pkl
*_feature_importance.csv
```

### 8. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. Navigate through the **Case Assessment**, **Model Insights**, and **How It Works** tabs to interact with predictions, inspect model behaviour including the calibration reliability diagram, and explore similar precedent cases retrieved by the CBR engine.

---

## Adding New IBBI Quarterly Files

When IBBI releases a new quarterly newsletter:

1. Download the `.xlsx` file and place it in `data/raw/ibbi/`
2. Run `python check_ibbi_files.py` — it will confirm whether the parser can locate the correct sheets, or flag which file needs attention
3. If a file is flagged, open `debug_xlsx.py`, point it at the new file, and identify which table number IBBI used for resolution and liquidation data in that release
4. Add the new table number to `_find_sheet_by_content` keywords in `src/parsers/ibbi_excel.py`
5. Re-run the full pipeline from Step 2 above

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

ILFRA is an **advisory tool only**. Its predictions are based on statistical patterns and retrieved precedents, and carry inherent uncertainty. They should not be treated as legal advice or as a guarantee of case outcome. All funding and legal decisions must involve qualified legal professionals.