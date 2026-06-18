# ILFRA — Indian Litigation Finance Risk Assessor

**ILFRA** is a machine learning advisory tool for evaluating litigation finance risk under India's Insolvency and Bankruptcy Code (IBC). It is designed to help third-party litigation funders — such as LegalPay and FightRight — assess CIRP (Corporate Insolvency Resolution Process) cases by generating structured, data-driven estimates of case duration, probability of a favourable outcome, and expected creditor recovery.

ILFRA is an **advisory tool only**. It provides a structured data point to support, but not replace, independent legal due diligence by qualified professionals.

---

## Problem Statement

Third-party litigation funders in India provide upfront capital for legal costs in exchange for a share of any successful recovery. Case evaluation today relies entirely on manual due diligence by in-house lawyers — reviewing case documents, lawyer track records, opposing party strength, and available public information. This process is time-intensive and resource-heavy, limiting the number of cases that can be assessed in detail.

ILFRA trains ML models on publicly available IBBI CIRP metadata to generate structured risk estimates across three dimensions: expected duration, probability of resolution (vs. liquidation), and percentage of admitted claims likely to be recovered. These outputs serve as a fast, consistent first-pass signal alongside lawyer review.

---

## What Does It Predict?

For each CIRP case submitted, ILFRA produces:

1. **Expected Duration** — Median case length in months with P10 (optimistic) and P90 (pessimistic) quantile bounds.
2. **Probability of Favourable Outcome** — Calibrated probability of a Resolution Plan being approved (vs. liquidation or withdrawal).
3. **Recovery / Realisation Range** — Expected percentage of admitted financial claims likely to be recovered, with P10/P90 confidence bounds.
4. **Composite Risk Score** — A 0–100 score combining all three signals weighted by funder relevance.
5. **Similar Precedents (CBR)** — The 5 most similar historical CIRP cases, retrieved by weighted cosine similarity, with similarity-weighted outcome estimates and a natural language precedent summary.
6. **SHAP Feature Contributions** — Per-case explanation of which features drove each prediction up or down, rendered as waterfall charts.

---

## System Architecture

### Data & Ingestion

- **`src/ingestion/ibbi_channel.py`** — Full IBBI ingestion orchestrator. Scans `data/raw/ibbi/` for quarterly files, routes each to the correct parser, deduplicates across quarters on `(company_name, cirp_start_date)`, derives features, validates, and writes `data/raw/ibbi_real.csv`. Currently processes **~1,932 real CIRP cases** from **36 quarterly Excel and PDF files** (Q1 2017 – Q2 2026).
- **`src/parsers/ibbi_excel.py`** — Parses IBBI quarterly `.xlsx` files. Extracts the resolution plan table ("CIRPs Yielding Resolution") and the closed liquidation table ("Details of Closed Liquidations") using content-based sheet detection — robust to IBBI's frequent table renumbering across quarterly releases.
- **`src/parsers/ibbi_pdf.py`** — Parses IBBI quarterly `.pdf` files using `pdfplumber`.
- **`check_ibbi_files.py`** — Maintenance utility. Run before adding new quarterly files to verify the parser finds the correct sheets.

### Feature Engineering

- **`src/training/feature_engineering.py`** — Transforms `ibbi_real.csv` into model-ready feature matrices. Key engineered features:
  - `log_admitted_claim` — log₁₊ of admitted claims in ₹ crore
  - `log_liquidation_value` — log₁₊ of liquidation value
  - `claim_to_liquidation_ratio` — admitted claims / liquidation value, clipped at 100 (the strongest predictive feature)
  - `is_large_case` — flag for cases above ₹500 crore
  - `admission_year` — year CIRP was admitted
  - `sector_enc` — label-encoded sector

### ML Models

Three LightGBM models are trained in `src/pipelines/ibbi_pipeline.py`:

| Model | Target | Features | Metric |
|---|---|---|---|
| Outcome classifier | `favourable_outcome` (binary) | Outcome feature cols (excl. duration — leaky) | AUC 0.852 |
| Duration regressor | `duration_days` | Duration feature cols | R² 0.256, MAE ~307 days |
| Realisation regressor | `realisation_pct` | Full IBC feature cols (incl. duration, outcome — known for closed cases) | R² 0.573 |

Quantile variants (P10, P90) are trained using LightGBM's `objective="quantile"` with pinball loss to produce calibrated confidence intervals.

**Calibration:** The outcome classifier uses **Platt (sigmoid) scaling** via `CalibratedClassifierCV(method="sigmoid")` on a held-out 20% calibration set. Isotonic regression was found unstable at the ~380-row calibration set size (overfits to near-zero ECE). Calibration curves are saved for the reliability diagram in the Streamlit dashboard.

### Inference

- **`src/inference/predict.py`** — Loads all models and SHAP explainers, runs inference for a single submitted case, computes per-case SHAP values for all three models, retrieves CBR precedents, and returns the full structured result.

### Case-Based Reasoning (CBR)

ILFRA augments ML predictions with a CBR engine that reasons by precedent — mirroring how legal practitioners actually evaluate cases.

The engine follows the classical **4R CBR cycle**: **Retrieve** → **Reuse** → **Revise** → **Retain**.

**Similarity metric — weighted cosine similarity:**

Raw Euclidean distance on mixed features is unreliable because features like claim amount and sector encoding have incompatible scales. ILFRA uses weighted cosine similarity where each feature dimension is multiplied by a domain importance weight informed by LightGBM feature importances.

**IBC CBR feature weights:**

| Feature | Weight |
|---|---|
| `claim_to_liquidation_ratio` | 3.0 |
| `log_admitted_claim` | 2.8 |
| `log_liquidation_value` | 2.5 |
| `duration_days` | 2.0 |
| `admission_year` | 1.5 |
| `favourable_outcome` | 1.2 |
| Other features | 0.3 – 0.5 |

CBR retrieval is restricted to **resolved cases only** — ongoing cases have no outcome to reuse. Adapted estimates use similarity-weighted averaging (closer precedents contribute proportionally more).

- **`src/cbr/cbr_engine.py`** — Weighted cosine similarity retrieval and similarity-weighted outcome adaptation.
- **`src/cbr/cbr_explainer.py`** — Natural language precedent summaries and blended ML + CBR interpretation.
- **`src/cbr_case_base.py`** — Builds `models/cbr_case_base.pkl` from processed IBC feature data. Run once after feature engineering.

### SHAP Explainability

ILFRA uses `shap.TreeExplainer` (exact Tree SHAP algorithm) to compute per-case feature contributions for each of the three models. This answers not just "which features matter globally" but "why did this specific case receive this specific score."

- At training time, global mean |SHAP| values are saved to `models/{name}_shap_values.csv` for the Model Insights tab.
- At inference time, per-case SHAP values are returned under `result["shap"]` and rendered as horizontal waterfall charts.
- SHAP importance is preferred over LightGBM's native gain-based importance because it is measured in prediction units and correctly handles feature correlations.

### GenAI Utilities

- **`src/genai/genai_utils.py`** — HuggingFace Inference API integration (`mistralai/Mistral-7B-Instruct-v0.3`) for two tasks:
  - **Column detection** — maps raw Excel headers to canonical schema names (falls back to empty mapping if API unavailable)
  - **Risk narrative generation** — 2–3 sentence plain-English risk summary for the PDF report (falls back to deterministic template)
  - All numerical predictions are fully deterministic; LLM is used only for natural language and ambiguous column mapping.

### Frontend & Reporting

- **`streamlit_app.py`** — Three-tab Streamlit dashboard:
  - **Tab 1 (Case Assessment)** — Form input, KPI cards, duration bar chart, outcome gauge, per-case SHAP waterfall charts, CBR precedents, PDF download.
  - **Tab 2 (Model Insights)** — Training metrics, global SHAP importance charts, calibration reliability diagram.
  - **Tab 3 (How It Works)** — Pipeline architecture and ethical guardrails.
- **`src/inference/report_generator.py`** — ReportLab PDF report with metric cards, outcome bar, risk factors, GenAI narrative, and disclaimer.

---

## Confirmed Model Metrics

Trained on ~1,932 real IBBI CIRP cases (36 quarterly files, Q1 2017 – Q2 2026):

| Model | Metric | Value |
|---|---|---|
| Outcome classifier | AUC | 0.852 |
| Outcome classifier | Accuracy | 76% |
| Realisation regressor | R² | 0.593 |
| Duration regressor | R² | 0.256 |

Duration R² of 0.26 reflects genuine data-level ambiguity (CIRP duration is highly noisy and case-specific) rather than a modelling deficiency — the model correctly learns the general distribution shape even if individual case predictions carry wide uncertainty.

---

## Known Limitations

- **Dataset size:** ~1,932 cases; ~136 positive test cases for AUC estimation — confidence intervals are wide.
- **Survival bias:** Recent quarterly files contain a higher proportion of fast-resolving cases because slow-resolving ongoing cases are not yet in the data. This inflates observed resolution rates for post-2022 cohorts.
- **NJDG pipeline:** The NJDG data pipeline is designed but inactive — NJDG does not expose case-level CSV exports, and scraping is not permitted under India's IT Act 2000. The NJDG pipeline is framed as a future capability pending data access.
- **CBR weights:** IBC CBR feature weights are informed by LightGBM feature importances but assigned manually, not algorithmically derived.
- **48-month normalisation:** The duration risk normalisation constant (48 months = IBC statutory outer limit) is a domain heuristic, not learned from data.

---

## How to Run


### 1. Set Up the Environment

```bash
pip install -r requirements.txt
```

### 2. Ingest Real IBBI Data

Place IBBI quarterly `.xlsx` files into `data/raw/ibbi/`, then:

```bash
python check_ibbi_files.py              # verify sheets are parseable
python src/ingestion/ibbi_channel.py    # → data/raw/ibbi_real.csv
```

### 3. Run the IBBI Pipeline (Feature Engineering + Training)

```bash
python src/train.py                     # calls ibbi_pipeline.py internally
```

This produces all model artefacts in `models/`:

```
ibc_duration_model.pkl    ibc_duration_q10.pkl      ibc_duration_q90.pkl
ibc_outcome_model.pkl
realisation_model.pkl     realisation_q10.pkl       realisation_q90.pkl
ibc_encoders.pkl
training_metrics.csv      best_params.json
ibc_outcome_shap_explainer.pkl    ibc_outcome_shap_values.csv
ibc_duration_shap_explainer.pkl   ibc_duration_shap_values.csv
realisation_shap_explainer.pkl    realisation_shap_values.csv
*_feature_importance.csv
```

### 4. Build the CBR Case Base

```bash
python src/cbr_case_base.py             # → models/cbr_case_base.pkl
```

### 5. (Optional) Tune Hyperparameters

```bash
python src/tune.py                      # → models/best_params.json
```

Run before Step 3 if you want Optuna-tuned hyperparameters (40 trials × 5-fold CV per model, ~5–10 min). `train.py` falls back to sensible defaults if `best_params.json` is absent.

### 6. Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Adding New IBBI Quarterly Files

1. Drop the new `.xlsx` file into `data/raw/ibbi/`
2. Run `python check_ibbi_files.py` — it prints which sheet was found for resolution and liquidation tables and counts data rows
3. If a sheet is flagged as NOT FOUND, inspect with `debug_xlsx.py` and add the relevant keyword to `_find_sheet_by_content()` in `src/parsers/ibbi_excel.py`
4. Re-run from Step 2 above

---

## Data Sources

| Source | URL | Used for |
|---|---|---|
| IBBI CIRP quarterly newsletters | `ibbi.gov.in` | All three models; IBC CBR case base (~1,932 real cases) |
| NJDG | `njdg.ecourts.gov.in` | Future: duration and outcome models for civil cases |
| eCourts | `judgments.ecourts.gov.in` | Future: judgment outcome labels |

---

## Project Structure

```
ILFRA/
├── data/
│   ├── raw/
│   │   ├── ibbi/           ← drop quarterly .xlsx files here
│   │   └── ibbi_real.csv   ← output of ibbi_channel.py
│   └── processed/
│       └── ibc_features.csv
├── models/                 ← all .pkl and .csv artefacts
├── src/
│   ├── ingestion/
│   │   └── ibbi_channel.py
│   ├── parsers/
│   │   ├── ibbi_excel.py
│   │   └── ibbi_pdf.py
│   ├── training/
│   │   └── feature_engineering.py
│   ├── pipelines/
│   │   └── ibbi_pipeline.py
│   ├── inference/
│   │   ├── predict.py
│   │   └── report_generator.py
│   ├── cbr/
│   │   ├── cbr_engine.py
│   │   └── cbr_explainer.py
│   └── genai/
│       └── genai_utils.py
├── streamlit_app.py
├── cbr_case_base.py
├── train.py
├── tune.py
├── check_ibbi_files.py
├── debug_xlsx.py
└── requirements.txt
```

---

## Ethical Disclaimer

ILFRA is an **advisory tool only**. Its predictions are based on statistical patterns in historical IBBI data and retrieved precedents, and carry inherent uncertainty. They should not be treated as legal advice or a guarantee of case outcome. SHAP values explain model behaviour but do not imply causal relationships between features and outcomes. All litigation funding and legal decisions must involve qualified legal professionals.
