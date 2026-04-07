"""
app/streamlit_app.py
--------------------
Streamlit UI for the Litigation Finance Risk Assessor.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.report_generator import generate_assessment_report

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import date
from src.cbr_explainer import summarise_precedents, blend_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LitFin Risk Assessor — India",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Lazy model loader (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    from src.predict import load_models as _load
    return _load()


@st.cache_data
def load_feature_importance():
    files = {
        "Duration": MODELS_DIR / "duration_feature_importance.csv",
        "Outcome": MODELS_DIR / "outcome_feature_importance.csv",
        "Realisation": MODELS_DIR / "realisation_feature_importance.csv",
    }
    return {
        k: pd.read_csv(v, index_col=0) for k, v in files.items() if v.exists()
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png", width=120)
    st.title("⚖️ LitFin Risk Assessor")
    st.caption("India — Civil & Commercial Disputes")
    st.divider()
    st.markdown("""
**Data sources**
- 🏛 NJDG (njdg.ecourts.gov.in)
- 📋 IBBI CIRP Data (ibbi.gov.in)
- 📄 eCourts Judgments

**Models**
- LightGBM Regressor (Duration)
- LightGBM Classifier (Outcome)
- LightGBM Regressor (Realisation %)

**Confidence intervals**: Quantile regression (P10 / P90)
    """)
    st.divider()
    st.warning("⚠️ Advisory tool only. All outputs must be reviewed by qualified lawyers before funding decisions.")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Case Assessment", "📊 Model Insights", "📖 How It Works"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Case Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Case Risk Assessment")
    st.markdown("Fill in the case details below to generate a structured risk estimate.")

    with st.form("case_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Case Details")
            case_type = st.selectbox("Case Type", [
                "Civil Suit", "Money Recovery", "Injunction", "Partition",
                "Specific Performance", "Arbitration", "Commercial Dispute",
                "CIRP (IBC)", "Liquidation (IBC)", "Writ (HC)", "Appeal (HC)",
                "Consumer Dispute", "Labour / Employment", "IP Infringement",
            ])
            court = st.selectbox("Court", [
                "District Court", "Commercial Court",
                "High Court (Original)", "High Court (Appeal)",
                "NCLT", "NCLAT", "Supreme Court",
                "Consumer Forum (District)", "Consumer Forum (State)",
            ])
            state = st.selectbox("State / UT", [
                "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana",
                "Gujarat", "West Bengal", "Rajasthan", "Uttar Pradesh", "Punjab",
            ])
            sector = st.selectbox("Sector", [
                "Real Estate", "Banking & Finance", "Infrastructure",
                "Manufacturing", "IT / Technology", "Healthcare",
                "Retail", "Telecom", "Energy", "Others",
            ])

        with col2:
            st.subheader("💰 Financial Details")
            claim_amount_lakhs = st.number_input(
                "Claim Amount (₹ Lakhs)", min_value=1.0, max_value=100000.0,
                value=100.0, step=10.0,
            )
            filing_date = st.date_input("Filing Date", value=date(2022, 1, 1))
            filing_year = filing_date.year
            filing_quarter = (filing_date.month - 1) // 3 + 1
            today = date.today()
            case_age_months = max(0, (today - filing_date).days // 30)
            st.info(f"📅 Case age: **{case_age_months} months**")

            st.subheader("🔄 Proceedings")
            num_prior_adjournments = st.slider("Prior Adjournments", 0, 60, 5)
            has_interim_order = st.checkbox("Interim / Stay Order Obtained")

        with col3:
            st.subheader("👥 Party & Counsel")
            claimant_lawyer_win_rate = st.slider(
                "Claimant Lawyer Historical Win Rate", 0.0, 1.0, 0.50, 0.05,
                help="Estimated from past cases; use 0.5 if unknown",
            )
            represented_by_senior_counsel = st.checkbox("Senior Counsel Engaged")
            respondent_is_govt = st.checkbox("Respondent is Government Body")
            respondent_is_psu = st.checkbox("Respondent is PSU")

            # IBC-specific
            is_ibc = "IBC" in case_type or "Liquidation" in case_type
            if is_ibc:
                st.subheader("🏦 IBC / CIRP Details")
                no_of_financial_creditors = st.number_input("No. of Financial Creditors", 1, 500, 10)
                resolution_applicants_received = st.number_input("Resolution Applicants Received", 0, 50, 3)
                ip_changed = st.checkbox("Insolvency Professional Changed")
                litigation_pending = st.checkbox("Pending Litigation by Creditors")
            else:
                no_of_financial_creditors = 1
                resolution_applicants_received = 1
                ip_changed = False
                litigation_pending = False

        submitted = st.form_submit_button("🚀 Generate Risk Assessment", type="primary", use_container_width=True)

    # ── Results ──────────────────────────────────────────────────────────────
    if submitted:
        try:
            models = load_models()
            from src.predict import predict_case

            case_input = {
                "case_type": case_type,
                "court": court,
                "state": state,
                "sector": sector,
                "claim_amount_lakhs": claim_amount_lakhs,
                "filing_year": filing_year,
                "filing_quarter": filing_quarter,
                "case_age_months": case_age_months,
                "num_prior_adjournments": num_prior_adjournments,
                "has_interim_order": has_interim_order,
                "claimant_lawyer_win_rate": claimant_lawyer_win_rate,
                "represented_by_senior_counsel": represented_by_senior_counsel,
                "respondent_is_govt": respondent_is_govt,
                "respondent_is_psu": respondent_is_psu,
                "no_of_financial_creditors": no_of_financial_creditors,
                "resolution_applicants_received": resolution_applicants_received,
                "ip_changed": ip_changed,
                "litigation_pending": litigation_pending,
            }

            result = predict_case(case_input, models)
            st.divider()
            st.subheader("📊 Risk Assessment Results")

            # Recommendation banner
            rec = result["recommendation"]
            if "Strong" in rec:
                st.success(f"✅ {rec}")
            elif "Moderate" in rec:
                st.warning(f"⚠️ {rec}")
            else:
                st.error(f"❌ {rec}")

            # KPI cards
            k1, k2, k3, k4 = st.columns(4)
            dur = {"p10": result["duration_low"], "p50": result["duration_months"], "p90": result["duration_high"]}
            p_fav = result["outcome_prob"]
            risk = result["risk_score"]

            with k1:
                st.metric("Expected Duration (median)", f"{dur['p50']:.0f} months",
                          f"Range: {dur['p10']:.0f}–{dur['p90']:.0f} mo")
            with k2:
                st.metric("P(Favourable Outcome)", f"{p_fav*100:.1f}%",
                          "Higher is better for claimant")
            with k3:
                if result["realisation_pct"] > 0:
                    st.metric("Expected Recovery", f"{result['realisation_pct']:.1f}%",
                              f"Range: {result['realisation_low']:.1f}–{result['realisation_high']:.1f}%")
                else:
                    st.metric("Recovery Model", "N/A (non-monetary)")
            with k4:
                colour = "normal" if risk >= 50 else "inverse"
                st.metric("Composite Risk Score", f"{risk:.0f} / 100",
                          "↑ higher = lower risk", delta_color=colour)

            st.divider()
            col_a, col_b = st.columns(2)

            if "cbr" in result and result["cbr"]["similar_cases"]:
                st.divider()
                st.subheader("Similar Precedents")

                # Blend commentary
                blend = blend_summary(result, result["cbr"]["adapted"])
                if blend:
                    st.info(blend)

                # Natural language summary
                mode = "ibc" if "IBC" in case_input.get("case_type","") else "njdg"
                st.markdown(summarise_precedents(result["cbr"]["similar_cases"], mode=mode))

                # Detailed precedent cards
                with st.expander("View individual precedent cases"):
                    for c in result["cbr"]["similar_cases"]:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Similarity",    f"{c.similarity:.0%}")
                        col2.metric("Duration",      f"{c.duration_months:.0f} mo" if c.duration_months else "—")
                        col3.metric("Outcome",       "Favourable" if c.favourable == 1 else "Unfavourable" if c.favourable == 0 else "—")
                        col4.metric("Recovery",      f"{c.realisation_pct:.1f}%" if c.realisation_pct else "—")
                        st.caption(f"Case ID: {c.case_id}  |  Court: {c.court or '—'}  |  Type: {c.case_type or '—'}  |  Filed: {c.filing_year or '—'}")
                        st.divider()

            # Duration confidence chart
            with col_a:
                st.subheader("⏱ Duration Distribution")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["P10 (Optimistic)", "P50 (Median)", "P90 (Pessimistic)"],
                    y=[dur["p10"], dur["p50"], dur["p90"]],
                    marker_color=["#2ecc71", "#3498db", "#e74c3c"],
                    text=[f"{v:.0f} mo" for v in [dur["p10"], dur["p50"], dur["p90"]]],
                    textposition="outside",
                ))
                fig.update_layout(yaxis_title="Months to Disposal", height=320,
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            # Outcome probability gauge
            with col_b:
                st.subheader("🎯 Outcome Probability")
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=p_fav * 100,
                    delta={"reference": 50},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#3498db"},
                        "steps": [
                            {"range": [0, 40], "color": "#e74c3c"},
                            {"range": [40, 60], "color": "#f39c12"},
                            {"range": [60, 100], "color": "#2ecc71"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                    title={"text": "P(Favourable) %"},
                ))
                fig2.update_layout(height=320)
                st.plotly_chart(fig2, use_container_width=True)

            # Realisation chart (if applicable)
            if result["realisation_pct"] > 0:
                st.subheader("💰 Recovery / Realisation Range")
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=["P10 (Pessimistic)", "P50 (Median)", "P90 (Optimistic)"],
                    y=[result["realisation_low"], result["realisation_pct"], result["realisation_high"]],
                    marker_color=["#e74c3c", "#3498db", "#2ecc71"],
                    text=[f"{v:.1f}%" for v in [result["realisation_low"], result["realisation_pct"], result["realisation_high"]]],
                    textposition="outside",
                ))
                fig3.update_layout(yaxis_title="Recovery %", yaxis_range=[0, 105],
                    height=300, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)

            # Raw output expander
            with st.expander("🔎 Raw Model Output (JSON)"):
                st.json(result)

            # PDF report download
            pdf_bytes = generate_assessment_report(
                case_inputs=case_input,
                predictions=result,
            )
            st.download_button(
                label="📄 Download Assessment Report (PDF)",
                data=pdf_bytes,
                file_name=f"ILFRA_Assessment_{case_type.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

        except FileNotFoundError as e:
            st.error(f"Models not found. Please run the training pipeline first.\n\n`{e}`")
            st.code("cd litfin && python src/data_ingestion.py && python src/feature_engineering.py && python src/train.py")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Model Insights & Feature Importance")

    try:
        fi_data = load_feature_importance()
        metrics_path = MODELS_DIR / "training_metrics.csv"

        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path, index_col=0)
            st.subheader("Training Metrics")
            st.dataframe(metrics_df.style.format("{:.3f}", na_rep="—"), use_container_width=True)
            st.divider()

        for model_name, fi_df in fi_data.items():
            st.subheader(f"{model_name} Model — Top Features")
            top = fi_df.head(12)
            fig = px.bar(
                top.reset_index(), x="importance", y="index",
                orientation="h", color="importance",
                color_continuous_scale="Blues",
                labels={"index": "Feature", "importance": "Importance Score"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380,
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Calibration reliability diagram ───────────────────────────────
        # Replace the single cal_path check with:
        raw_path = MODELS_DIR / "calibration_curve_raw.csv"
        cal_path = MODELS_DIR / "calibration_curve_cal.csv"

        if raw_path.exists() and cal_path.exists():
            st.divider()
            st.subheader("Outcome Model — Calibration Reliability Diagram")
            raw_df = pd.read_csv(raw_path)
            cal_df = pd.read_csv(cal_path)

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines", name="Perfect calibration",
                line=dict(dash="dash", color="gray", width=1)
            ))
            fig_cal.add_trace(go.Scatter(
                x=raw_df["mean_predicted"], y=raw_df["fraction_positive"],
                mode="lines+markers", name="Before calibration",
                line=dict(color="#E24B4A")
            ))
            fig_cal.add_trace(go.Scatter(
                x=cal_df["mean_predicted"], y=cal_df["fraction_positive"],
                mode="lines+markers", name="After calibration",
                line=dict(color="#1D9E75")
            ))
            fig_cal.update_layout(
                xaxis_title="Mean predicted probability",
                yaxis_title="Fraction of positive outcomes",
                height=380, plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.caption(
                "A perfectly calibrated model follows the dashed diagonal. "
                "Points above = model underestimates; below = overestimates."
            )

    except Exception:
        st.info("Train the models first to see feature importance charts here.\n\n"
                "Run: `python src/train.py`")
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📖 How the Risk Assessor Works")
    st.markdown("""
### Architecture Overview

```
Public Data Sources
  ├── NJDG CSV exports        ─┐
  ├── IBBI CIRP Excel          ├── data_ingestion.py → data/raw/
  └── eCourts judgment PDFs  ─┘

Feature Engineering (feature_engineering.py)
  ├── Categorical encoding (case type, court, state, sector)
  ├── Financial features (log claim amount, claim bucket)
  ├── Process signals (adjournments, interim orders)
  ├── Party/counsel signals (lawyer win rate, govt respondent)
  └── Aggregate stats (historical court disposal rates)

Model Training (train.py)
  ├── Duration model     → LightGBM Regressor (+ quantile P10/P90)
  ├── Outcome model      → LightGBM Classifier (AUC scored)
  └── Realisation model  → LightGBM Regressor (+ quantile P10/P90)

Streamlit App (this interface)
  └── Prediction API (predict.py) → structured risk report
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LightGBM over XGBoost | Faster on tabular data, handles categoricals natively |
| Quantile regression for CIs | Distribution-free, no normality assumption required |
| Separate IBC model | IBC proceedings have fundamentally different dynamics than civil suits |
| Log-transform on claim amount | Claim amounts span 4+ orders of magnitude |
| Adjournment density feature | Raw count without normalising for case age is misleading |

### Ethical Guardrails
- All training data is from **public government sources** only
- Model outputs are **advisory signals**, not legal opinions
- **No individual litigant data** is stored or processed
- Feature importance reports provide **full transparency**
- The tool explicitly recommends seeking qualified legal review

### Limitations
- Trained on public metadata — cannot access sealed records or private settlement terms
- Lawyer win rates require manual input; no automated lookup yet
- Predictions degrade for highly novel case types not well-represented in training data
- Model should be retrained quarterly as new NJDG data becomes available
    """)