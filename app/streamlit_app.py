"""
app/streamlit_app.py — ILFRA Streamlit dashboard.

Tab 1 — Case Assessment   : form, ML predictions, per-case SHAP waterfall,
                             CBR similar precedents
Tab 2 — Model Insights    : training metrics, SHAP global summary plots
                             (replaces LightGBM feature importance),
                             calibration reliability diagram
Tab 3 — How It Works      : architecture overview and ethical guardrails
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.predict import load_models, predict_case

MODELS_DIR = Path(__file__).parent.parent / "models"

st.set_page_config(
    page_title="ILFRA — Litigation Finance Risk Assessor",
    page_icon="⚖️",
    layout="wide",
)


# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def get_models():
    return load_models()


# ── SHAP chart helpers ────────────────────────────────────────────────────────

def _shap_waterfall_chart(shap_map: dict, title: str,
                           top_n: int = 12) -> go.Figure:
    """
    Renders a horizontal bar chart of per-case SHAP values.
    Positive values (red) push the prediction higher.
    Negative values (blue) push the prediction lower.
    Only the top_n features by absolute magnitude are shown.
    """
    items = sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features = [k for k, _ in items]
    values   = [v for _, v in items]
    colors   = ["#E24B4A" if v >= 0 else "#3B82F6" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (impact on model output)",
        yaxis={"categoryorder": "total ascending"},
        height=max(300, top_n * 32),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=60, t=40, b=10),
    )
    return fig


def _shap_summary_chart(shap_csv_path: Path, title: str,
                         top_n: int = 15) -> go.Figure | None:
    """
    Loads the global mean-absolute-SHAP CSV saved by train.py and
    renders a horizontal bar chart for the Model Insights tab.
    Returns None if the file doesn't exist yet.
    """
    if not shap_csv_path.exists():
        return None

    df = pd.read_csv(shap_csv_path, index_col=0)
    df.columns = ["mean_abs_shap"]
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n)

    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"],
        y=df.index,
        orientation="h",
        marker_color="#6366F1",
        text=[f"{v:.4f}" for v in df["mean_abs_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP value| (average impact on model output)",
        yaxis={"categoryorder": "total ascending"},
        height=max(300, top_n * 32),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=60, t=40, b=10),
    )
    return fig


# ── Page layout ───────────────────────────────────────────────────────────────

st.title("⚖️ ILFRA — Indian Litigation Finance Risk Assessor")
st.caption("ML-powered advisory tool for evaluating civil and commercial litigation risk in India.")

tab1, tab2, tab3 = st.tabs(["📋 Case Assessment", "📊 Model Insights", "ℹ️ How It Works"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Case Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Case Assessment")

    with st.form("case_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Case Identity")
            case_type = st.selectbox("Case Type", [
                "Civil Suit", "Money Recovery", "Commercial Dispute",
                "Writ Petition", "CIRP (IBC)", "Liquidation (IBC)",
                "Arbitration", "Injunction", "Other"
            ])
            court = st.selectbox("Court", [
                "District Court", "High Court", "Supreme Court",
                "Commercial Court", "City Civil Court", "Magistrate Court"
            ])
            state = st.selectbox("State", [
                "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu",
                "West Bengal", "Gujarat", "Rajasthan", "Uttar Pradesh",
                "Andhra Pradesh", "Telangana", "Other"
            ])
            sector = st.selectbox("Sector", [
                "Others", "Real Estate", "Manufacturing", "Banking & Finance",
                "IT / Technology", "Energy", "Retail", "Healthcare",
                "Infrastructure", "Agriculture"
            ])

        with col2:
            st.subheader("Timeline & Process")
            filing_year    = st.number_input("Filing Year",    2010, 2025, 2022)
            filing_quarter = st.selectbox("Filing Quarter",    [1, 2, 3, 4])
            case_age       = st.number_input("Case Age (months)", 0, 240, 18)
            num_adjourn    = st.number_input("Prior Adjournments", 0, 100, 5)
            has_interim    = st.checkbox("Has Interim Order")
            senior_counsel = st.checkbox("Senior Counsel Appearing")

        with col3:
            st.subheader("Financial & Party")
            claim_amount       = st.number_input("Claim Amount (₹ Lakhs)", 0.1, 100000.0, 50.0)
            lawyer_win_rate    = st.slider("Claimant Lawyer Win Rate", 0.0, 1.0, 0.5, 0.05)
            respondent_is_govt = st.checkbox("Respondent is Government")
            respondent_is_psu  = st.checkbox("Respondent is PSU")

            st.subheader("IBC / Money Recovery")
            num_creditors  = st.number_input("No. of Financial Creditors",     1, 500,  1)
            num_applicants = st.number_input("Resolution Applicants Received", 0, 50,   0)
            ip_changed     = st.checkbox("IP Changed During CIRP")
            lit_pending    = st.checkbox("Litigation Pending")

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        case_input = {
            "case_type":                    case_type,
            "court":                        court,
            "state":                        state,
            "sector":                       sector,
            "filing_year":                  filing_year,
            "filing_quarter":               filing_quarter,
            "case_age_months":              case_age,
            "num_prior_adjournments":       num_adjourn,
            "has_interim_order":            has_interim,
            "represented_by_senior_counsel": senior_counsel,
            "claim_amount_lakhs":           claim_amount,
            "claimant_lawyer_win_rate":     lawyer_win_rate,
            "respondent_is_govt":           respondent_is_govt,
            "respondent_is_psu":            respondent_is_psu,
            "no_of_financial_creditors":    num_creditors,
            "resolution_applicants_received": num_applicants,
            "ip_changed":                   ip_changed,
            "litigation_pending":           lit_pending,
        }

        with st.spinner("Running assessment..."):
            try:
                models = get_models()
                result = predict_case(case_input, models)
            except Exception as e:
                st.error(f"Assessment failed: {e}")
                st.stop()

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Assessment Results")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Risk Score",       f"{result['risk_score']} / 100")
        k2.metric("Recommendation",    result["recommendation"])
        k3.metric("Outcome Probability",
                  f"{result['outcome_prob']:.0%}",
                  delta=result["outcome_label"])
        k4.metric("Expected Duration",
                  f"{result['duration_months']} mo",
                  delta=f"{result['duration_low']}–{result['duration_high']} mo range")

        if result["realisation_pct"] > 0:
            r1, r2, r3 = st.columns(3)
            r1.metric("Realisation (P50)", f"{result['realisation_pct']}%")
            r2.metric("Realisation (P10)", f"{result['realisation_low']}%")
            r3.metric("Realisation (P90)", f"{result['realisation_high']}%")

        # ── Duration and outcome charts ────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            fig_dur = go.Figure(go.Bar(
                x=["Optimistic (P10)", "Median (P50)", "Pessimistic (P90)"],
                y=[result["duration_low"], result["duration_months"], result["duration_high"]],
                marker_color=["#1D9E75", "#3B82F6", "#E24B4A"],
            ))
            fig_dur.update_layout(
                title="Duration Estimate (months)",
                yaxis_title="Months",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
            )
            st.plotly_chart(fig_dur, use_container_width=True)

        with c2:
            fig_out = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["outcome_prob"] * 100,
                title={"text": "P(Favourable Outcome) %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#1D9E75"},
                    "steps": [
                        {"range": [0,  40], "color": "#FEE2E2"},
                        {"range": [40, 65], "color": "#FEF3C7"},
                        {"range": [65, 100],"color": "#D1FAE5"},
                    ],
                },
            ))
            fig_out.update_layout(height=280)
            st.plotly_chart(fig_out, use_container_width=True)

        # ── Per-case SHAP explanations ─────────────────────────────────────────
        shap = result.get("shap", {})
        has_shap = any(v is not None for v in shap.values())

        if has_shap:
            st.divider()
            st.subheader("🔍 Why This Prediction? — Feature Contributions (SHAP)")
            st.caption(
                "Each bar shows how much a feature pushed the prediction "
                "**up** (red) or **down** (blue) from the model's baseline. "
                "Longer bars = stronger influence on this specific case."
            )

            shap_tabs = []
            shap_data = []
            if shap.get("outcome"):
                shap_tabs.append("Outcome")
                shap_data.append(("outcome", "Outcome Model — Feature Contributions (this case)"))
            if shap.get("duration"):
                shap_tabs.append("Duration")
                shap_data.append(("duration", "Duration Model — Feature Contributions (this case)"))
            if shap.get("realisation"):
                shap_tabs.append("Realisation")
                shap_data.append(("realisation", "Realisation Model — Feature Contributions (this case)"))

            if shap_tabs:
                shap_tab_objects = st.tabs(shap_tabs)
                for tab_obj, (key, title) in zip(shap_tab_objects, shap_data):
                    with tab_obj:
                        fig = _shap_waterfall_chart(shap[key], title)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            f"Base value (model average) plus these contributions = "
                            f"the final prediction for this case."
                        )
        else:
            st.info(
                "SHAP explainers not found. Re-run `python src/train.py` "
                "after installing SHAP (`pip install shap`) to enable "
                "per-case feature contribution charts.",
                icon="ℹ️",
            )

        # ── CBR similar precedents ─────────────────────────────────────────────
        cbr = result.get("cbr", {})
        if cbr.get("similar_cases"):
            st.divider()
            st.subheader("📚 Similar Precedents")

            mode = "ibc" if "IBC" in case_input.get("case_type", "") else "njdg"
            try:
                from src.cbr_explainer import summarise_precedents, blend_summary
                blend = blend_summary(result, cbr.get("adapted", {}))
                if blend:
                    st.info(blend)
                st.markdown(summarise_precedents(cbr["similar_cases"], mode=mode))
            except Exception:
                pass

            with st.expander("View individual precedent cases"):
                for c in cbr["similar_cases"]:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Similarity",  f"{c.similarity:.0%}")
                    col2.metric("Duration",    f"{c.duration_months:.0f} mo"
                                              if c.duration_months else "—")
                    col3.metric("Outcome",     "Favourable"   if c.favourable == 1
                                              else "Unfavourable" if c.favourable == 0
                                              else "—")
                    col4.metric("Recovery",    f"{c.realisation_pct:.1f}%"
                                              if c.realisation_pct else "—")
                    st.caption(
                        f"Case ID: {c.case_id}  |  "
                        f"Court: {c.court or '—'}  |  "
                        f"Type: {c.case_type or '—'}  |  "
                        f"Filed: {c.filing_year or '—'}"
                    )
                    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Model Insights & Feature Importance")

    try:
        # ── Training metrics ───────────────────────────────────────────────────
        metrics_path = MODELS_DIR / "training_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path, index_col=0)
            st.subheader("Training Metrics")
            st.dataframe(
                metrics_df.style.format("{:.3f}", na_rep="—"),
                use_container_width=True,
            )
            st.divider()

        # ── SHAP global summary (preferred over LightGBM importance) ──────────
        shap_files = {
            "Outcome":      MODELS_DIR / "outcome_shap_values.csv",
            "Duration":     MODELS_DIR / "duration_shap_values.csv",
            "Realisation":  MODELS_DIR / "realisation_shap_values.csv",
        }

        shap_available = any(p.exists() for p in shap_files.values())

        if shap_available:
            st.subheader("Global Feature Importance — SHAP (Mean |SHAP Value|)")
            st.caption(
                "SHAP-based global importance measures each feature's average "
                "contribution to predictions across all training cases. "
                "This is more reliable than LightGBM's gain-based importance "
                "because it is measured in the model's output units and accounts "
                "for feature interactions."
            )
            for model_name, shap_path in shap_files.items():
                fig = _shap_summary_chart(
                    shap_path,
                    f"{model_name} Model — Global Feature Importance (SHAP)"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()

        else:
            # ── Fallback: LightGBM native importance ───────────────────────────
            st.subheader("Feature Importance (LightGBM native)")
            st.caption(
                "SHAP explainers not found. Showing LightGBM gain-based importance. "
                "Re-run `python src/train.py` after `pip install shap` "
                "for more reliable SHAP-based importance."
            )
            fi_paths = {
                "Outcome":     MODELS_DIR / "outcome_feature_importance.csv",
                "Duration":    MODELS_DIR / "duration_feature_importance.csv",
                "Realisation": MODELS_DIR / "realisation_feature_importance.csv",
            }
            for model_name, fi_path in fi_paths.items():
                if not fi_path.exists():
                    continue
                fi_df = pd.read_csv(fi_path, index_col=0)
                fi_df.columns = ["importance"]
                top = fi_df.sort_values("importance", ascending=False).head(12)
                fig = px.bar(
                    top.reset_index(), x="importance", y="index",
                    orientation="h", color="importance",
                    color_continuous_scale="Blues",
                    labels={"index": "Feature", "importance": "Importance Score"},
                    title=f"{model_name} Model — Top Features",
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=380,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Calibration reliability diagram ───────────────────────────────────
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
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.caption(
                "A perfectly calibrated model follows the dashed diagonal. "
                "Points above = model underestimates; below = overestimates."
            )

    except Exception:
        st.info(
            "Train the models first to see insights here.\n\n"
            "Run: `python src/train.py`"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("ℹ️ How It Works")

    st.markdown("""
### Pipeline Overview

ILFRA combines three complementary approaches to litigation risk assessment:

**1. ML Prediction (LightGBM)**
Three LightGBM models are trained on historical case data from NJDG and IBBI CIRP:
- **Duration model** — predicts expected case length with P10/P50/P90 confidence intervals using quantile regression
- **Outcome classifier** — predicts probability of a favourable outcome, calibrated with isotonic regression
- **Realisation model** — predicts financial recovery percentage for IBC and Money Recovery cases

**2. SHAP Explainability**
Every prediction is accompanied by SHAP (SHapley Additive exPlanations) values that show exactly which features drove the prediction for this specific case. SHAP values are in the model's output units — positive values push the prediction higher, negative values lower.

**3. Case-Based Reasoning (CBR)**
The CBR engine retrieves the K most similar historical cases using weighted cosine similarity, where features are weighted by their domain importance. The retrieved cases provide concrete precedents with known outcomes, and their results are blended with the ML prediction via similarity-weighted averaging.

### Design Decisions

| Decision | Rationale |
|---|---|
| Quantile regression for duration | Gives statistically valid confidence intervals without distributional assumptions |
| Isotonic calibration over Platt scaling | Fewer assumptions about miscalibration shape; better for LightGBM outputs |
| SHAP TreeExplainer | Exact algorithm for tree models; computationally efficient; output in prediction units |
| Weighted cosine similarity for CBR | Handles mixed feature scales; domain-important features penalise mismatch more |
| Content-based IBBI sheet detection | Robust to IBBI's quarterly table renumbering without code changes |

### Ethical Guardrails

- ILFRA is an **advisory tool only** — not a substitute for qualified legal advice
- All predictions carry uncertainty and should be interpreted as probabilistic estimates
- SHAP values explain model behaviour but do not imply causal relationships
- CBR precedents are retrieved from available data and may not represent all relevant case types
- Funding and legal decisions must involve qualified legal professionals
    """)