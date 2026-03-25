"""
src/report_generator.py
-----------------------
Generates a professional PDF assessment report for ILFRA.

Usage (from Streamlit or predict.py):
    from report_generator import generate_assessment_report

    pdf_bytes = generate_assessment_report(
        case_inputs=case_inputs_dict,
        predictions=predictions_dict,
        risk_factors=risk_factors_list,   # optional — SHAP values when available
    )

    st.download_button("Download Report", pdf_bytes,
                       file_name="ILFRA_Assessment.pdf",
                       mime="application/pdf")

Parameters
----------
case_inputs : dict
    Raw inputs provided by the user / analyst. Expected keys mirror the
    Streamlit form fields. Unknown keys are shown as-is.

predictions : dict
    Model outputs. Expected structure:
    {
        "duration_months":    float,
        "duration_low":       float,          # 10th pct or CI lower
        "duration_high":      float,          # 90th pct or CI upper
        "outcome_prob":       float,          # 0-1 probability of favourable outcome
        "outcome_label":      str,            # "Favourable" / "Unfavourable"
        "realisation_pct":    float,          # point estimate
        "realisation_low":    float,          # lower CI
        "realisation_high":   float,          # upper CI
        "risk_score":         float,          # 0-100 composite (optional)
        "data_source":        str,            # "NJDG" / "IBC"
    }

risk_factors : list[dict], optional
    Each entry: {"feature": str, "impact": str, "direction": str}
    direction ∈ {"positive", "negative", "neutral"}
    Populated by SHAP later; stubbed list shown if not provided.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand colours ─────────────────────────────────────────────────────────────
NAVY     = colors.HexColor("#0D1B2A")
TEAL     = colors.HexColor("#1A7A8A")
GOLD     = colors.HexColor("#C9982A")
LIGHT_BG = colors.HexColor("#F4F6F9")
MID_GREY = colors.HexColor("#6B7280")
RED      = colors.HexColor("#C0392B")
GREEN    = colors.HexColor("#1A7A4A")
AMBER    = colors.HexColor("#C9982A")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ── Helper: horizontal gauge bar drawn as a Table cell ───────────────────────

def _gauge_table(value: float, low: float = 0.0, high: float = 100.0,
                 colour: colors.Color = TEAL, width: float = 120) -> Table:
    """Returns a single-row Table that visually represents a filled bar."""
    pct = min(max((value - low) / max(high - low, 1), 0.0), 1.0)
    filled  = width * pct
    empty   = width - filled
    data = [[""]]
    col_widths = [filled, empty] if filled > 0 else [0.5, width - 0.5]
    t = Table(data, colWidths=col_widths, rowHeights=[8])
    style = [
        ("BACKGROUND", (0, 0), (0, 0), colour),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E5E7EB")),
        ("LINEABOVE",  (0, 0), (-1, -1), 0, colours.white if False else colour),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]
    t.setStyle(TableStyle(style))
    return t


def _outcome_colour(prob: float) -> colors.Color:
    if prob >= 0.60:
        return GREEN
    if prob >= 0.40:
        return AMBER
    return RED


def _risk_score_colour(score: float) -> colors.Color:
    """Lower score = lower risk = greener."""
    if score <= 33:
        return GREEN
    if score <= 66:
        return AMBER
    return RED


# ── Style factory ─────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    s = {}

    s["title"] = ParagraphStyle(
        "ILFRATitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=NAVY,
        spaceAfter=2,
        alignment=TA_LEFT,
    )
    s["subtitle"] = ParagraphStyle(
        "ILFRASubtitle",
        fontName="Helvetica",
        fontSize=10,
        textColor=MID_GREY,
        spaceAfter=2,
        alignment=TA_LEFT,
    )
    s["section_head"] = ParagraphStyle(
        "SectionHead",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=4,
    )
    s["label"] = ParagraphStyle(
        "Label",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=MID_GREY,
        spaceAfter=1,
    )
    s["value"] = ParagraphStyle(
        "Value",
        fontName="Helvetica",
        fontSize=10,
        textColor=NAVY,
        spaceAfter=4,
    )
    s["big_metric"] = ParagraphStyle(
        "BigMetric",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=NAVY,
        spaceAfter=0,
        alignment=TA_CENTER,
    )
    s["metric_label"] = ParagraphStyle(
        "MetricLabel",
        fontName="Helvetica",
        fontSize=8,
        textColor=MID_GREY,
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    s["metric_range"] = ParagraphStyle(
        "MetricRange",
        fontName="Helvetica",
        fontSize=9,
        textColor=MID_GREY,
        spaceAfter=0,
        alignment=TA_CENTER,
    )
    s["disclaimer"] = ParagraphStyle(
        "Disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        textColor=MID_GREY,
        spaceBefore=4,
        spaceAfter=2,
        leading=11,
    )
    s["risk_pos"] = ParagraphStyle(
        "RiskPos", fontName="Helvetica", fontSize=9,
        textColor=GREEN, spaceAfter=2,
    )
    s["risk_neg"] = ParagraphStyle(
        "RiskNeg", fontName="Helvetica", fontSize=9,
        textColor=RED, spaceAfter=2,
    )
    s["risk_neu"] = ParagraphStyle(
        "RiskNeu", fontName="Helvetica", fontSize=9,
        textColor=MID_GREY, spaceAfter=2,
    )
    s["footer"] = ParagraphStyle(
        "Footer", fontName="Helvetica", fontSize=7,
        textColor=MID_GREY, alignment=TA_CENTER,
    )
    return s


# ── Section builders ──────────────────────────────────────────────────────────

def _header_block(styles: dict, ref: str, generated_at: str) -> list:
    elems = []

    # Logo-style wordmark row
    header_data = [
        [
            Paragraph("<b>ILFRA</b>", ParagraphStyle(
                "Logo", fontName="Helvetica-Bold", fontSize=26,
                textColor=TEAL,
            )),
            Paragraph(
                f"<font color='#6B7280'>Ref: {ref}</font><br/>"
                f"<font color='#6B7280'>Generated: {generated_at}</font>",
                ParagraphStyle("HeaderRight", fontName="Helvetica",
                               fontSize=9, textColor=MID_GREY,
                               alignment=TA_RIGHT),
            ),
        ]
    ]
    t = Table(header_data, colWidths=[PAGE_W - 2 * MARGIN - 80, 80])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    elems.append(t)
    elems.append(Paragraph(
        "Indian Litigation Finance Risk Assessor",
        ParagraphStyle("TagLine", fontName="Helvetica", fontSize=9,
                       textColor=MID_GREY),
    ))
    elems.append(HRFlowable(width="100%", thickness=2, color=TEAL,
                             spaceAfter=8))
    return elems


def _case_details_block(styles: dict, case_inputs: dict) -> list:
    elems = []
    elems.append(Paragraph("Case Details Submitted", styles["section_head"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E5E7EB"),
                             spaceAfter=6))

    # Build a clean 2-column key-value table
    DISPLAY_MAP = {
        "case_type":                   "Case Type",
        "court":                       "Court",
        "state":                       "State",
        "sector":                      "Sector",
        "claim_amount_lakhs":          "Claim Amount (₹ lakhs)",
        "claimant_lawyer_win_rate":    "Claimant Lawyer Win Rate",
        "num_prior_adjournments":      "Prior Adjournments",
        "has_interim_order":           "Interim Order Obtained",
        "represented_by_senior_counsel": "Senior Counsel",
        "respondent_is_govt":          "Respondent is Government",
        "respondent_is_psu":           "Respondent is PSU",
        "filing_date":                 "Filing Date",
        # IBC fields
        "bench":                       "NCLT Bench",
        "admitted_claim_cr":           "Admitted Claim (₹ Cr)",
        "no_of_financial_creditors":   "No. of Financial Creditors",
        "resolution_applicants_received": "Resolution Applicants",
        "resolution_status":           "Resolution Status",
        "ip_changed":                  "IP Changed",
        "litigation_pending":          "Litigation Pending",
    }

    rows = []
    for key, label in DISPLAY_MAP.items():
        val = case_inputs.get(key)
        if val is None:
            continue
        if isinstance(val, bool) or (isinstance(val, int) and key not in
                                     ("num_prior_adjournments", "no_of_financial_creditors",
                                      "resolution_applicants_received")):
            val = "Yes" if val else "No"
        if isinstance(val, float):
            if "rate" in key or "pct" in key:
                val = f"{val:.1%}"
            else:
                val = f"{val:,.2f}"
        rows.append([
            Paragraph(label, styles["label"]),
            Paragraph(str(val), styles["value"]),
        ])

    col_w = (PAGE_W - 2 * MARGIN) / 2 - 4
    # Split into two columns
    left_rows  = rows[:len(rows)//2 + len(rows) % 2]
    right_rows = rows[len(rows)//2 + len(rows) % 2:]

    max_r = max(len(left_rows), len(right_rows))
    empty = [Paragraph("", styles["label"]), Paragraph("", styles["value"])]
    while len(left_rows) < max_r:  left_rows.append(empty)
    while len(right_rows) < max_r: right_rows.append(empty)

    combined = []
    for l, r in zip(left_rows, right_rows):
        combined.append(l + [Spacer(8, 1)] + r)

    if combined:
        t = Table(combined, colWidths=[col_w * 0.38, col_w * 0.62,
                                        8, col_w * 0.38, col_w * 0.62])
        t.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1),
             [colors.white, LIGHT_BG]),
        ]))
        elems.append(t)

    return elems


def _predictions_block(styles: dict, preds: dict) -> list:
    elems = []
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Model Predictions", styles["section_head"]))
    elems.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#E5E7EB"), spaceAfter=6))

    dur   = preds.get("duration_months", 0)
    dur_l = preds.get("duration_low",   dur * 0.7)
    dur_h = preds.get("duration_high",  dur * 1.4)

    prob   = preds.get("outcome_prob", 0.5)
    o_label = preds.get("outcome_label", "Favourable" if prob >= 0.5 else "Unfavourable")

    real   = preds.get("realisation_pct", 0)
    real_l = preds.get("realisation_low",  max(real - 15, 0))
    real_h = preds.get("realisation_high", min(real + 15, 100))

    risk   = preds.get("risk_score", None)

    oc = _outcome_colour(prob)

    # Three metric cards side-by-side
    dur_cell = [
        Paragraph(f"{dur:.0f} months", styles["big_metric"]),
        Paragraph("Estimated Duration", styles["metric_label"]),
        Paragraph(f"Range: {dur_l:.0f} – {dur_h:.0f} months", styles["metric_range"]),
    ]

    out_cell = [
        Paragraph(
            f'<b>{prob:.0%}</b>',
            ParagraphStyle("BigMetricColoured", fontName="Helvetica-Bold",
                           fontSize=22, textColor=oc, spaceAfter=0,
                           alignment=TA_CENTER),
        ),
        Paragraph("Outcome Probability", styles["metric_label"]),
        Paragraph(f"Prediction: {o_label}", styles["metric_range"]),
    ]

    real_cell = [
        Paragraph(f"{real:.1f}%", styles["big_metric"]),
        Paragraph("Estimated Realisation", styles["metric_label"]),
        Paragraph(f"Range: {real_l:.1f}% – {real_h:.1f}%", styles["metric_range"]),
    ]

    card_w = (PAGE_W - 2 * MARGIN) / 3 - 4
    t = Table(
        [[dur_cell, out_cell, real_cell]],
        colWidths=[card_w, card_w, card_w],
        rowHeights=None,
    )
    t.setStyle(TableStyle([
        ("BOX",           (0, 0), (0, 0), 1, colors.HexColor("#E5E7EB")),
        ("BOX",           (1, 0), (1, 0), 1, colors.HexColor("#E5E7EB")),
        ("BOX",           (2, 0), (2, 0), 1, colors.HexColor("#E5E7EB")),
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 8))

    # Outcome probability bar
    elems.append(Paragraph("Outcome Probability", styles["label"]))
    bar_w = PAGE_W - 2 * MARGIN
    bar_filled = bar_w * prob
    bar_empty  = bar_w - bar_filled

    bar_data = [[""]]
    bar_cols  = [bar_filled, bar_empty] if bar_filled > 1 else [1, bar_w - 1]
    bar_t = Table(bar_data, colWidths=bar_cols, rowHeights=[12])
    bar_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), oc),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E5E7EB")),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    elems.append(bar_t)
    elems.append(Spacer(1, 10))

    # Optional composite risk score
    if risk is not None:
        rc = _risk_score_colour(risk)
        risk_label = "Low Risk" if risk <= 33 else ("Moderate Risk" if risk <= 66 else "High Risk")
        elems.append(Paragraph("Composite Risk Score", styles["label"]))
        risk_filled = bar_w * (risk / 100)
        risk_empty  = bar_w - risk_filled
        r_bar = Table([[""]], colWidths=[risk_filled if risk_filled > 1 else 1,
                                          risk_empty if risk_empty > 1 else 1],
                       rowHeights=[12])
        r_bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), rc),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E5E7EB")),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))
        elems.append(r_bar)
        elems.append(Paragraph(
            f"Score: {risk:.0f}/100 — {risk_label}",
            ParagraphStyle("RiskScoreLabel", fontName="Helvetica",
                           fontSize=8, textColor=rc, spaceAfter=4),
        ))

    return elems


def _risk_factors_block(styles: dict, risk_factors: list[dict]) -> list:
    elems = []
    elems.append(Paragraph("Key Risk Factors", styles["section_head"]))
    elems.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#E5E7EB"), spaceAfter=6))

    STUB_NOTE = False
    if not risk_factors:
        STUB_NOTE = True
        risk_factors = [
            {"feature": "Claimant lawyer win rate", "impact": "Stronger win rate history increases favourable outcome probability", "direction": "positive"},
            {"feature": "Prior adjournments",        "impact": "Higher adjournment count is associated with longer duration", "direction": "negative"},
            {"feature": "Interim order obtained",    "impact": "Presence of interim order associated with better realisation", "direction": "positive"},
            {"feature": "Court average duration",    "impact": "Court-level historical disposal speed influences estimates", "direction": "neutral"},
            {"feature": "Claim amount (log-scaled)", "impact": "Larger claims tend to face longer pendency", "direction": "negative"},
        ]

    rows = []
    for f in risk_factors:
        direction = f.get("direction", "neutral")
        arrow = "▲" if direction == "positive" else ("▼" if direction == "negative" else "●")
        style_key = {"positive": "risk_pos", "negative": "risk_neg"}.get(direction, "risk_neu")
        rows.append([
            Paragraph(f"{arrow} {f['feature']}", styles[style_key]),
            Paragraph(f['impact'], styles["value"]),
        ])

    col_w = PAGE_W - 2 * MARGIN
    t = Table(rows, colWidths=[col_w * 0.28, col_w * 0.72])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.white, LIGHT_BG]),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.25, colors.HexColor("#E5E7EB")),
    ]))
    elems.append(t)

    if STUB_NOTE:
        elems.append(Spacer(1, 4))
        elems.append(Paragraph(
            "Note: Feature importances above are indicative domain-level factors. "
            "Case-specific SHAP explanations will be available in a future release.",
            styles["disclaimer"],
        ))

    return elems


def _disclaimer_block(styles: dict, data_source: str) -> list:
    elems = []
    elems.append(Spacer(1, 10))
    elems.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=6))
    elems.append(Paragraph("Important Disclaimer", styles["section_head"]))

    text = (
        "This report has been generated by the Indian Litigation Finance Risk Assessor (ILFRA), "
        "a machine learning system trained on publicly available court case metadata "
        f"({'NJDG case data' if data_source == 'NJDG' else 'IBBI/IBC resolution data'}). "
        "All predictions are probabilistic estimates based on statistical patterns in historical data "
        "and do not constitute legal advice, a legal opinion, or a guarantee of outcome. "
        "The model cannot account for case-specific facts, judicial discretion, witness credibility, "
        "or qualitative legal arguments. This output is intended solely as a supplementary data point "
        "to support — and not replace — independent legal due diligence by qualified lawyers. "
        "Litigation funding decisions should not be made solely on the basis of this report. "
        "ILFRA and its developers accept no liability for funding decisions made in reliance on these estimates."
    )
    elems.append(Paragraph(text, styles["disclaimer"]))
    return elems


def _footer(styles: dict, ref: str, page: int, total: int) -> list:
    text = f"ILFRA Confidential Assessment | {ref} | Page {page} of {total}"
    return [Paragraph(text, styles["footer"])]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_assessment_report(
    case_inputs: dict,
    predictions: dict,
    risk_factors: Optional[list[dict]] = None,
    reference: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF assessment report and return it as bytes.

    Parameters
    ----------
    case_inputs  : dict  — user-submitted form fields
    predictions  : dict  — model output (see module docstring for schema)
    risk_factors : list  — optional SHAP-driven factor list
    reference    : str   — optional case reference string; auto-generated if None

    Returns
    -------
    bytes — raw PDF content, ready for st.download_button or file I/O
    """
    buf = io.BytesIO()

    ref = reference or f"ILFRA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    generated_at = datetime.now().strftime("%d %B %Y, %H:%M IST")
    data_source = predictions.get("data_source", "NJDG")

    styles = _build_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"ILFRA Assessment — {ref}",
        author="ILFRA",
        subject="Litigation Finance Risk Assessment",
    )

    story = []
    story += _header_block(styles, ref, generated_at)
    story += _case_details_block(styles, case_inputs)
    story += _predictions_block(styles, predictions)
    story += _risk_factors_block(styles, risk_factors or [])
    story += _disclaimer_block(styles, data_source)

    doc.build(story)
    return buf.getvalue()


# ── Predictions schema validation ────────────────────────────────────────────

REQUIRED_PREDICTION_KEYS = {
    "duration_months":  (int, float),
    "duration_low":     (int, float),
    "duration_high":    (int, float),
    "outcome_prob":     (int, float),
    "outcome_label":    (str,),
    "realisation_pct":  (int, float),
    "realisation_low":  (int, float),
    "realisation_high": (int, float),
    "data_source":      (str,),
}

OPTIONAL_PREDICTION_KEYS = {
    "risk_score": (int, float),   # composite 0-100; omit if not computed
}


def validate_predictions(predictions: dict) -> list[str]:
    """
    Validates the predictions dict against the expected schema.
    Returns a list of error strings — empty list means valid.

    Call this in predict.py before passing to generate_assessment_report()
    to catch missing or wrongly-typed keys early.

    Example
    -------
    errors = validate_predictions(preds)
    if errors:
        raise ValueError("Invalid predictions dict:\\n" + "\\n".join(errors))
    """
    errors = []
    for key, expected_types in REQUIRED_PREDICTION_KEYS.items():
        if key not in predictions:
            errors.append(f"Missing required key: '{key}'")
        elif not isinstance(predictions[key], expected_types):
            errors.append(
                f"'{key}' must be {expected_types}, "
                f"got {type(predictions[key]).__name__}"
            )
    for key, expected_types in OPTIONAL_PREDICTION_KEYS.items():
        if key in predictions and not isinstance(predictions[key], expected_types):
            errors.append(
                f"'{key}' must be {expected_types}, "
                f"got {type(predictions[key]).__name__}"
            )
    if "outcome_prob" in predictions:
        p = predictions["outcome_prob"]
        if isinstance(p, (int, float)) and not (0.0 <= p <= 1.0):
            errors.append(f"'outcome_prob' must be between 0 and 1, got {p}")
    if "data_source" in predictions:
        if predictions["data_source"] not in ("NJDG", "IBC"):
            errors.append(
                f"'data_source' must be 'NJDG' or 'IBC', "
                f"got '{predictions['data_source']}'"
            )
    return errors


if __name__ == "__main__":
    print("Required prediction keys:")
    for k, t in REQUIRED_PREDICTION_KEYS.items():
        print(f"  {k:25s} {[x.__name__ for x in t]}")
    print("\nOptional prediction keys:")
    for k, t in OPTIONAL_PREDICTION_KEYS.items():
        print(f"  {k:25s} {[x.__name__ for x in t]}")