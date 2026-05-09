"""
src/cbr_explainer.py
--------------------
Converts retrieved SimilarCase objects into natural language
precedent summaries for the Streamlit UI.
"""

from src.cbr_engine import SimilarCase


def summarise_precedents(cases: list[SimilarCase], mode: str = "njdg") -> str:
    """
    Generates a concise natural language summary of retrieved precedents.
    mode: "njdg" for civil/commercial, "ibc" for insolvency cases.
    """
    if not cases:
        return "No sufficiently similar precedents found in the case base."

    n = len(cases)
    avg_sim = sum(c.similarity for c in cases) / n

    if mode == "njdg":
        favourable_count = sum(1 for c in cases if c.favourable == 1)
        avg_duration = sum(c.duration_months for c in cases
                           if c.duration_months) / n

        lines = [
            f"**{n} similar precedents found** (avg. similarity: {avg_sim:.0%})",
            f"- **{favourable_count} of {n}** resolved favourably for the claimant",
            f"- Average resolution time: **{avg_duration:.0f} months**",
        ]

        courts = list({c.court for c in cases if c.court})
        if courts:
            lines.append(f"- Precedents from: {', '.join(courts[:3])}")

        years = [c.filing_year for c in cases if c.filing_year]
        if years:
            lines.append(f"- Filed between {min(years)} and {max(years)}")

    else:   # IBC
        avg_recovery = sum(c.realisation_pct for c in cases
                           if c.realisation_pct is not None) / n
        statuses = [c.resolution_status for c in cases if c.resolution_status]
        status_summary = ", ".join(
            f"{s} ({statuses.count(s)})" for s in sorted(set(statuses))
        ) if statuses else "—"

        lines = [
            f"**{n} similar CIRP precedents found** (avg. similarity: {avg_sim:.0%})",
            f"- Average realisation: **{avg_recovery:.1f}%** of admitted claims",
            f"- Resolution outcomes: {status_summary}",
        ]

    return "\n".join(lines)


def blend_summary(ml_pred: dict, cbr_adapted: dict) -> str:
    """
    Generates a one-paragraph blended interpretation
    combining ML model output with CBR-adapted estimates.
    """
    lines = []

    # Duration blending
    if "duration" in ml_pred and "cbr_duration_months" in cbr_adapted:
        ml_dur  = ml_pred["duration"]["p50"]
        cbr_dur = cbr_adapted["cbr_duration_months"]
        diff    = abs(ml_dur - cbr_dur)
        if diff < 6:
            lines.append(
                f"The ML model and precedent cases agree on duration: "
                f"both suggest approximately **{(ml_dur + cbr_dur) / 2:.0f} months**."
            )
        else:
            lines.append(
                f"The ML model predicts **{ml_dur:.0f} months**; "
                f"similar precedents averaged **{cbr_dur:.0f} months** — "
                f"treat the wider range as the realistic window."
            )

    # Outcome blending
    if "p_favourable" in ml_pred and "cbr_p_favourable" in cbr_adapted:
        ml_p  = ml_pred["p_favourable"]
        cbr_p = cbr_adapted["cbr_p_favourable"]
        diff  = abs(ml_p - cbr_p)
        if diff < 0.10:
            lines.append(
                f"Both the model ({ml_p:.0%}) and precedents ({cbr_p:.0%}) "
                f"align on outcome probability — high confidence signal."
            )
        else:
            lines.append(
                f"Outcome probability diverges: model says {ml_p:.0%}, "
                f"precedents show {cbr_p:.0%}. "
                f"Review precedent details carefully before committing."
            )

    return " ".join(lines) if lines else ""