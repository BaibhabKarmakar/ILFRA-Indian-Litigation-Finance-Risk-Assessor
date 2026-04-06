"""
src/genai/genai_utils.py
------------------------
Deterministic replacements for all GenAI utility functions.
No API calls. No cost. No external dependencies beyond rapidfuzz.

Install: pip install rapidfuzz
"""

from rapidfuzz import process, fuzz

# ── Canonical column vocabulary ───────────────────────────────────────────────

CANONICAL_COLUMNS = {
    "company_name",
    "cirp_start_date",
    "resolution_date",
    "cirp_initiated_by",
    "admitted_claim_cr",
    "liquidation_value",
    "fair_value",
    "realisable_amount",
    "realisation_pct",
    "resolution_status",
}

# Manual alias map — covers known IBBI header variations across quarters.
# Add new aliases here as you encounter them in real files.
_COLUMN_ALIASES = {
    # company name
    "name of cd":                        "company_name",
    "name of corporate debtor":          "company_name",
    "corporate debtor":                  "company_name",
    "company":                           "company_name",

    # cirp start date
    "date of commencement of cirp":      "cirp_start_date",
    "date of commencement":              "cirp_start_date",
    "cirp commencement date":            "cirp_start_date",
    "admission date":                    "cirp_start_date",
    "date of admission":                 "cirp_start_date",

    # resolution date
    "date of approval of resolution plan": "resolution_date",
    "date of approval":                  "resolution_date",
    "resolution plan approval date":     "resolution_date",
    "date of liquidation order":         "resolution_date",
    "date of dissolution":               "resolution_date",

    # initiated by
    "cirp initiated by":                 "cirp_initiated_by",
    "initiated by":                      "cirp_initiated_by",
    "applicant":                         "cirp_initiated_by",

    # admitted claims
    "total admitted claims":             "admitted_claim_cr",
    "amount of claims admitted":         "admitted_claim_cr",
    "admitted claims":                   "admitted_claim_cr",
    "claims admitted (rs crore)":        "admitted_claim_cr",
    "admitted claim":                    "admitted_claim_cr",

    # liquidation value
    "liquidation value":                 "liquidation_value",
    "liquidation value (rs crore)":      "liquidation_value",

    # fair value
    "fair value":                        "fair_value",
    "fair value (rs crore)":             "fair_value",

    # realisable amount
    "total realisable amount":           "realisable_amount",
    "realisable amount":                 "realisable_amount",
    "amount distributed":                "realisable_amount",
    "amount distributed to stakeholders": "realisable_amount",
    "amount realised":                   "realisable_amount",

    # realisation pct
    "realisable as % of admitted claims": "realisation_pct",
    "realisation %":                     "realisation_pct",
    "recovery rate":                     "realisation_pct",
    "realisable (% of admitted)":        "realisation_pct",

    # resolution status
    "status":                            "resolution_status",
    "current status":                    "resolution_status",
    "cirp status":                       "resolution_status",
    "outcome":                           "resolution_status",
}


def detect_columns(raw_headers: list[str]) -> dict[str, str]:
    """
    Maps raw IBBI column headers to canonical names.

    Strategy (in order):
    1. Exact match against alias map (after lowercasing + stripping)
    2. Fuzzy match against alias map keys (threshold 80)
    3. Fuzzy match directly against canonical names (threshold 75)
    4. Omit if nothing matches confidently

    Parameters
    ----------
    raw_headers : list of strings from the actual file header row

    Returns
    -------
    dict mapping raw_header → canonical_name
    """
    mapping = {}
    alias_keys = list(_COLUMN_ALIASES.keys())

    for raw in raw_headers:
        if not raw or not raw.strip():
            continue

        normalised = raw.strip().lower()

        # 1. Exact alias match
        if normalised in _COLUMN_ALIASES:
            mapping[raw] = _COLUMN_ALIASES[normalised]
            continue

        # 2. Fuzzy match against alias keys
        result = process.extractOne(
            normalised, alias_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=80
        )
        if result:
            matched_alias, score, _ = result
            mapping[raw] = _COLUMN_ALIASES[matched_alias]
            continue

        # 3. Fuzzy match directly against canonical names as last resort
        result = process.extractOne(
            normalised, list(CANONICAL_COLUMNS),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=75
        )
        if result:
            canonical, score, _ = result
            mapping[raw] = canonical

    print(f"  [detect_columns] Mapped {len(mapping)}/{len(raw_headers)} headers")
    return mapping


def normalise_company_names(names: list[str]) -> dict[str, str]:
    """
    Normalises Indian company name spellings deterministically.

    Rules applied:
    - Title Case
    - "Ltd"/"Ltd." → "Limited"
    - "Pvt"/"Pvt." → "Private"
    - "& " → "and "
    - Strip extra spaces and trailing punctuation
    - Common abbreviations expanded
    """
    import re

    substitutions = [
        (r"\bLtd\.?\b",      "Limited"),
        (r"\bPvt\.?\b",      "Private"),
        (r"\bCo\.?\b",       "Company"),
        (r"\bCorp\.?\b",     "Corporation"),
        (r"\bMfg\.?\b",      "Manufacturing"),
        (r"\bInfra\.?\b",    "Infrastructure"),
        (r"\bTech\.?\b",     "Technologies"),
        (r"\bIndl\.?\b",     "Industries"),
        (r"\bInds\.?\b",     "Industries"),
        (r"&",               "and"),
    ]

    result = {}
    for name in names:
        if not name or not str(name).strip():
            result[name] = name
            continue

        cleaned = str(name).strip()

        # Apply substitutions
        for pattern, replacement in substitutions:
            cleaned = re.sub(pattern, replacement, cleaned,
                             flags=re.IGNORECASE)

        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Strip trailing punctuation
        cleaned = cleaned.rstrip(".,;:")

        # Title Case
        cleaned = cleaned.title()

        # Fix "And" back to "and" (title case overcorrects)
        cleaned = re.sub(r"\bAnd\b", "and", cleaned)

        result[name] = cleaned

    return result


def normalise_company_names_batched(names: list[str],
                                     batch_size: int = 100) -> dict[str, str]:
    """
    Batched version of normalise_company_names.
    batch_size param kept for API compatibility even though
    the deterministic version doesn't need batching.
    """
    if not names:
        return {}

    unique_names = list(dict.fromkeys(names))
    print(f"  [normalise] Normalising {len(unique_names)} unique company names...")
    result = normalise_company_names(unique_names)
    print(f"  [normalise] Done.")
    return result


def generate_risk_narrative(assessment: dict) -> str:
    """
    Generates a plain-English risk narrative from assessment scores.
    Template-based — deterministic, no API needed.
    """
    duration    = assessment.get("duration_months", "?")
    dur_low     = assessment.get("duration_low", "?")
    dur_high    = assessment.get("duration_high", "?")
    outcome_prob = assessment.get("outcome_prob", 0)
    outcome_label = assessment.get("outcome_label", "Uncertain")
    real_pct    = assessment.get("realisation_pct", None)
    risk_score  = assessment.get("risk_score", 0)
    rec         = assessment.get("recommendation", "Review Required")

    # Duration tone
    if isinstance(duration, (int, float)):
        if duration <= 24:
            dur_tone = "a relatively short"
        elif duration <= 48:
            dur_tone = "a moderate"
        else:
            dur_tone = "a prolonged"
    else:
        dur_tone = "an uncertain"

    # Outcome tone
    if outcome_prob >= 0.65:
        out_tone = "strong"
    elif outcome_prob >= 0.50:
        out_tone = "moderate"
    else:
        out_tone = "weak"

    # Build narrative
    narrative = (
        f"This case presents {dur_tone} timeline, with a median expected "
        f"duration of {duration} months (range: {dur_low}–{dur_high} months). "
        f"The probability of a favourable outcome is {out_tone} at "
        f"{outcome_prob:.0%}, suggesting the claimant's position is "
        f"{'above' if outcome_prob >= 0.5 else 'below'} the risk threshold. "
    )

    if real_pct and real_pct > 0:
        narrative += (
            f"Expected financial recovery stands at {real_pct:.1f}% of the "
            f"admitted claim, "
            f"{'which is commercially viable' if real_pct >= 30 else 'which warrants caution on capital deployment'}. "
        )

    narrative += (
        f"Overall risk score: {risk_score}/100. "
        f"Assessment recommendation: {rec}."
    )

    return narrative