"""
src/genai/genai_utils.py
------------------------
GenAI-powered utilities for ILFRA's data ingestion and reporting pipeline.

Three functions, each calling the HuggingFace Inference API:

  1. detect_columns(raw_headers)         → maps messy Excel headers to canonical schema
  2. normalise_company_names(names)      → deduplicates corporate debtor name variants
  3. generate_risk_narrative(case, preds) → writes a plain-English risk summary for PDF

Design principles:
  - GenAI handles ambiguity; deterministic code handles numbers and decisions.
  - Every function has a safe fallback — the pipeline never crashes due to a failed
    API call. Column detection falls back to an empty mapping (caller handles missing
    columns gracefully). Normalisation falls back to the original names. Narrative
    falls back to a generic template string.
  - All calls are synchronous and stateless — no conversation history needed.
  - The HF token is read from the environment (.env file via python-dotenv).

Environment variables required (.env):
  HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
  HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3   # optional override

Install dependencies:
  pip install requests python-dotenv
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID: str = os.getenv(
    "HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"
)
HF_API_URL: str = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Timeout and retry settings
REQUEST_TIMEOUT: int = 60          # seconds per request
MAX_RETRIES: int = 3               # retry on model-loading (503) responses
RETRY_WAIT: int = 20               # seconds to wait between retries (HF cold start)

# ── Canonical IBBI schema ─────────────────────────────────────────────────────
# These are the exact column names the rest of the pipeline expects.
# detect_columns() maps arbitrary Excel headers to these names.

CANONICAL_COLUMNS: list[str] = [
    "cirp_id",                       # case / serial number
    "company_name",                  # corporate debtor name
    "sector",                        # industry sector
    "bench",                         # NCLT bench (city)
    "admission_date",                # date CIRP was admitted
    "resolution_status",             # current status (Resolution Plan / Liquidation / etc.)
    "duration_days",                 # days taken from admission to closure
    "admitted_claim_cr",             # total admitted financial claims (₹ crore)
    "liquidation_value_cr",          # liquidation value of assets (₹ crore)
    "fair_value_cr",                 # fair value of assets (₹ crore)
    "realisation_cr",                # amount actually realised (₹ crore)
    "realisation_pct",               # realisation as % of admitted claims
    "no_of_financial_creditors",     # count of financial creditors
    "resolution_applicants_received",# number of resolution plans received
    "ip_changed",                    # whether insolvency professional was replaced (0/1)
    "litigation_pending",            # whether creditor litigation is ongoing (0/1)
    "initiated_by",                  # who filed (Financial Creditor / Operational Creditor / Corporate Debtor)
]


# ── Internal HTTP helper ──────────────────────────────────────────────────────
def is_connected():
    try:
        socket.setdefaulttimeout(3)
        socket.getaddrinfo("api-inference.huggingface.co", 443)
        return True
    except OSError:
        return False

def _hf_generate(prompt: str, max_new_tokens: int = 512) -> Optional[str]:
    """
    Sends a prompt to the HuggingFace Inference API and returns the generated text.

    Handles cold-start 503 responses from HF (model not loaded yet) with retries.
    Returns None on any unrecoverable failure so callers can use their fallback.

    Parameters
    ----------
    prompt         : The full formatted prompt string (instruction + context).
    max_new_tokens : Maximum tokens to generate in the response.

    Returns
    -------
    str | None — raw generated text, or None if the API call failed.
    """
    if not HF_API_TOKEN:
        logger.warning(
            "[genai] HF_API_TOKEN not set in .env — GenAI features disabled. "
            "Add HF_API_TOKEN=hf_... to your .env file."
        )
        return None

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,        # low temperature → more deterministic, better for structured output
            "do_sample": True,
            "return_full_text": False, # return only the newly generated text, not the prompt
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            # HF returns 503 when the model is still loading (cold start)
            if response.status_code == 503:
                if attempt < MAX_RETRIES:
                    logger.info(
                        f"[genai] Model loading (503) — waiting {RETRY_WAIT}s "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_WAIT)
                    continue
                else:
                    logger.error("[genai] Model still loading after max retries.")
                    return None

            response.raise_for_status()
            result = response.json()

            # HF returns a list of generated sequences
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "").strip()

            logger.error(f"[genai] Unexpected response format: {result}")
            return None

        except requests.exceptions.Timeout:
            logger.warning(f"[genai] Request timed out (attempt {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"[genai] HTTP error: {e}")
            return None

    return None


def _extract_json(text: str) -> Optional[dict | list]:
    """
    Extracts the first valid JSON object or array from a model response string.

    Models sometimes wrap JSON in markdown code blocks or add prose before/after.
    This strips that noise and returns the parsed Python object.

    Returns None if no valid JSON is found.
    """
    if not text:
        return None

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try to find a JSON object {...} or array [...]
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    return None


# ── 1. Smart column detection ─────────────────────────────────────────────────

def detect_columns(raw_headers: list[str]) -> dict[str, str]:
    """
    Maps raw Excel column headers to ILFRA's canonical schema using an LLM.

    The IBBI quarterly Excels change column naming conventions across quarters:
    "Admitted Amount of Claims (in Cr.)" → "admitted_claim_cr"
    "% of Realisation"                   → "realisation_pct"
    "Name of CD"                         → "company_name"
    etc.

    Deterministic string matching breaks across quarters. This function asks
    the LLM to interpret headers semantically and return a JSON mapping.

    Parameters
    ----------
    raw_headers : list of column header strings exactly as read from the Excel file.

    Returns
    -------
    dict mapping raw_header → canonical_name for headers the model is confident about.
    Unrecognised headers are omitted — the caller must handle missing columns.

    Example
    -------
    >>> detect_columns(["Sr. No.", "Name of CD", "% of Realisation wrt Admitted Claims"])
    {
        "Sr. No.": "cirp_id",
        "Name of CD": "company_name",
        "% of Realisation wrt Admitted Claims": "realisation_pct"
    }

    Fallback
    --------
    Returns {} if the API call fails or the response cannot be parsed.
    The caller (ibbi_excel.py) should handle an empty mapping gracefully
    by logging a warning and skipping the file.
    """
    canonical_list = "\n".join(f"  - {c}" for c in CANONICAL_COLUMNS)
    headers_str = json.dumps(raw_headers, indent=2)

    prompt = f"""<s>[INST]
You are a data engineering assistant helping parse Indian government Excel files from the Insolvency and Bankruptcy Board of India (IBBI).

Your task: map raw Excel column headers to canonical column names used in our database schema.

CANONICAL COLUMN NAMES (these are the only valid output values):
{canonical_list}

RAW HEADERS FROM THIS EXCEL FILE:
{headers_str}

INSTRUCTIONS:
1. For each raw header you recognise, map it to the best matching canonical name.
2. Only include mappings you are confident about. Skip headers that don't match any canonical column.
3. Multiple raw headers must not map to the same canonical name — pick the best one.
4. Return ONLY a valid JSON object. No explanation, no markdown, no prose.

Example output format:
{{"Sr. No.": "cirp_id", "Name of Corporate Debtor": "company_name", "% Realisation": "realisation_pct"}}

JSON mapping:
[/INST]"""

    raw_response = _hf_generate(prompt, max_new_tokens=300)

    if not raw_response:
        logger.warning("[genai] detect_columns: API call failed — returning empty mapping.")
        return {}

    parsed = _extract_json(raw_response)

    if not isinstance(parsed, dict):
        logger.warning(
            f"[genai] detect_columns: Could not parse JSON from response:\n{raw_response}"
        )
        return {}

    # Validate: only keep entries where the value is a known canonical column
    canonical_set = set(CANONICAL_COLUMNS)
    validated: dict[str, str] = {}
    seen_canonical: set[str] = set()

    for raw_col, canonical_col in parsed.items():
        if canonical_col not in canonical_set:
            logger.debug(f"[genai] Skipping unknown canonical name: '{canonical_col}'")
            continue
        if canonical_col in seen_canonical:
            logger.debug(f"[genai] Duplicate mapping for '{canonical_col}' — skipping '{raw_col}'")
            continue
        if raw_col not in raw_headers:
            logger.debug(f"[genai] Raw header '{raw_col}' not in actual headers — skipping")
            continue
        validated[raw_col] = canonical_col
        seen_canonical.add(canonical_col)

    logger.info(f"[genai] detect_columns: mapped {len(validated)}/{len(raw_headers)} headers.")
    return validated


# ── 2. Company name normalisation ─────────────────────────────────────────────

def normalise_company_names(names: list[str]) -> dict[str, str]:
    """
    Normalises corporate debtor name variants to a single canonical form.

    IBBI data across quarterly files uses inconsistent naming:
    "Videocon Industries Ltd", "VIDEOCON INDUSTRIES LIMITED", "Videocon Ind. Ltd"
    → all should map to "Videocon Industries Limited"

    This deduplication is essential before training — duplicate rows with the same
    company inflates the dataset and introduces label leakage.

    Parameters
    ----------
    names : list of raw company name strings (may contain duplicates/variants).

    Returns
    -------
    dict mapping each input name → its normalised canonical form.
    Names that appear unique and clean are mapped to themselves.

    Example
    -------
    >>> normalise_company_names([
    ...     "Videocon Industries Ltd",
    ...     "VIDEOCON INDUSTRIES LIMITED",
    ...     "Jet Airways (India) Ltd",
    ...     "JET AIRWAYS INDIA LIMITED",
    ... ])
    {
        "Videocon Industries Ltd": "Videocon Industries Limited",
        "VIDEOCON INDUSTRIES LIMITED": "Videocon Industries Limited",
        "Jet Airways (India) Ltd": "Jet Airways India Limited",
        "JET AIRWAYS INDIA LIMITED": "Jet Airways India Limited",
    }

    Fallback
    --------
    Returns {name: name for name in names} (identity mapping) if the API call fails.
    This is safe — the pipeline continues with un-normalised names rather than crashing.

    Performance note
    ----------------
    Send names in batches of 30–50 for best results. Very large lists (200+) may
    exceed the model's context window and produce incomplete output.
    """
    fallback = {name: name for name in names}

    if not names:
        return fallback
    
    if not is_connected():
        logger.debug("[genai] normalise_company_names: no network — using identity mapping.")
        return fallback
    # Deduplicate before sending to avoid wasting tokens on exact duplicates
    unique_names = list(dict.fromkeys(names))  # preserves order, removes exact dupes

    names_str = json.dumps(unique_names, indent=2)

    prompt = f"""<s>[INST]
You are a data cleaning assistant working with Indian corporate insolvency records.

Your task: normalise company names. The same company often appears with different spellings, 
abbreviations, or casing across different quarterly reports. Group variants and pick one 
canonical form for each company.

Rules for the canonical form:
1. Use Title Case (e.g. "Videocon Industries Limited" not "VIDEOCON INDUSTRIES LIMITED")
2. Spell out abbreviations where clear (Ltd → Limited, Pvt → Private, Co → Company)
3. Remove redundant bracketed clarifications if they add no information (e.g. "(India)" when it's already an Indian company)
4. Keep legally significant words (e.g. keep "India" if it distinguishes from a foreign entity)

COMPANY NAMES TO NORMALISE:
{names_str}

Return ONLY a valid JSON object mapping each input name to its canonical form.
Every input name must appear as a key. Do not skip any.
No explanation, no markdown, no prose.

Example output:
{{"VIDEOCON INDUSTRIES LTD": "Videocon Industries Limited", "Videocon Ind. Ltd": "Videocon Industries Limited"}}

JSON mapping:
[/INST]"""

    raw_response = _hf_generate(prompt, max_new_tokens=600)

    if not raw_response:
        logger.warning("[genai] normalise_company_names: API call failed — using identity mapping.")
        return fallback

    parsed = _extract_json(raw_response)

    if not isinstance(parsed, dict):
        logger.warning(
            f"[genai] normalise_company_names: Could not parse JSON:\n{raw_response}"
        )
        return fallback

    # Build final mapping covering all original (non-deduplicated) input names
    result: dict[str, str] = {}
    for name in names:
        # Use the normalised form if the model returned it, else fall back to original
        result[name] = parsed.get(name, name)

    logger.info(
        f"[genai] normalise_company_names: processed {len(unique_names)} unique names."
    )
    return result


# ── 3. Dynamic risk narrative ─────────────────────────────────────────────────

def generate_risk_narrative(case_inputs: dict, predictions: dict) -> str:
    """
    Generates a 2–3 sentence plain-English risk narrative for the PDF report.

    This replaces the static disclaimer boilerplate in the PDF with a case-specific
    summary paragraph that a funder client can read without interpreting numbers.

    Parameters
    ----------
    case_inputs  : dict — raw user-submitted case fields (same dict passed to predict_case)
    predictions  : dict — model output dict (same flat schema as predict.py returns)

    Returns
    -------
    A 2–3 sentence narrative string. Example:

    "This Commercial Dispute filed in the Delhi High Court carries a moderate risk profile,
    with the model estimating a 67% probability of a favourable outcome for the claimant
    and an expected duration of 28 months. The estimated financial recovery range of
    42–71% of the admitted claim, combined with a composite risk score of 61/100,
    suggests this matter is a viable candidate for funding subject to legal due diligence.
    Key risk factors include a high adjournment count and the absence of an interim order."

    Fallback
    --------
    Returns a generic template string populated with the prediction values if the
    API call fails. The PDF is never left with a blank narrative section.
    """
    # Build a concise context string for the prompt
    case_type     = case_inputs.get("case_type", "legal dispute")
    court         = case_inputs.get("court", "court")
    state         = case_inputs.get("state", "")
    sector        = case_inputs.get("sector", "")
    claim         = case_inputs.get("claim_amount_lakhs", 0)

    dur           = predictions.get("duration_months", 0)
    dur_low       = predictions.get("duration_low", 0)
    dur_high      = predictions.get("duration_high", 0)
    prob          = predictions.get("outcome_prob", 0.5)
    real          = predictions.get("realisation_pct", 0)
    real_low      = predictions.get("realisation_low", 0)
    real_high     = predictions.get("realisation_high", 0)
    risk_score    = predictions.get("risk_score", 50)
    recommendation = predictions.get("recommendation", "")
    data_source   = predictions.get("data_source", "NJDG")

    adjournments  = case_inputs.get("num_prior_adjournments", 0)
    interim_order = case_inputs.get("has_interim_order", False)
    senior_counsel = case_inputs.get("represented_by_senior_counsel", False)
    lawyer_wr     = case_inputs.get("claimant_lawyer_win_rate", 0.5)

    # Summarise key risk signals for the model to reason about
    risk_signals = []
    if adjournments > 15:
        risk_signals.append(f"high adjournment count ({adjournments})")
    if not interim_order:
        risk_signals.append("no interim order obtained")
    if senior_counsel:
        risk_signals.append("senior counsel engaged")
    if lawyer_wr >= 0.65:
        risk_signals.append(f"strong claimant lawyer track record ({lawyer_wr:.0%} win rate)")
    elif lawyer_wr <= 0.35:
        risk_signals.append(f"weak claimant lawyer track record ({lawyer_wr:.0%} win rate)")

    risk_signal_str = (
        ", ".join(risk_signals) if risk_signals else "no notable risk signals identified"
    )

    realisation_line = (
        f"Estimated financial recovery: {real_low:.1f}%–{real_high:.1f}% of admitted claims "
        f"(median {real:.1f}%)."
        if real > 0 else
        "Financial recovery estimate not applicable for this case type."
    )

    prompt = f"""<s>[INST]
You are a senior litigation finance analyst writing a brief risk summary for a funder client.

Write exactly 2–3 sentences summarising the risk assessment below. 
Use plain English. Be specific — mention actual numbers. 
Do not use bullet points, headers, or markdown. 
Do not start with "This report" or "Based on". 
Do not include any disclaimer or legal caveat — those appear elsewhere in the report.

CASE DETAILS:
- Type: {case_type}
- Court: {court}, {state}
- Sector: {sector}
- Claim amount: ₹{claim:,.0f} lakhs
- Data source: {data_source}

MODEL PREDICTIONS:
- Probability of favourable outcome: {prob:.0%}
- Expected duration: {dur:.0f} months (range: {dur_low:.0f}–{dur_high:.0f} months)
- {realisation_line}
- Composite risk score: {risk_score:.0f}/100
- Overall recommendation: {recommendation}

KEY RISK SIGNALS:
- {risk_signal_str}

Write the 2–3 sentence narrative now:
[/INST]"""

    raw_response = _hf_generate(prompt, max_new_tokens=200)

    if raw_response and len(raw_response.strip()) > 30:
        # Light cleanup — strip any accidental leading/trailing quotes or newlines
        narrative = raw_response.strip().strip('"').strip("'").strip()
        logger.info("[genai] generate_risk_narrative: narrative generated successfully.")
        return narrative

    # Fallback: template-based narrative using prediction values
    logger.warning("[genai] generate_risk_narrative: API call failed — using template fallback.")
    fallback_narrative = (
        f"This {case_type} filed at {court} carries a composite risk score of "
        f"{risk_score:.0f}/100, with the model estimating a {prob:.0%} probability of a "
        f"favourable outcome and an expected duration of {dur:.0f} months "
        f"(range: {dur_low:.0f}–{dur_high:.0f} months). "
    )
    if real > 0:
        fallback_narrative += (
            f"The estimated financial recovery range is {real_low:.1f}%–{real_high:.1f}% "
            f"of admitted claims, with a median estimate of {real:.1f}%. "
        )
    fallback_narrative += f"Overall assessment: {recommendation}."
    return fallback_narrative


# ── Convenience batch wrapper ─────────────────────────────────────────────────

def normalise_company_names_batched(
    names: list[str], batch_size: int = 40
) -> dict[str, str]:
    """
    Wrapper around normalise_company_names() that splits large name lists into
    batches to avoid exceeding the model's context window.

    Parameters
    ----------
    names      : full list of company name strings
    batch_size : number of names per API call (default 40)

    Returns
    -------
    Combined dict mapping all input names to their normalised forms.
    """
    if not names:
        return {}

    result: dict[str, str] = {}
    for i in range(0, len(names), batch_size):
        batch = names[i: i + batch_size]
        batch_result = normalise_company_names(batch)
        result.update(batch_result)
        logger.info(
            f"[genai] normalise_company_names_batched: "
            f"processed batch {i // batch_size + 1} ({len(batch)} names)"
        )

    return result