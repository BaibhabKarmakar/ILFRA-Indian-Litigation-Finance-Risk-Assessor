"""
src/parsers/ibbi_pdf.py
-----------------------
Parses IBBI quarterly newsletter PDF files using pdfplumber.

Mirrors the output schema of ibbi_excel.py exactly.

Key insight from actual PDF inspection:
  - Early PDFs (2017, early 2018) contain only "Table A: CIRP Applications
    admitted" — no resolution/liquidation outcome tables yet.
  - Later PDFs (mid-2018 onwards) contain:
      * "Table N: CIRPs Yielding Resolution" — resolution outcomes with
        admitted claims, liquidation value, realisation figures.
      * "Table N: CIRPs Ending with Orders for Liquidation" — liquidation
        outcomes with CIRP commencement and liquidation order dates.
  - pdfplumber's extract_tables() works well on these PDFs; the issue was
    that the old parser looked for wrong keywords and tried row-by-row
    detection instead of using the structured table output.

Strategy:
  1. Walk every page with pdfplumber, collecting ALL tables.
  2. For each table, check its header row for resolution/liquidation keywords
     using a header-fingerprinting approach.
  3. Parse columns by matching header text to canonical fields rather than
     relying on fixed positional indices.
  4. For the large "blob" tables (where pdfplumber merges everything into
     one giant cell), fall back to parsing the page's raw text with regex.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)

# ── Output schema (must match ibbi_excel.py) ──────────────────────────────────
_SCHEMA_COLS = [
    "company_name", "cirp_start_date", "resolution_date",
    "admitted_claim_cr", "liquidation_value", "realisable_amount",
    "realisation_pct", "resolution_status", "source_table", "quarter",
]

# ── Header fingerprints ───────────────────────────────────────────────────────
# These substrings identify a table's type from its header row.
_RES_HEADER_KEYWORDS = [
    "cirps yielding resolution",
    "yielding resolution",
    "resolution plan",          # "CIRPs Yielding Resolution Plan"
]
_LIQ_HEADER_KEYWORDS = [
    "ending with orders for liquidation",
    "orders for liquidation",
    "closed liquidation",
    "liquidation order",
]

# ── Column name → canonical field mapping ─────────────────────────────────────
# Each tuple is (substring_to_match_in_header_text, canonical_field_name).
# Matching is case-insensitive substring search.
_RES_COL_MAP = [
    ("name of cd",          "company_name"),
    ("name of corporate",   "company_name"),
    ("corporate debtor",    "company_name"),       # fallback
    ("commencement",        "cirp_start_date"),
    ("approval",            "resolution_date"),
    ("date of resolution",  "resolution_date"),
    ("total admitted",      "admitted_claim_cr"),
    ("admitted claim",      "admitted_claim_cr"),
    ("liquidation value",   "liquidation_value"),
    ("realisation by fcs",  "realisable_amount"),  # "Realisation by FCs" column
    ("realisable",          "realisable_amount"),
    ("realisation\n",       "realisable_amount"),
    ("% of their claims",   "realisation_pct"),
    ("realisation by fcs as % of their", "realisation_pct"),
    ("% realisation",       "realisation_pct"),
]
_LIQ_COL_MAP = [
    ("name of corporate",   "company_name"),
    ("corporate debtor",    "company_name"),
    ("commencement",        "cirp_start_date"),
    ("liquidation order",   "resolution_date"),
    ("date of liquidation", "resolution_date"),
    ("admitted",            "admitted_claim_cr"),
    ("liquidation value",   "liquidation_value"),
    ("distributed",         "realisable_amount"),
    ("realisation",         "realisable_amount"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = _normalise(text)
    return any(kw in t for kw in keywords)


def _parse_number(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r"[^\d.]", "", str(val).replace(",", ""))
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _try_parse_date(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    for fmt in ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y",
                "%b %Y", "%B %Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None


def _map_header_to_fields(header_row: list, col_map: list) -> dict[int, str]:
    """
    Given a header row (list of cell values) and a col_map (list of
    (substring, canonical) tuples), return {col_index: canonical_field}.
    Each canonical field is only assigned once (first match wins).
    """
    assigned: dict[str, int] = {}   # canonical → col_index (to avoid duplicates)
    result: dict[int, str] = {}

    for col_idx, cell in enumerate(header_row):
        cell_norm = _normalise(cell)
        if not cell_norm:
            continue
        for substring, canonical in col_map:
            if substring in cell_norm and canonical not in assigned:
                assigned[canonical] = col_idx
                result[col_idx] = canonical
                break

    return result


def _is_header_row(row: list, col_map: list) -> bool:
    """Returns True if this row looks like a column header (has 2+ mappable fields)."""
    mappings = _map_header_to_fields(row, col_map)
    return len(mappings) >= 2


def _is_data_row(row: list) -> bool:
    """Returns True if the first non-None cell looks like a serial number."""
    for cell in row[:3]:
        if cell is None:
            continue
        s = str(cell).strip()
        if s.isdigit():
            return True
        # Also accept "1.", "1)" etc.
        if re.match(r"^\d+[.)]*$", s):
            return True
    return False


# ── Table-type detection ──────────────────────────────────────────────────────

def _detect_table_type(table: list[list]) -> Optional[str]:
    """
    Inspects the first few rows of a pdfplumber table to determine
    whether it is a resolution or liquidation outcomes table.
    Returns "resolution", "liquidation", or None.
    """
    # Scan first 4 rows for type keywords
    for row in table[:4]:
        row_text = " ".join(_normalise(c) for c in row if c)
        if _contains_any(row_text, _RES_HEADER_KEYWORDS):
            return "resolution"
        if _contains_any(row_text, _LIQ_HEADER_KEYWORDS):
            return "liquidation"
    return None


# ── Structured table parsers ──────────────────────────────────────────────────

def _parse_resolution_table(table: list[list], quarter: str) -> list[dict]:
    """
    Parses a structured pdfplumber table that has been identified as a
    resolution outcomes table. Returns a list of row dicts.
    """
    rows_out = []
    col_mapping: dict[int, str] = {}
    header_found = False

    for row in table:
        # Skip completely empty rows
        if all(c is None or str(c).strip() == "" for c in row):
            continue

        row_text = " ".join(_normalise(c) for c in row if c)

        # Re-detect header rows even mid-table (multi-page tables repeat headers)
        if not header_found or _is_header_row(row, _RES_COL_MAP):
            candidate = _map_header_to_fields(row, _RES_COL_MAP)
            if len(candidate) >= 3:
                col_mapping = candidate
                header_found = True
                continue

        if not col_mapping:
            continue

        # Skip non-data rows (section titles, totals, notes)
        if not _is_data_row(row):
            continue

        # Build record from column mapping
        def get(canonical: str):
            for idx, name in col_mapping.items():
                if name == canonical and idx < len(row):
                    return row[idx]
            return None

        company    = get("company_name")
        start_date = _try_parse_date(get("cirp_start_date"))
        res_date   = _try_parse_date(get("resolution_date"))
        admitted   = _parse_number(get("admitted_claim_cr"))
        liq_val    = _parse_number(get("liquidation_value"))
        realisable = _parse_number(get("realisable_amount"))
        real_pct   = _parse_number(get("realisation_pct"))

        if admitted is None or admitted <= 0:
            continue

        # Compute realisation_pct if not directly present
        if real_pct is None and realisable is not None and admitted > 0:
            real_pct = round(realisable / admitted * 100, 2)

        rows_out.append({
            "company_name":      str(company).strip() if company else None,
            "cirp_start_date":   start_date,
            "resolution_date":   res_date,
            "admitted_claim_cr": admitted,
            "liquidation_value": liq_val if liq_val is not None else np.nan,
            "fair_value":        np.nan,
            "realisable_amount": realisable if realisable is not None else np.nan,
            "realisation_pct":   real_pct if real_pct is not None else np.nan,
            "resolution_status": "Resolution Plan Approved",
            "source_table":      "pdf_resolution",
            "quarter":           quarter,
        })

    return rows_out


def _parse_liquidation_table(table: list[list], quarter: str) -> list[dict]:
    """
    Parses a structured pdfplumber table identified as a liquidation
    outcomes table. Returns a list of row dicts.
    """
    rows_out = []
    col_mapping: dict[int, str] = {}
    header_found = False

    for row in table:
        if all(c is None or str(c).strip() == "" for c in row):
            continue

        if not header_found or _is_header_row(row, _LIQ_COL_MAP):
            candidate = _map_header_to_fields(row, _LIQ_COL_MAP)
            if len(candidate) >= 2:
                col_mapping = candidate
                header_found = True
                continue

        if not col_mapping:
            continue

        if not _is_data_row(row):
            continue

        def get(canonical: str):
            for idx, name in col_mapping.items():
                if name == canonical and idx < len(row):
                    return row[idx]
            return None

        company    = get("company_name")
        start_date = _try_parse_date(get("cirp_start_date"))
        liq_date   = _try_parse_date(get("resolution_date"))
        admitted   = _parse_number(get("admitted_claim_cr"))
        liq_val    = _parse_number(get("liquidation_value"))
        distributed = _parse_number(get("realisable_amount"))

        if company is None:
            continue

        real_pct = np.nan
        if distributed is not None and admitted and admitted > 0:
            real_pct = round(distributed / admitted * 100, 2)

        rows_out.append({
            "company_name":      str(company).strip(),
            "cirp_start_date":   start_date,
            "resolution_date":   liq_date,
            "admitted_claim_cr": admitted if admitted is not None else np.nan,
            "liquidation_value": liq_val if liq_val is not None else np.nan,
            "fair_value":        np.nan,
            "realisable_amount": distributed if distributed is not None else np.nan,
            "realisation_pct":   real_pct,
            "resolution_status": "Liquidation Order",
            "source_table":      "pdf_liquidation",
            "quarter":           quarter,
        })

    return rows_out


# ── Blob-text fallback parser ─────────────────────────────────────────────────
# Some PDF pages merge the entire content into a single large cell.
# In that case we parse the raw page text with regex.

_DATE_PAT = r"\d{2}[-./]\d{2}[-./]\d{4}"

def _parse_resolution_from_text(text: str, quarter: str) -> list[dict]:
    """
    Extracts resolution rows from raw page text using regex.
    Matches lines like:
      1  Company Name  Yes/No  DD-MM-YYYY  DD-MM-YYYY  FC  1234.56  567.89  890.12  ...
    """
    rows_out = []
    # Pattern: serial, company name (anything), BIFR flag, two dates, initiator, numbers
    pattern = re.compile(
        r"^(\d+)\s+"                             # serial number
        r"(.+?)\s+"                              # company name (non-greedy)
        r"(?:Yes|No)\s+"                         # BIFR flag
        r"(" + _DATE_PAT + r")\s+"              # CIRP commencement date
        r"(" + _DATE_PAT + r")\s+"              # approval date
        r"(FC|OC|CD)\s+"                         # initiated by
        r"([\d,.]+)\s+"                          # total admitted claims
        r"([\d,.]+)\s+"                          # liquidation value
        r"([\d,.]+)",                            # realisation by FCs
        re.IGNORECASE | re.MULTILINE,
    )
    for m in pattern.finditer(text):
        company    = m.group(2).strip()
        start_date = _try_parse_date(m.group(3))
        res_date   = _try_parse_date(m.group(4))
        admitted   = _parse_number(m.group(6))
        liq_val    = _parse_number(m.group(7))
        realisable = _parse_number(m.group(8))

        if admitted is None or admitted <= 0:
            continue

        real_pct = round(realisable / admitted * 100, 2) if realisable and admitted > 0 else np.nan

        rows_out.append({
            "company_name":      company,
            "cirp_start_date":   start_date,
            "resolution_date":   res_date,
            "admitted_claim_cr": admitted,
            "liquidation_value": liq_val if liq_val is not None else np.nan,
            "fair_value":        np.nan,
            "realisable_amount": realisable if realisable is not None else np.nan,
            "realisation_pct":   real_pct,
            "resolution_status": "Resolution Plan Approved",
            "source_table":      "pdf_resolution_text",
            "quarter":           quarter,
        })
    return rows_out


def _parse_liquidation_from_text(text: str, quarter: str) -> list[dict]:
    """
    Extracts liquidation rows from raw page text.
    Matches lines like:
      1  Company Name  Yes/No  FC  DD-MM-YYYY  DD-MM-YYYY
    """
    rows_out = []
    pattern = re.compile(
        r"^(\d+)\s+"
        r"(.+?)\s+"
        r"(?:Yes|No)\s+"
        r"(FC|OC|CD)\s+"
        r"(" + _DATE_PAT + r")\s+"
        r"(" + _DATE_PAT + r")",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in pattern.finditer(text):
        company    = m.group(2).strip()
        start_date = _try_parse_date(m.group(4))
        liq_date   = _try_parse_date(m.group(5))

        rows_out.append({
            "company_name":      company,
            "cirp_start_date":   start_date,
            "resolution_date":   liq_date,
            "admitted_claim_cr": np.nan,
            "liquidation_value": np.nan,
            "fair_value":        np.nan,
            "realisable_amount": np.nan,
            "realisation_pct":   0.0,     # liquidation → 0% assumed
            "resolution_status": "Liquidation Order",
            "source_table":      "pdf_liquidation_text",
            "quarter":           quarter,
        })
    return rows_out


# ── Main parser entry point ───────────────────────────────────────────────────

def parse(path: Path) -> pd.DataFrame:
    """
    Main entry point called by ibbi_channel.py for .pdf files.

    Returns a DataFrame with the same schema as ibbi_excel.py:
        company_name, cirp_start_date, resolution_date,
        admitted_claim_cr, liquidation_value, fair_value,
        realisable_amount, realisation_pct,
        resolution_status, source_table, quarter
    """
    if not HAS_PDFPLUMBER:
        raise ImportError(
            "pdfplumber is required. Install with: pip install pdfplumber"
        )

    quarter_label = path.stem
    resolution_rows: list[dict] = []
    liquidation_rows: list[dict] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            logger.info(f"[ibbi_pdf] Opened {path.name} — {len(pdf.pages)} page(s)")

            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                tables = page.extract_tables() or []

                # ── Pass 1: check structured tables ───────────────────────
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # Skip tables where pdfplumber merged everything into 1 cell
                    # (these have ≤2 non-None cells in first row)
                    first_row_non_null = sum(
                        1 for c in table[0] if c and str(c).strip()
                    )
                    if first_row_non_null <= 2 and len(table[0]) <= 3:
                        # Try blob text parsing for this page instead
                        continue

                    table_type = _detect_table_type(table)

                    if table_type == "resolution":
                        rows = _parse_resolution_table(table, quarter_label)
                        resolution_rows.extend(rows)
                        logger.debug(
                            f"[ibbi_pdf] p{page_num}: resolution table → {len(rows)} rows"
                        )

                    elif table_type == "liquidation":
                        rows = _parse_liquidation_table(table, quarter_label)
                        liquidation_rows.extend(rows)
                        logger.debug(
                            f"[ibbi_pdf] p{page_num}: liquidation table → {len(rows)} rows"
                        )

                # ── Pass 2: text-based fallback for pages with blob tables ─
                # Only apply if no structured rows were found on this page
                # AND the page text contains the relevant section keywords.
                page_res_found = any(
                    r["source_table"] == "pdf_resolution" for r in resolution_rows
                )
                page_liq_found = any(
                    r["source_table"] == "pdf_liquidation" for r in liquidation_rows
                )

                if not page_res_found and _contains_any(
                    page_text, _RES_HEADER_KEYWORDS
                ):
                    rows = _parse_resolution_from_text(page_text, quarter_label)
                    if rows:
                        resolution_rows.extend(rows)
                        logger.debug(
                            f"[ibbi_pdf] p{page_num}: text fallback → {len(rows)} resolution rows"
                        )

                if not page_liq_found and _contains_any(
                    page_text, _LIQ_HEADER_KEYWORDS
                ):
                    rows = _parse_liquidation_from_text(page_text, quarter_label)
                    if rows:
                        liquidation_rows.extend(rows)
                        logger.debug(
                            f"[ibbi_pdf] p{page_num}: text fallback → {len(rows)} liquidation rows"
                        )

    except Exception as e:
        logger.error(f"[ibbi_pdf] Failed to parse {path.name}: {e}")
        return pd.DataFrame()

    print(
        f"  [ibbi_pdf] {path.name} → "
        f"{len(resolution_rows)} resolution rows, {len(liquidation_rows)} liquidation rows"
    )

    # Early PDFs (2017, early 2018) genuinely have no resolution/liquidation
    # outcome tables — those only appear once enough CIRPs have concluded.
    # Return an empty DataFrame rather than an error.
    frames = []
    if resolution_rows:
        frames.append(pd.DataFrame(resolution_rows))
    if liquidation_rows:
        frames.append(pd.DataFrame(liquidation_rows))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Ensure schema columns exist
    for col in _SCHEMA_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Drop the internal fair_value column (not in schema)
    df = df[_SCHEMA_COLS]
    return df