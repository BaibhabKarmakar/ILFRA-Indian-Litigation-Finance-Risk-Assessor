"""
src/parsers/ibbi_pdf.py
------------------------
Parses IBBI quarterly newsletter PDF files.
Uses pdfplumber to extract tables from each page.

Install: pip install pdfplumber
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required: pip install pdfplumber")


def parse(path: Path) -> pd.DataFrame:
    """
    Main entry point — called by ibbi_channel.py for .pdf files.
    Extracts all tables from the PDF and returns rows that look
    like CIRP case data (have financial figures and a company name).
    """
    quarter_label = path.stem
    all_rows = []
    header = None

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            table = page.extract_table()
            if not table or len(table) < 2:
                continue

            # Detect header row — first row with multiple non-empty strings
            if header is None:
                candidate = [str(c).strip() if c else "" for c in table[0]]
                if sum(1 for c in candidate if c) >= 3:
                    header = candidate
                    data_rows = table[1:]
                else:
                    data_rows = table
            else:
                # Check if this page repeats the header (common in PDFs)
                first_row = [str(c).strip().lower() if c else "" for c in table[0]]
                header_lower = [h.lower() for h in header]
                if first_row == header_lower:
                    data_rows = table[1:]
                else:
                    data_rows = table

            all_rows.extend(data_rows)

    if not header or not all_rows:
        print(f"  [ibbi_pdf] No table data found in {path.name}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=header)
    df = df.dropna(how="all")

    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_")
                  .replace("(", "").replace(")", "")
                  for c in df.columns]

    # Try to extract the key fields needed by SCHEMA
    rows = []
    for _, row in df.iterrows():
        try:
            admitted = _to_float(row.get("admitted_claims_cr")
                                  or row.get("amount_of_claims_admitted_cr"))
            realisable = _to_float(row.get("realisable_amount_cr")
                                    or row.get("amount_distributed_cr"))
            if admitted is None or realisable is None:
                continue

            real_pct = round(realisable / admitted * 100, 2) if admitted > 0 else np.nan

            rows.append({
                "company_name":      _to_str(row.get("name_of_cd") or row.get("company_name")),
                "cirp_start_date":   row.get("date_of_commencement") or row.get("cirp_start_date"),
                "resolution_date":   row.get("date_of_approval") or row.get("resolution_date"),
                "cirp_initiated_by": _to_str(row.get("initiated_by")),
                "admitted_claim_cr": admitted,
                "liquidation_value": _to_float(row.get("liquidation_value_cr")),
                "fair_value":        _to_float(row.get("fair_value_cr")),
                "realisable_amount": realisable,
                "realisation_pct":   real_pct,
                "resolution_status": "Resolution Plan Approved",
                "source_table":      "PDF",
                "quarter":           quarter_label,
            })
        except Exception:
            continue

    print(f"  [ibbi_pdf] {path.name} → {len(rows)} rows extracted")
    return pd.DataFrame(rows)


def _to_float(val) -> float | None:
    """Converts a cell value to float, handling commas and dashes."""
    if val is None:
        return None
    s = str(val).strip().replace(",", "").replace("–", "").replace("-", "")
    try:
        return float(s)
    except ValueError:
        return None


def _to_str(val) -> str | None:
    if val is None:
        return None
    return str(val).strip() or None