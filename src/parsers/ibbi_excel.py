"""
src/ingestion/parsers/ibbi_excel.py
------------------------------------
Parses IBBI quarterly newsletter Excel files.
Extracts Table 8 (Resolution Plans) and Table 14 (Closed Liquidations).

This is a format-specific parser — it knows about the internal
structure of IBBI Excel files and nothing else. It returns a
raw DataFrame. All further cleaning happens in ibbi_channel.py.

If IBBI changes their Excel structure in a future quarter,
this is the only file you need to touch.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from src.genai.genai_utils import detect_columns


# ── Table detection helpers ───────────────────────────────────────────────────

def _is_data_row(row: tuple) -> bool:
    """
    A real data row has an integer serial number in column index 1.
    This cleanly skips title rows, header rows, Part A/B label rows,
    and empty rows — all of which have None or a string in that position.
    """
    return isinstance(row[1], int)


def _find_table_sheet(wb, keywords: list[str]) -> str | None:
    """
    Finds a sheet whose name contains any of the given keywords.
    Handles cases where IBBI renames sheets slightly across quarters
    e.g. 'Table 8' vs 'Table8' vs 'Tbl 8'.
    Returns the sheet name or None if not found.
    """
    for name in wb.sheetnames:
        normalised = name.strip().lower().replace(" ", "")
        for kw in keywords:
            if kw.lower().replace(" ", "") in normalised:
                return name
    return None

def _get_col_index(ws, table_name: str) -> dict[str, int]:
    """
    Reads the header row from a sheet and uses GenAI to map headers
    to canonical column names. Returns {canonical_name: col_index}.
    Falls back to empty dict if no headers found.
    """
    headers = []
    header_row_idx = None

    for i, row in enumerate(ws.iter_rows(values_only=True)):
        # Header row has strings in most cells and no integer serial in col 1
        non_empty = [c for c in row if c is not None]
        if len(non_empty) >= 3 and not isinstance(row[1], int):
            headers = [str(c).strip() if c is not None else "" for c in row]
            header_row_idx = i
            break

    if not headers:
        print(f"  [ibbi_excel] Could not find header row in {table_name}")
        return {}

    mapping = detect_columns(headers)  # {raw_header → canonical_name}

    # Invert to {canonical_name → col_index}
    col_index = {}
    for raw_header, canonical in mapping.items():
        if raw_header in headers:
            col_index[canonical] = headers.index(raw_header)

    print(f"  [ibbi_excel] {table_name}: mapped {len(col_index)} columns via GenAI")
    return col_index


# ── Table 8: Resolution Plans ─────────────────────────────────────────────────

def _parse_table8(wb, quarter_label: str) -> pd.DataFrame:
    """
    Extracts individual CIRP cases that ended with an approved
    resolution plan. These are the cases with meaningful realisation %.

    Raw columns from the Excel:
        col 2  — company name
        col 4  — CIRP commencement date
        col 5  — resolution plan approval date
        col 6  — initiated by (FC / OC / CD)
        col 7  — total admitted claims (₹ crore)
        col 8  — liquidation value (₹ crore)
        col 9  — fair value (₹ crore)
        col 10 — total realisable amount by claimants (₹ crore)
        col 11 — realisable as % of admitted claims (pre-calculated)
        col 12 — realisable as % of liquidation value
        col 13 — realisable as % of fair value
    """
    sheet_name = _find_table_sheet(wb, ["Table 8", "Table8", "Resolution Plan"])
    if not sheet_name:
        print(f"  [ibbi_excel] Table 8 not found in {quarter_label} — skipping.")
        return pd.DataFrame()

    ws = wb[sheet_name]
    rows = []

    for row in ws.iter_rows(values_only=True):
        if not _is_data_row(row):
            continue

        admitted   = row[7]
        realisable = row[10]
        real_pct   = row[11]

        # Skip rows where financial fields are dashes or None
        # (some cases report '-' when data is not yet available)
        if not isinstance(admitted, (int, float)):
            continue
        if not isinstance(realisable, (int, float)):
            continue

        admitted   = float(admitted)
        realisable = float(realisable)

        # Use pre-calculated % if available, otherwise compute it
        if isinstance(real_pct, (int, float)):
            realisation_pct = float(real_pct)
        elif admitted > 0:
            realisation_pct = round(realisable / admitted * 100, 2)
        else:
            realisation_pct = np.nan

        rows.append({
            "company_name":      str(row[2]).strip() if row[2] else None,
            "cirp_start_date":   row[4],
            "resolution_date":   row[5],
            "cirp_initiated_by": str(row[6]).strip() if row[6] else None,
            "admitted_claim_cr": admitted,
            "liquidation_value": float(row[8]) if isinstance(row[8], (int, float)) else np.nan,
            "fair_value":        float(row[9]) if isinstance(row[9], (int, float)) else np.nan,
            "realisable_amount": realisable,
            "realisation_pct":   realisation_pct,
            "resolution_status": "Resolution Plan Approved",
            "source_table":      "Table 8",
            "quarter":           quarter_label,
        })

    return pd.DataFrame(rows)


# ── Table 14: Closed Liquidations ─────────────────────────────────────────────

def _parse_table14(wb, quarter_label: str) -> pd.DataFrame:
    """
    Extracts individual CIRP cases that ended in liquidation and
    have been formally closed (dissolution order passed).

    Raw columns from the Excel:
        col 2  — company name
        col 3  — date of liquidation order
        col 4  — admitted claims (₹ crore)
        col 5  — liquidation value (₹ crore)
        col 6  — sale proceeds (₹ crore)
        col 7  — amount distributed to stakeholders (₹ crore)
        col 8  — date of dissolution / closure order

    Note: realisation % is not pre-calculated in Table 14,
    so we compute it as distributed / admitted * 100.
    """
    sheet_name = _find_table_sheet(wb, ["Table 14", "Table14", "Closed Liquidation"])
    if not sheet_name:
        print(f"  [ibbi_excel] Table 14 not found in {quarter_label} — skipping.")
        return pd.DataFrame()

    ws = wb[sheet_name]
    col = _get_col_index(ws, "Table 8")

    # Fall back to hardcoded positions if GenAI mapping failed
    admitted_idx   = col.get("admitted_claim_cr",   7)
    realisable_idx = col.get("realisation_cr",      10)
    real_pct_idx   = col.get("realisation_pct",     11)
    company_idx    = col.get("company_name",         2)
    start_idx      = col.get("admission_date",       4)
    res_date_idx   = col.get("resolution_date",      5)  # approximate — not in canonical but close
    initiated_idx  = col.get("initiated_by",         6)
    liq_val_idx    = col.get("liquidation_value_cr", 8)
    fair_val_idx   = col.get("fair_value_cr",        9)

    rows = []
    for row in ws.iter_rows(values_only=True):
        if not _is_data_row(row):
            continue
        # ... rest of the existing extraction logic, replacing row[7] with row[admitted_idx] etc.


# ── Public API ────────────────────────────────────────────────────────────────

def parse(path: Path) -> pd.DataFrame:
    """
    Main entry point for the Excel parser.
    Called by ibbi_channel.py when it detects a .xlsx file.

    Opens the workbook, extracts Table 8 and Table 14,
    and returns them stacked into a single DataFrame.

    Parameters
    ----------
    path : Path
        Full path to a single IBBI quarterly .xlsx file.

    Returns
    -------
    pd.DataFrame with columns:
        company_name, cirp_start_date, resolution_date,
        cirp_initiated_by, admitted_claim_cr, liquidation_value,
        fair_value, realisable_amount, realisation_pct,
        resolution_status, source_table, quarter
    """
    quarter_label = path.stem

    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception as e:
        print(f"  [ibbi_excel] Could not open {path.name}: {e}")
        return pd.DataFrame()

    t8  = _parse_table8(wb, quarter_label)
    t14 = _parse_table14(wb, quarter_label)

    print(f"  [ibbi_excel] {path.name} → "
          f"{len(t8)} resolution rows, {len(t14)} liquidation rows")

    frames = [f for f in [t8, t14] if not f.empty]
    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)