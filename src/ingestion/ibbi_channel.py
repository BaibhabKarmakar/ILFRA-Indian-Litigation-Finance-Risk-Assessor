"""
src/ingestion/ibbi_channel.py
------------------------------
IBBI data channel — the orchestrator.

This is the single entry point for all IBBI data regardless of
file format. It:
    1. Scans the IBBI source folder for files
    2. Detects each file's format (.xlsx, .pdf, etc.)
    3. Routes to the correct parser
    4. Combines all parsed data into one DataFrame
    5. Deduplicates across quarters
    6. Derives features
    7. Saves to data/raw/ibbi_real.csv

If IBBI releases data in a new format in the future,
add a new parser in parsers/ and register it in PARSERS below.
No other file needs to change.

Usage:
    python src/ingestion/ibbi_channel.py

    Or programmatically:
        from src.ingestion.ibbi_channel import run
        df = run()
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from pathlib import Path

from src.ingestion.parsers import ibbi_excel, ibbi_pdf

# ── Config ────────────────────────────────────────────────────────────────────

# Drop your downloaded IBBI quarterly files here
IBBI_SOURCE_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "ibbi"

# Final cleaned output
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "ibbi_real.csv"

# ── Format registry ───────────────────────────────────────────────────────────
# Maps file extension → parser module.
# To add a new format, add one line here and create the parser file.

PARSERS = {
    ".xlsx": ibbi_excel,
    ".xls":  ibbi_excel,   # same parser handles legacy xls
    ".pdf":  ibbi_pdf,
}

# ── Unified output schema ─────────────────────────────────────────────────────
# Every parser must return a DataFrame with exactly these columns.
# Any extra columns from a parser are dropped here.

SCHEMA = [
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
    "source_table",
    "quarter",
]


# ── Pipeline steps ────────────────────────────────────────────────────────────

def _scan_files(source_dir: Path) -> dict[str, list[Path]]:
    """
    Scans source_dir for files with known extensions.
    Returns a dict mapping extension → list of file paths.
    Ignores files with unsupported extensions and prints a warning.
    """
    found = {ext: [] for ext in PARSERS}
    unknown = []

    for path in sorted(source_dir.iterdir()):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext in PARSERS:
            found[ext].append(path)
        elif path.name != ".gitkeep":   # ignore placeholder files
            unknown.append(path.name)

    if unknown:
        print(f"[ibbi_channel] Skipping unsupported files: {unknown}")

    total = sum(len(v) for v in found.values())
    print(f"[ibbi_channel] Found {total} file(s) in {source_dir}")
    for ext, files in found.items():
        if files:
            print(f"  {ext}: {len(files)} file(s)")

    return found


def _route_and_parse(found: dict[str, list[Path]]) -> pd.DataFrame:
    """
    Routes each file to its parser and collects all results.
    Files that fail to parse are skipped with a warning.
    """
    all_frames = []

    for ext, paths in found.items():
        if not paths:
            continue

        parser = PARSERS[ext]

        for path in paths:
            print(f"\n[ibbi_channel] Parsing: {path.name} ({ext})")
            try:
                df = parser.parse(path)
                if df.empty:
                    print(f"  ⚠ No data extracted from {path.name}")
                    continue
                all_frames.append(df)
            except NotImplementedError as e:
                print(f"  ⚠ {e}")
            except Exception as e:
                print(f"  ✗ Failed to parse {path.name}: {e}")

    if not all_frames:
        raise ValueError(
            "No data extracted from any file. "
            "Check that your files are valid IBBI quarterly newsletters."
        )

    combined = pd.concat(all_frames, ignore_index=True)

    # Enforce schema — keep only known columns, in order
    for col in SCHEMA:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[SCHEMA]

    return combined


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each quarterly file contains a 'Part A: Prior Period' section
    repeating cases from earlier quarters. This removes them by
    keeping the most recent record per unique case.

    Key: company_name + cirp_start_date — IBBI doesn't publish
    unique case IDs in the Excel files.
    """
    before = len(df)
    df = df.drop_duplicates(
        subset=["company_name", "cirp_start_date"],
        keep="last"
    )
    after = len(df)
    removed = before - after
    if removed:
        print(f"[ibbi_channel] Dedup: removed {removed} duplicate rows "
              f"({after} unique cases remain)")
    return df


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives additional ML-ready features from raw extracted columns.
    All new columns are appended — original columns are preserved.
    """
    df = df.copy()

    # Parse dates
    df["cirp_start_date"] = pd.to_datetime(df["cirp_start_date"], errors="coerce")
    df["resolution_date"] = pd.to_datetime(df["resolution_date"],  errors="coerce")

    # Duration
    df["duration_days"] = (df["resolution_date"] - df["cirp_start_date"]).dt.days

    # Outcome flag — 1 = resolution (good for creditors), 0 = liquidation
    df["favourable_outcome"] = (
        df["resolution_status"] == "Resolution Plan Approved"
    ).astype(int)

    # Log-scale claim amount (spans several orders of magnitude)
    df["log_admitted_claim"] = np.log1p(df["admitted_claim_cr"])

    # Admission year
    df["admission_year"] = df["cirp_start_date"].dt.year

    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final quality checks before saving.
        - Drops rows missing the three critical ML fields
        - Clips realisation % to [0, 150] — anything above 150% is likely a data error
        - Clips duration to positive values
    """
    before = len(df)

    df = df.dropna(subset=["admitted_claim_cr", "realisation_pct", "duration_days"])
    df = df[df["duration_days"] > 0]
    df["realisation_pct"] = df["realisation_pct"].clip(0, 150)

    dropped = before - len(df)
    if dropped:
        print(f"[ibbi_channel] Validation: dropped {dropped} rows with "
              f"missing/invalid critical fields")

    return df.reset_index(drop=True)


def _print_summary(df: pd.DataFrame) -> None:
    """Prints a quick summary of the final dataset."""
    print("\n── IBBI Dataset Summary ──────────────────────────────")
    print(f"  Total cases        : {len(df)}")
    print(f"  Resolution cases   : {(df['resolution_status'] == 'Resolution Plan Approved').sum()}")
    print(f"  Liquidation cases  : {(df['resolution_status'] == 'Liquidation Order').sum()}")
    print(f"  Realisation % avg  : {df['realisation_pct'].mean():.1f}%")
    print(f"  Realisation % range: {df['realisation_pct'].min():.1f}% "
          f"— {df['realisation_pct'].max():.1f}%")
    print(f"  Duration range     : {df['duration_days'].min():.0f} "
          f"— {df['duration_days'].max():.0f} days")
    print(f"  Year range         : {int(df['admission_year'].min())} "
          f"— {int(df['admission_year'].max())}")
    print(f"  Quarters covered   : {sorted(df['quarter'].unique())}")
    print(f"\n  Output columns     : {list(df.columns)}")


# ── Public API ────────────────────────────────────────────────────────────────

def run(source_dir: Path = IBBI_SOURCE_DIR,
        output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """
    Runs the full IBBI ingestion pipeline.

    Parameters
    ----------
    source_dir  : folder containing IBBI quarterly files (any supported format)
    output_path : where to save ibbi_real.csv

    Returns
    -------
    pd.DataFrame — the cleaned, feature-enriched dataset
    """
    if not source_dir.exists():
        raise FileNotFoundError(
            f"IBBI source folder not found: {source_dir}\n"
            f"Create it and drop your IBBI quarterly files inside."
        )

    print("=" * 60)
    print("ILFRA — IBBI Channel")
    print("=" * 60)

    # 1. Scan
    found = _scan_files(source_dir)

    # 2. Parse
    df = _route_and_parse(found)

    # 2b. Normalise company names across quarters
    from src.genai.genai_utils import normalise_company_names_batched
    print("\n[ibbi_channel] Normalising company names...")
    name_map = normalise_company_names_batched(df["company_name"].dropna().tolist())
    df["company_name"] = df["company_name"].map(name_map).fillna(df["company_name"])
    
    # 3. Deduplicate
    df = _deduplicate(df)
    

    # 4. Derive features
    df = _derive_features(df)

    # 5. Validate
    df = _validate_and_clean(df)

    # 6. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[ibbi_channel] Saved → {output_path}")

    # 7. Summary
    _print_summary(df)

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()