"""
src/ingestion/parsers/ibbi_pdf.py
----------------------------------
PDF parser for IBBI quarterly newsletters.

STATUS: Stub — not yet implemented.

When implemented, this will:
    - Extract tables from IBBI quarterly PDF newsletters
    - Parse the same Table 8 and Table 14 structure
    - Return a DataFrame in the same schema as ibbi_excel.py

PDF parsing is significantly harder than Excel because:
    - Tables span multiple pages
    - Column alignment relies on whitespace, not cell boundaries
    - Some PDFs are scanned images requiring OCR

Libraries to evaluate when implementing:
    - pdfplumber  — best for structured tables in text PDFs
    - camelot     — good for bordered tables
    - pytesseract — for scanned/image PDFs (OCR)

How to add this:
    1. Implement the parse(path) function below
    2. Make sure it returns a DataFrame with the same
       columns as ibbi_excel.parse() — the channel
       doesn't care which parser ran, only the schema matters.
"""

from pathlib import Path
import pandas as pd


def parse(path: Path) -> pd.DataFrame:
    """
    Parse a single IBBI quarterly PDF file.
    Returns a DataFrame matching the ibbi_excel.py output schema.

    NOT YET IMPLEMENTED.
    """
    raise NotImplementedError(
        f"PDF parsing is not yet implemented.\n"
        f"File: {path.name}\n"
        f"To add PDF support, implement this function in "
        f"src/ingestion/parsers/ibbi_pdf.py"
    )