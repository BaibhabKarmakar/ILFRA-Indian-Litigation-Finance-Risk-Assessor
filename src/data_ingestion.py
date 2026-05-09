"""
src/data_ingestion.py
----------------------
Main entry point for ILFRA data ingestion.

This file is now a thin wrapper — all actual ingestion logic
lives in src/ingestion/ channels.

To ingest data:
    python src/data_ingestion.py

What it does:
    - Runs the IBBI channel (reads from data/raw/ibbi/)
    - Outputs data/raw/ibbi_real.csv

To add a new source (e.g. NJDG) in the future:
    1. Create src/ingestion/njdg_channel.py
    2. Import and call it here
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.ibbi_channel import run as run_ibbi


def main():
    # IBBI channel — reads all files from data/raw/ibbi/
    run_ibbi()

    # Future channels can be added here:
    # run_njdg()
    # run_ecourts()


if __name__ == "__main__":
    main()