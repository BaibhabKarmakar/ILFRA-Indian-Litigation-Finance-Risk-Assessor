"""
src/cbr_case_base.py
--------------------
Builds the CBR case base from processed feature data.
Run once after feature_engineering.py.

Output:
    models/cbr_case_base.pkl   — dict with feature matrix, outcomes, metadata
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.feature_engineering import get_feature_cols, get_ibc_feature_cols

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"

def build_case_base():
    njdg = pd.read_csv(PROCESSED_DIR / "njdg_features.csv")
    ibc  = pd.read_csv(PROCESSED_DIR / "ibc_features.csv")

    fc_njdg = get_feature_cols()
    fc_ibc  = get_ibc_feature_cols()

    # NJDG case base
    njdg_base = {
        "features":       njdg[fc_njdg].fillna(0).values.astype(np.float32),
        "feature_cols":   fc_njdg,
        "duration_months": njdg["duration_months"].values,
        "favourable":     njdg["favourable_outcome"].values,
        "case_type":      njdg["case_type"].values   if "case_type" in njdg.columns else None,
        "court":          njdg["court"].values        if "court" in njdg.columns else None,
        "claim_amount":   njdg["claim_amount_lakhs"].values if "claim_amount_lakhs" in njdg.columns else None,
        "filing_year":    njdg["filing_year"].values  if "filing_year" in njdg.columns else None,
        "case_id":        njdg["case_id"].values      if "case_id" in njdg.columns else None,
    }

    # IBC case base
    ibc_base = {
        "features":         ibc[fc_ibc].fillna(0).values.astype(np.float32),
        "feature_cols":     fc_ibc,
        "realisation_pct":  ibc["realisation_pct"].values,
        "resolution_status": ibc["resolution_status"].values if "resolution_status" in ibc.columns else None,
        "duration_days":    ibc["duration_days"].values if "duration_days" in ibc.columns else None,
        "sector":           ibc["sector"].values if "sector" in ibc.columns else None,
        "cirp_id":          ibc["cirp_id"].values if "cirp_id" in ibc.columns else None,
    }

    case_base = {"njdg": njdg_base, "ibc": ibc_base}
    joblib.dump(case_base, MODELS_DIR / "cbr_case_base.pkl")
    print(f"[cbr] Case base built: {len(njdg)} NJDG cases, {len(ibc)} IBC cases")
    return case_base


if __name__ == "__main__":
    build_case_base()