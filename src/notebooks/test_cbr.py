# save as test_cbr.py at D:\ILFRA root and run: python test_cbr.py
import sys
sys.path.insert(0, r"D:\ILFRA")
from src.inference.predict import load_models, predict_case

import traceback
from src.inference.predict import load_models, predict_case

models = load_models()
case = {
    "case_type": "CIRP (IBC)",
    "sector": "Manufacturing",
    "filing_year": 2021,
    "claim_amount_lakhs": 5000,
    "no_of_financial_creditors": 8,
    "resolution_applicants_received": 2,
    "ip_changed": False,
    "litigation_pending": False,
}

try:
    result = predict_case(case, models)
    cbr = result["cbr"]
    print(f"CBR cases: {len(cbr['similar_cases'])}")
    for c in cbr["similar_cases"]:
        print(f"  {c.case_id} | sim={c.similarity:.2f} | real={c.realisation_pct} | status={c.resolution_status}")
    print("Adapted:", cbr["adapted"])
except Exception:
    traceback.print_exc()