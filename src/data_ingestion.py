"""
src/data_ingestion.py
---------------------
Handles:
  1. Downloading / caching public data from NJDG, IBBI, eCourts
  2. Generating a realistic synthetic dataset for development when
     live downloads are unavailable (offline mode).

Real usage notes:
  - NJDG CSV exports: navigate to njdg.ecourts.gov.in → Reports →
    Download CSV for district/high courts filtered by case type.
  - IBBI resolution data: ibbi.gov.in → Data → CIRP Data (Excel/CSV).
  - Drop downloaded files into data/raw/ and set USE_SYNTHETIC=False.
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants mirroring NJDG / IBBI field vocabulary ─────────────────────────

CASE_TYPES = [
    "Civil Suit", "Money Recovery", "Injunction", "Partition",
    "Specific Performance", "Arbitration", "Commercial Dispute",
    "CIRP (IBC)", "Liquidation (IBC)", "Writ (HC)", "Appeal (HC)",
    "Consumer Dispute", "Labour / Employment", "IP Infringement",
]

COURTS = {
    "District Court": {"hierarchy": 1, "avg_disposal_months": 48},
    "Commercial Court": {"hierarchy": 2, "avg_disposal_months": 24},
    "High Court (Original)": {"hierarchy": 3, "avg_disposal_months": 36},
    "High Court (Appeal)": {"hierarchy": 3, "avg_disposal_months": 42},
    "NCLT": {"hierarchy": 2, "avg_disposal_months": 30},
    "NCLAT": {"hierarchy": 3, "avg_disposal_months": 18},
    "Supreme Court": {"hierarchy": 4, "avg_disposal_months": 60},
    "Consumer Forum (District)": {"hierarchy": 1, "avg_disposal_months": 20},
    "Consumer Forum (State)": {"hierarchy": 2, "avg_disposal_months": 28},
}

STATES = [
    "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana",
    "Gujarat", "West Bengal", "Rajasthan", "Uttar Pradesh", "Punjab",
]

SECTORS = [
    "Real Estate", "Banking & Finance", "Infrastructure", "Manufacturing",
    "IT / Technology", "Healthcare", "Retail", "Telecom", "Energy", "Others",
]

DISPOSAL_CATEGORIES = [
    "Decree in favour of plaintiff",
    "Decree in favour of defendant",
    "Settlement / Compromise",
    "Dismissed for default",
    "Dismissed on merits",
    "Withdrawn",
    "Transferred",
]

# ── Synthetic data generator ──────────────────────────────────────────────────

def _sample_outcome(case_type: str, claimant_lawyer_win_rate: float) -> dict:
    """Return simulated outcome fields based on loose domain heuristics."""
    ibc = "IBC" in case_type or "Liquidation" in case_type
    base_p_favour = claimant_lawyer_win_rate * 0.7 + 0.1
    if case_type == "Money Recovery":
        base_p_favour += 0.10
    if case_type in ("Writ (HC)", "Appeal (HC)"):
        base_p_favour -= 0.05
    base_p_favour = float(np.clip(base_p_favour, 0.10, 0.90))

    favourable = random.random() < base_p_favour
    disposal_cat = (
        random.choice([DISPOSAL_CATEGORIES[0], "Settlement / Compromise"])
        if favourable
        else random.choice(DISPOSAL_CATEGORIES[1:])
    )

    # Realisation % — only meaningful for IBC / money recovery
    if ibc:
        realisation_pct = (
            np.random.beta(2, 3) * 60 + 10 if favourable  # 10–70 %
            else np.random.beta(1, 4) * 15              # 0–15 %
        )
    elif case_type == "Money Recovery":
        realisation_pct = (
            np.random.beta(3, 2) * 80 + 20 if favourable
            else np.random.beta(1, 3) * 30
        )
    else:
        realisation_pct = np.nan  # not applicable

    return {
        "favourable_outcome": int(favourable),
        "disposal_category": disposal_cat,
        "realisation_pct": round(realisation_pct, 2) if not np.isnan(realisation_pct) else np.nan,
    }


def generate_synthetic_njdg(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the field structure you would
    get after joining NJDG case metadata with eCourts judgment outcomes.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    rows = []

    for i in range(n):
        court_name = rng.choice(list(COURTS.keys()))
        court_meta = COURTS[court_name]
        case_type = rng.choice(CASE_TYPES)

        filing_date = datetime(2010, 1, 1) + timedelta(
            days=rng.randint(0, 365 * 12)  # 2010–2022
        )

        # Duration: log-normal centred on court average
        mu = np.log(court_meta["avg_disposal_months"])
        duration_months = int(np.clip(np.random.lognormal(mu, 0.5), 1, 240))
        disposal_date = filing_date + timedelta(days=duration_months * 30)

        # Claim amount (INR lakhs)
        claim_amount_lakhs = round(np.random.lognormal(4, 1.5), 2)  # ~1L to 10Cr

        # Lawyer / party features
        claimant_lawyer_win_rate = round(rng.uniform(0.25, 0.80), 2)
        respondent_is_govt = rng.random() < 0.15
        respondent_is_psu = rng.random() < 0.10
        num_prior_adjournments = rng.randint(0, 40)
        has_interim_order = rng.random() < 0.35
        represented_by_senior_counsel = rng.random() < 0.20

        outcome = _sample_outcome(case_type, claimant_lawyer_win_rate)

        rows.append({
            "case_id": f"SYNTH-{i:05d}",
            "case_type": case_type,
            "court": court_name,
            "court_hierarchy": court_meta["hierarchy"],
            "state": rng.choice(STATES),
            "sector": rng.choice(SECTORS),
            "filing_date": filing_date.date(),
            "disposal_date": disposal_date.date(),
            "duration_months": duration_months,
            "claim_amount_lakhs": claim_amount_lakhs,
            "claimant_lawyer_win_rate": claimant_lawyer_win_rate,
            "respondent_is_govt": int(respondent_is_govt),
            "respondent_is_psu": int(respondent_is_psu),
            "num_prior_adjournments": num_prior_adjournments,
            "has_interim_order": int(has_interim_order),
            "represented_by_senior_counsel": int(represented_by_senior_counsel),
            **outcome,
        })

    df = pd.DataFrame(rows)
    print(f"[data_ingestion] Generated {len(df)} synthetic cases.")
    return df


def generate_synthetic_ibc(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic IBBI CIRP resolution dataset matching fields available
    from ibbi.gov.in Data → CIRP Data Excel downloads.
    """
    rng = random.Random(seed + 1)
    np.random.seed(seed + 1)
    rows = []

    resolution_statuses = [
        "Resolution Plan Approved", "Liquidation Order", "Withdrawn (Sec 12A)",
        "Appeal Pending", "Settled", "Ongoing"
    ]

    for i in range(n):
        admission_date = datetime(2017, 1, 1) + timedelta(
            days=rng.randint(0, 365 * 6)
        )
        status = rng.choice(resolution_statuses)
        completed = status in ("Resolution Plan Approved", "Liquidation Order",
                               "Withdrawn (Sec 12A)", "Settled")

        duration_days = (
            rng.randint(60, 1100) if completed else rng.randint(30, 600)
        )

        admitted_claim_cr = round(np.random.lognormal(4, 1.5), 2)

        if status == "Resolution Plan Approved":
            realisation_pct = round(np.random.beta(2, 3) * 60 + 10, 2)
        elif status == "Liquidation Order":
            realisation_pct = round(np.random.beta(1, 5) * 20, 2)
        elif status in ("Withdrawn (Sec 12A)", "Settled"):
            realisation_pct = round(np.random.beta(3, 2) * 70 + 20, 2)
        else:
            realisation_pct = np.nan

        rows.append({
            "cirp_id": f"CIRP-{i:05d}",
            "sector": rng.choice(SECTORS),
            "bench": rng.choice(["NCLT Mumbai", "NCLT Delhi", "NCLT Kolkata",
                                  "NCLT Hyderabad", "NCLT Ahmedabad", "NCLT Chennai"]),
            "admission_date": admission_date.date(),
            "resolution_status": status,
            "duration_days": duration_days,
            "admitted_claim_cr": admitted_claim_cr,
            "realisation_pct": realisation_pct,
            "no_of_financial_creditors": rng.randint(1, 150),
            "resolution_applicants_received": rng.randint(0, 20),
            "ip_changed": int(rng.random() < 0.15),
            "litigation_pending": int(rng.random() < 0.30),
            "favourable_outcome": int(status in ("Resolution Plan Approved",
                                                   "Withdrawn (Sec 12A)", "Settled")),
        })

    df = pd.DataFrame(rows)
    print(f"[data_ingestion] Generated {len(df)} synthetic IBBI CIRP records.")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    njdg = generate_synthetic_njdg()
    ibc = generate_synthetic_ibc()

    njdg.to_csv(RAW_DIR / "njdg_synthetic.csv", index=False)
    ibc.to_csv(RAW_DIR / "ibbi_synthetic.csv", index=False)
    print(f"[data_ingestion] Saved raw data to {RAW_DIR}")


if __name__ == "__main__":
    main()