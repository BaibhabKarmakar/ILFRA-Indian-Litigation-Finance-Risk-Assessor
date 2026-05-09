"""
src/cbr_engine.py
-----------------
Retrieves the K most similar historical cases for a new query
and adapts their outcomes into a CBR prediction.
"""

import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

MODELS_DIR = Path(__file__).parent.parent / "models"

# Feature importance weights — higher = more important to similarity
# These mirror the LightGBM feature importance rankings from training
NJDG_WEIGHTS = {
    "case_type_enc":           3.0,   # most important — case type drives everything
    "court_enc":               2.5,
    "court_hierarchy":         2.0,
    "claimant_lawyer_win_rate": 2.0,
    "log_claim_amount":        1.8,
    "adjournment_density":     1.5,
    "has_interim_order":       1.5,
    "filing_year":             1.2,
    "filing_quarter":          0.8,
    "state_enc":               1.0,
    "sector_enc":              1.0,
    "respondent_is_govt":      1.3,
    "respondent_is_psu":       1.0,
    "represented_by_senior_counsel": 1.2,
    "num_prior_adjournments":  1.0,
    "case_age_months":         1.0,
    "claim_bucket_enc":        1.0,
    "court_avg_duration":      1.5,
    "court_avg_win_rate":      1.5,
}

IBC_WEIGHTS = {
    "resolution_status_enc":       3.0,
    "log_admitted_claim":          2.0,
    "no_of_financial_creditors":   1.8,
    "resolution_applicants_received": 1.8,
    "duration_days":               1.5,
    "ip_changed":                  1.5,
    "litigation_pending":          1.5,
    "applicants_per_creditor":     1.5,
    "sector_enc":                  1.0,
    "bench_enc":                   1.0,
    "admission_year":              1.0,
    "favourable_outcome":          2.0,
}


@dataclass
class SimilarCase:
    """Represents one retrieved similar case."""
    rank: int
    similarity: float          # 0–1, higher = more similar
    case_id: str
    case_type: Optional[str]
    court: Optional[str]
    filing_year: Optional[int]
    claim_amount_lakhs: Optional[float]
    duration_months: Optional[float]
    favourable: Optional[int]
    realisation_pct: Optional[float]
    resolution_status: Optional[str]


class CBREngine:
    def __init__(self):
        self._case_base = None

    def _load(self):
        if self._case_base is None:
            path = MODELS_DIR / "cbr_case_base.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    "cbr_case_base.pkl not found. Run src/cbr_case_base.py first."
                )
            self._case_base = joblib.load(path)

    def _weighted_cosine_similarity(self, query: np.ndarray,
                                     base: np.ndarray,
                                     feature_cols: list,
                                     weight_map: dict) -> np.ndarray:
        """
        Computes weighted cosine similarity between a query vector and
        every row in the case base matrix.

        Weighted cosine: instead of treating all dimensions equally,
        multiply each dimension by its domain importance weight before
        computing the dot product. This means case_type mismatch hurts
        similarity much more than filing_quarter mismatch.
        """
        weights = np.array(
            [weight_map.get(col, 1.0) for col in feature_cols],
            dtype=np.float32
        )
        # Apply weights to both query and base
        q_w = query * weights                          # shape: (n_features,)
        b_w = base * weights                           # shape: (n_cases, n_features)

        dot   = b_w @ q_w                              # shape: (n_cases,)
        norm_q = np.linalg.norm(q_w) + 1e-9
        norm_b = np.linalg.norm(b_w, axis=1) + 1e-9  # shape: (n_cases,)

        return dot / (norm_b * norm_q)                 # cosine similarity per case

    def retrieve_njdg(self, query_features: np.ndarray,
                      feature_cols: list, k: int = 5) -> list[SimilarCase]:
        self._load()
        base = self._case_base["njdg"]

        sims = self._weighted_cosine_similarity(
            query_features, base["features"], feature_cols, NJDG_WEIGHTS
        )
        top_k = np.argsort(sims)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_k):
            results.append(SimilarCase(
                rank=rank + 1,
                similarity=float(sims[idx]),
                case_id=str(base["case_id"][idx]) if base["case_id"] is not None else f"CASE-{idx}",
                case_type=str(base["case_type"][idx]) if base["case_type"] is not None else None,
                court=str(base["court"][idx]) if base["court"] is not None else None,
                filing_year=int(base["filing_year"][idx]) if base["filing_year"] is not None else None,
                claim_amount_lakhs=float(base["claim_amount"][idx]) if base["claim_amount"] is not None else None,
                duration_months=float(base["duration_months"][idx]),
                favourable=int(base["favourable"][idx]),
                realisation_pct=None,
                resolution_status=None,
            ))
        return results

    def retrieve_ibc(self, query_features: np.ndarray,
                     feature_cols: list, k: int = 5) -> list[SimilarCase]:
        self._load()
        base = self._case_base["ibc"]

        # Filter to resolved cases only — ongoing cases have no outcome to reuse
        resolved_mask = ~np.isnan(base["realisation_pct"])
        filtered_features = base["features"][resolved_mask]
        filtered_idx      = np.where(resolved_mask)[0]

        sims = self._weighted_cosine_similarity(
            query_features, filtered_features, feature_cols, IBC_WEIGHTS
        )
        top_k = np.argsort(sims)[::-1][:k]

        results = []
        for rank, local_idx in enumerate(top_k):
            idx = filtered_idx[local_idx]
            results.append(SimilarCase(
                rank=rank + 1,
                similarity=float(sims[local_idx]),
                case_id=str(base["cirp_id"][idx]) if base["cirp_id"] is not None else f"IBC-{idx}",
                case_type="CIRP (IBC)",
                court=None,
                filing_year=None,
                claim_amount_lakhs=None,
                duration_months=float(base["duration_days"][idx] / 30) if base["duration_days"] is not None else None,
                favourable=None,
                realisation_pct=float(base["realisation_pct"][idx]),
                resolution_status=str(base["resolution_status"][idx]) if base["resolution_status"] is not None else None,
            ))
        return results

    def adapt(self, similar_cases: list[SimilarCase]) -> dict:
        """
        Derives CBR-adapted outcome estimates from retrieved cases.
        Uses similarity-weighted averaging — cases with higher similarity
        contribute more to the adapted estimate than distant ones.
        """
        if not similar_cases:
            return {}

        sims = np.array([c.similarity for c in similar_cases])
        weights = sims / (sims.sum() + 1e-9)   # normalise to sum to 1

        result = {}

        # Duration — weighted mean and std across retrieved cases
        durations = [c.duration_months for c in similar_cases if c.duration_months is not None]
        if durations:
            d = np.array(durations)
            result["cbr_duration_months"] = float(np.average(d, weights=weights[:len(d)]))
            result["cbr_duration_std"]    = float(np.std(d))

        # Outcome — similarity-weighted win rate
        outcomes = [c.favourable for c in similar_cases if c.favourable is not None]
        if outcomes:
            o = np.array(outcomes, dtype=float)
            result["cbr_p_favourable"] = float(np.average(o, weights=weights[:len(o)]))

        # Realisation — weighted mean recovery %
        reals = [c.realisation_pct for c in similar_cases if c.realisation_pct is not None]
        if reals:
            r = np.array(reals)
            result["cbr_realisation_pct"] = float(np.average(r, weights=weights[:len(r)]))
            result["cbr_realisation_std"] = float(np.std(r))

        return result


# Singleton — loaded once, reused across predictions
_engine = CBREngine()

def get_engine() -> CBREngine:
    return _engine