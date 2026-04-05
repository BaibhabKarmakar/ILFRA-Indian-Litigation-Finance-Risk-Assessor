"""
src/genai/__init__.py
---------------------
Public interface for the ILFRA GenAI utilities package.
"""

from .genai_utils import (
    detect_columns,
    normalise_company_names,
    normalise_company_names_batched,
    generate_risk_narrative,
    CANONICAL_COLUMNS,
)

__all__ = [
    "detect_columns",
    "normalise_company_names",
    "normalise_company_names_batched",
    "generate_risk_narrative",
    "CANONICAL_COLUMNS",
]