"""Central configuration: paths, LLM settings and business constants.

Every value can be overridden with an environment variable or a Streamlit
secret of the same name, so nothing sensitive is ever hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(os.getenv("RFP_DB_PATH", DATA_DIR / "rfp_evaluator.db"))

# --- Supplier metadata rules -------------------------------------------------
EXPERIENCE_RATING_MIN = 1
EXPERIENCE_RATING_MAX = 5

# --- Document handling ------------------------------------------------------
# Characters of proposal text sent to the LLM. Synthetic proposals are ~2-4
# pages (~8-12k characters), so this leaves comfortable headroom while
# protecting against very large uploads.
MAX_DOCUMENT_CHARS = 30_000
MIN_DOCUMENT_CHARS = 200  # below this the PDF is treated as unreadable/scanned

# --- LLM providers (all OpenAI-compatible) -----------------------------------
PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_name": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o-mini",
    },
    "OpenAI": {
        "base_url": None,
        "key_name": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_name": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
}
DEFAULT_PROVIDER = "OpenRouter"
LLM_TEMPERATURE = 0.0  # deterministic as possible
LLM_MAX_TOKENS = 2_000
LLM_TIMEOUT_SECONDS = 90
LLM_MAX_ATTEMPTS = 2  # first call + one JSON-repair retry

# --- Numeric precision --------------------------------------------------------
# Scores are rounded before sorting so that floating-point noise can never
# change the ranking (same inputs -> same order, always).
SCORE_DECIMALS = 4


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from env vars first, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:  # Streamlit is optional for scripts/tests
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default
