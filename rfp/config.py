"""Central configuration: paths, LLM providers and business constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(os.getenv("RFP_DB_PATH", DATA_DIR / "rfp_evaluator_v2.db"))
SAMPLE_DIR = PROJECT_ROOT / "sample_data"

# --- Supplier metadata rules -------------------------------------------------------
EXPERIENCE_RATING_MIN = 1
EXPERIENCE_RATING_MAX = 5

# --- Document handling ------------------------------------------------------------
MAX_DOCUMENT_CHARS = 30_000   # proposal text sent to the LLM (2-4 page PDFs are ~6-12k)
MIN_DOCUMENT_CHARS = 200      # below this a PDF is treated as unreadable / scanned

# --- Criteria rules ---------------------------------------------------------------------
MIN_CRITERIA = 2
MAX_CRITERIA = 10
DEFAULT_MAX_SCORE = 10

# --- LLM providers (all reached through the OpenAI-compatible chat API) ---------------
PROVIDERS = {
    "Cohere": {
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "default_model": "command-a-03-2025",
        "key_hint": "Trial keys are free at dashboard.cohere.com -> API Keys",
        "signup": "https://dashboard.cohere.com/api-keys",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "key_hint": "Create a key at openrouter.ai -> Keys (starts with sk-or-)",
        "signup": "https://openrouter.ai/keys",
    },
    "OpenAI": {
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "key_hint": "Create a key at platform.openai.com -> API keys",
        "signup": "https://platform.openai.com/api-keys",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "key_hint": "Free keys at console.groq.com -> API Keys",
        "signup": "https://console.groq.com/keys",
    },
}
DEFAULT_PROVIDER = "Cohere"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2_500
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_ATTEMPTS = 2          # first call + one JSON-repair retry
MAX_PARALLEL_LLM_CALLS = 3    # Cohere trial keys allow ~20 chat calls per minute

# --- Numeric precision --------------------------------------------------------------
SCORE_DECIMALS = 4            # rounding before sorting: float noise can never change ranks

# --- RAG (question answering over proposals) -------------------------------------
CHUNK_WORDS = 120
CHUNK_OVERLAP_WORDS = 30
RAG_TOP_K_PER_SUPPLIER = 3
