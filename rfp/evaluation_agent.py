"""Evaluation Agent: asks an LLM to score ONE supplier against the active criteria.

Responsibilities
* Build an evidence-grounded prompt from the *current* criteria (nothing about
  the criteria is hard-coded in the prompt).
* Call any OpenAI-compatible LLM (OpenRouter, OpenAI, Groq) in JSON mode with
  temperature 0.
* If the reply is not parseable JSON, send one "repair" request.
* Cache replies by hash(document + criteria + model + prompt version) so that
  re-running the same inputs returns the same scorecard and costs nothing.

The agent only *judges content*. It never computes totals, benchmarks or
ranks - that is the Ranking Tool's job.

An OFFLINE DEMO evaluator is also provided. It is a transparent keyword
heuristic that lets the app run without an API key (for demos, CI tests and
the validation-case demonstration). It is clearly labelled in the UI and must
not be used for real procurement decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from . import config, db
from .validation_tool import parse_llm_json

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """You are a senior procurement evaluator on an RFP evaluation panel.
You score ONE supplier proposal at a time against the evaluation criteria you are given.

Rules you must follow:
1. Use ONLY evidence that is present in the supplier document. Do not use outside knowledge,
   assumptions about the supplier, or information about other suppliers.
2. Return exactly one result for EVERY criterion listed, using its criterion_id.
3. Each score must be an integer between 0 and that criterion's max_score.
   If the document gives no evidence for a criterion, score it low (0-2) and say the information is missing.
4. "evidence" must be a short VERBATIM quote (max 40 words) copied from the document that supports the score.
   If there is no supporting text, set evidence to "" (empty string).
5. "justification" explains the score in 1-3 sentences, mentioning strengths AND gaps.
6. List the main delivery, commercial or compliance risks you see in "risks".
7. Output a single JSON object only - no markdown, no commentary.

Scoring guide (scale to each criterion's max_score; shown here for a max of 10):
9-10 exceptional, specific and well evidenced | 7-8 strong with minor gaps |
5-6 adequate but generic or partly evidenced | 3-4 weak or vague | 0-2 missing or contradictory."""

USER_TEMPLATE = """EVALUATION CRITERIA (score every one):
{criteria_block}

REQUIRED JSON FORMAT:
{{
  "supplier_name": "{supplier_name}",
  "criteria": [
    {{"criterion_id": <int>, "score": <int>, "max_score": <int>, "justification": "<text>", "evidence": "<verbatim quote>"}}
  ],
  "risks": ["<risk>", "..."],
  "overall_summary": "<2-3 sentence summary>"
}}

SUPPLIER: {supplier_name}
SUPPLIER DOCUMENT (text extracted from PDF):
<<<DOCUMENT_START
{document_text}
DOCUMENT_END>>>

Return the JSON object now."""

REPAIR_TEMPLATE = """Your previous reply could not be parsed as the required JSON object.
Problem: {error}
Return ONLY the corrected JSON object, with one entry for each criterion_id in {ids}. No other text."""


@dataclass
class EvaluationResponse:
    raw_text: str
    model: str
    from_cache: bool = False
    attempts: int = 1
    error: str = ""
    notes: list[str] = field(default_factory=list)


def criteria_block(criteria: list[dict]) -> str:
    lines = []
    for c in criteria:
        lines.append(
            f"- criterion_id {c['criterion_id']}: {c['name']} (max_score {c['max_score']}, weight {c['weight']:g}%)\n"
            f"  What to inspect: {c['description']}"
        )
    return "\n".join(lines)


def build_messages(supplier_name: str, document_text: str, criteria: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                criteria_block=criteria_block(criteria),
                supplier_name=supplier_name,
                document_text=document_text,
            ),
        },
    ]


def cache_key(document_sha: str, criteria: list[dict], model: str) -> str:
    """Identical document + identical criteria + same model -> identical scorecard.

    The supplier name is deliberately NOT part of the key: the evaluation depends only on
    the proposal content, so the same PDF can never receive two different scores.
    """
    crit = json.dumps(
        [(c["criterion_id"], c["name"], c["description"], c["max_score"]) for c in criteria], sort_keys=True
    )
    payload = f"{PROMPT_VERSION}|{model}|{document_sha}|{crit}"
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMEvaluator:
    """Scores proposals with a real LLM via an OpenAI-compatible API."""

    def __init__(self, provider: str, api_key: str, model: str | None = None, use_cache: bool = True, db_path=None):
        from openai import OpenAI

        settings = config.PROVIDERS[provider]
        self.provider = provider
        self.model = model or settings["default_model"]
        self.use_cache = use_cache
        self.db_path = db_path
        self.client = OpenAI(
            api_key=api_key,
            base_url=settings["base_url"],
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=2,  # transient network/rate-limit retries handled by the SDK
        )

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def _call(self, messages: list[dict]) -> str:
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        try:
            resp = self.client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except Exception as exc:
            # Some models/providers do not support JSON mode - fall back to prompt-only JSON.
            if "response_format" in str(exc) or "json" in str(exc).lower():
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""

    def evaluate(self, supplier_name: str, document_text: str, document_sha: str, criteria: list[dict]) -> EvaluationResponse:
        key = cache_key(document_sha, criteria, self.label)
        if self.use_cache:
            cached = db.cache_get(key, self.db_path)
            if cached:
                return EvaluationResponse(raw_text=cached, model=self.label, from_cache=True, attempts=0)

        messages = build_messages(supplier_name, document_text, criteria)
        ids = [c["criterion_id"] for c in criteria]
        raw, error = "", ""
        for attempt in range(1, config.LLM_MAX_ATTEMPTS + 1):
            try:
                raw = self._call(messages)
            except Exception as exc:
                return EvaluationResponse(raw_text="", model=self.label, attempts=attempt, error=f"LLM call failed: {exc}")
            _, error = parse_llm_json(raw)
            if not error:
                if self.use_cache:
                    db.cache_put(key, self.label, raw, self.db_path)
                return EvaluationResponse(raw_text=raw, model=self.label, attempts=attempt)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_TEMPLATE.format(error=error, ids=ids)},
            ]
        # Still invalid after repair - hand the raw text to the Validation Tool, which will
        # normalise it (missing criteria -> 0) and record warnings.
        return EvaluationResponse(raw_text=raw, model=self.label, attempts=config.LLM_MAX_ATTEMPTS, error=error)


# ---------------------------------------------------------------------------------
# Offline demo evaluator (no API key needed)
# ---------------------------------------------------------------------------------

_DEMO_KEYWORDS = {
    "technical": ["architecture", "api", "integration", "scalab", "microservice", "cloud", "uptime",
                  "availability", "load", "performance", "kubernetes", "sso", "data model", "rest"],
    "implementation": ["milestone", "week", "phase", "timeline", "project manager", "governance", "risk",
                       "mitigation", "go-live", "uat", "hypercare", "staffing", "team", "raid"],
    "commercial": ["inr", "price", "pricing", "total", "licen", "assumption", "exclusion", "payment",
                   "fixed", "cost", "discount", "gst", "year"],
    "security": ["iso 27001", "soc 2", "encrypt", "gdpr", "dpdp", "audit", "penetration", "access control",
                 "mfa", "privacy", "compliance", "certif", "vapt", "log"],
    "support": ["support", "sla", "24x7", "reference", "client", "similar", "years", "helpdesk",
                "response time", "account manager", "case study", "customer"],
}


def _demo_bucket(criterion: dict) -> list[str]:
    text = f"{criterion['name']} {criterion['description']}".lower()
    for bucket, words in _DEMO_KEYWORDS.items():
        if bucket in text or any(w in text for w in words[:3]):
            return words
    # Unknown criterion: use the longer words of its own description as keywords.
    return [w for w in re.findall(r"[a-z]{6,}", text)][:12]


def _best_sentence(text: str, keywords: list[str]) -> str:
    flat = re.sub(r"\[Page \d+\]", " ", text)
    flat = re.sub(r"(?<![.:!?])\n(?!\n)", " ", flat)  # re-join wrapped lines
    sentences = re.split(r"(?<=[.!?])\s+|\n", flat)
    best, best_hits = "", 0
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 25:
            continue
        hits = sum(k in s_clean.lower() for k in keywords)
        if hits > best_hits:
            best, best_hits = s_clean, hits
    words = best.split()
    return " ".join(words[:40])


class OfflineDemoEvaluator:
    """Deterministic keyword heuristic. Clearly NOT an AI judgement."""

    label = "OFFLINE_DEMO (keyword heuristic - no LLM)"
    model = label

    def __init__(self, simulate_malformed_for: set[str] | None = None):
        # Supplier names for which we deliberately return a broken response,
        # used to demonstrate the Validation Tool.
        self.simulate_malformed_for = {s.lower() for s in (simulate_malformed_for or set())}

    def evaluate(self, supplier_name: str, document_text: str, document_sha: str, criteria: list[dict]) -> EvaluationResponse:
        lower = document_text.lower()
        results = []
        for c in criteria:
            kws = _demo_bucket(c)
            distinct = sum(1 for k in kws if k in lower)
            total = sum(lower.count(k) for k in kws)
            raw_score = 1 + 0.55 * distinct + 0.08 * min(total, 40)
            score = int(max(0, min(c["max_score"], round(raw_score * c["max_score"] / 10))))
            evidence = _best_sentence(document_text, kws)
            results.append(
                {
                    "criterion_id": c["criterion_id"],
                    "score": score,
                    "max_score": c["max_score"],
                    "justification": f"Offline heuristic: {distinct} of {len(kws)} indicator terms found "
                    f"({total} mentions) for {c['name']}.",
                    "evidence": evidence,
                }
            )
        payload = {
            "supplier_name": supplier_name,
            "criteria": results,
            "risks": ["Offline demo mode: risks are not assessed without an LLM."],
            "overall_summary": "Scored by the offline keyword heuristic for demonstration only.",
        }
        if supplier_name.lower() in self.simulate_malformed_for:
            return EvaluationResponse(raw_text=_malformed_version(payload, criteria), model=self.label,
                                      notes=["Simulated malformed LLM output (validation demo)."])
        return EvaluationResponse(raw_text=json.dumps(payload), model=self.label)


def _malformed_version(payload: dict, criteria: list[dict]) -> str:
    """Break a valid scorecard in several realistic ways for the validation demo."""
    crit = payload["criteria"]
    broken = []
    if crit:
        first = dict(crit[0]); first["score"] = 14  # out of range -> clipped
        broken.append(first)
    if len(crit) > 1:
        second = dict(crit[1]); second["score"] = "eight"  # non-numeric -> 0
        broken.append(second)
    if len(crit) > 2:
        third = dict(crit[2]); third["score"] = -3; third["evidence"] = "This sentence does not appear anywhere."
        broken.append(third)
    broken.append({"criterion_id": 999, "score": 9, "justification": "Unknown criterion", "evidence": ""})
    # remaining criteria deliberately omitted -> filled with 0 + warning
    body = json.dumps({"supplier_name": payload["supplier_name"], "criteria": broken})
    return "Sure! Here is the evaluation you asked for:\n```json\n" + body + "\n```"


class FaultInjectingEvaluator:
    """Wraps any evaluator and corrupts the reply for chosen suppliers.

    Used only from the UI's "Testing tools" panel to demonstrate that the
    Validation Tool catches malformed, out-of-range, missing and hallucinated
    LLM output before it can affect the ranking.
    """

    def __init__(self, inner, supplier_names: set[str]):
        self.inner = inner
        self.names = {n.lower() for n in supplier_names}
        self.label = f"{inner.label} + simulated faults"
        self.model = self.label

    def evaluate(self, supplier_name, document_text, document_sha, criteria):
        resp = self.inner.evaluate(supplier_name, document_text, document_sha, criteria)
        if supplier_name.lower() not in self.names:
            return resp
        obj, _ = parse_llm_json(resp.raw_text)
        payload = obj or {"supplier_name": supplier_name, "criteria": []}
        resp.raw_text = _malformed_version(payload, criteria)
        resp.notes.append("Simulated malformed LLM output (validation demo).")
        resp.from_cache = False
        return resp
