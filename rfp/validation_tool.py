"""Validation Tool: makes the LLM output safe to calculate with.

The LLM is treated as an untrusted component. Before any arithmetic happens
this tool:

1. Extracts a JSON object from the raw reply (even if wrapped in prose or
   markdown fences).
2. Validates the structure with Pydantic.
3. Normalises every active criterion:
     missing criterion        -> score 0 + warning (conservative: no evidence, no points)
     non-numeric score        -> score 0 + warning
     out-of-range score       -> clipped to [0, max_score] + warning
     decimal score            -> rounded to nearest integer + warning
     unknown criterion_id     -> dropped + warning
     duplicate criterion_id   -> first kept + warning
     wrong max_score          -> replaced by the database value
     empty justification      -> "Not provided by evaluator" + warning
4. Checks that each evidence quote really appears in the supplier document
   (fuzzy match) and flags unverified evidence.

It also validates supplier metadata entered in the UI.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import config


# --------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------


class RawCriterionResult(BaseModel):
    """Loose schema: accepts what an LLM might send, so we can repair it."""

    model_config = ConfigDict(extra="ignore")
    criterion_id: Any = None
    criterion_name: Any = None
    name: Any = None
    score: Any = None
    max_score: Any = None
    justification: Any = ""
    evidence: Any = ""


class RawScorecard(BaseModel):
    model_config = ConfigDict(extra="ignore")
    supplier_name: Any = None
    criteria: list[RawCriterionResult] = Field(default_factory=list)
    risks: Any = Field(default_factory=list)
    overall_summary: Any = ""


class CriterionResult(BaseModel):
    """Strict schema: what the Ranking Tool is allowed to consume."""

    criterion_id: int
    name: str
    weight: float
    max_score: int
    score: int = Field(ge=0)
    justification: str
    evidence: str
    evidence_verified: bool
    status: str  # OK / NORMALISED / MISSING


class ValidatedScorecard(BaseModel):
    supplier_name: str
    criteria: list[CriterionResult]
    risks: list[str]
    overall_summary: str
    warnings: list[str]


# --------------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------------


def parse_llm_json(raw: str) -> tuple[dict | None, str]:
    """Return (object, error). Tolerates markdown fences and leading prose."""
    if not raw or not raw.strip():
        return None, "empty response"
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    candidates = [fence.group(1)] if fence else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    last_error = "no JSON object found"
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as exc:
            # common LLM slip: trailing commas
            try:
                obj = json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            except json.JSONDecodeError:
                last_error = f"invalid JSON ({exc.msg} at position {exc.pos})"
                continue
        if not isinstance(obj, dict):
            last_error = "top-level JSON value is not an object"
            continue
        if not isinstance(obj.get("criteria"), list):
            last_error = "JSON object has no 'criteria' list"
            continue
        return obj, ""
    return None, last_error


# --------------------------------------------------------------------------------
# Evidence verification
# --------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def evidence_in_document(evidence: str, document_text: str, threshold: float = 0.85) -> bool:
    """True if the quote (or most of it) appears in the document.

    Uses an exact normalised substring check first, then a fuzzy best-window
    match to tolerate small differences in whitespace or punctuation.
    """
    ev, doc = _norm(evidence), _norm(document_text)
    if len(ev) < 12:
        return False
    if ev in doc:
        return True
    ev_words = ev.split()
    doc_words = doc.split()
    n = len(ev_words)
    if n == 0 or len(doc_words) < n:
        return False
    # Anchor candidate windows on words from the quote to keep this fast.
    anchors = {w for w in ev_words if len(w) > 4} or set(ev_words)
    best = 0.0
    for i, w in enumerate(doc_words):
        if w not in anchors:
            continue
        for offset in range(0, min(n, 8)):
            s = max(0, i - offset)
            window = " ".join(doc_words[s : s + n])
            ratio = SequenceMatcher(None, ev, window).ratio()
            if ratio > best:
                best = ratio
                if best >= threshold:
                    return True
    return best >= threshold


# --------------------------------------------------------------------------------
# Scorecard normalisation
# --------------------------------------------------------------------------------


def _to_int_id(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_score(value: Any) -> tuple[float | None, str]:
    if isinstance(value, bool) or value is None:
        return None, "missing"
    if isinstance(value, (int, float)):
        return float(value), ""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(?:/\s*\d+)?\s*", str(value))
    if m:
        return float(m.group(1)), ""
    return None, f"non-numeric value {value!r}"


def validate_scorecard(
    raw_text: str,
    supplier_name: str,
    criteria: list[dict],
    document_text: str,
    llm_error: str = "",
) -> ValidatedScorecard:
    warnings: list[str] = []
    prefix = f"{supplier_name}:"

    obj, parse_error = parse_llm_json(raw_text)
    if llm_error and not raw_text:
        warnings.append(f"{prefix} {llm_error}. All criteria scored 0 - re-run this supplier.")
    if obj is None:
        if raw_text:
            warnings.append(f"{prefix} LLM output could not be parsed ({parse_error}). All criteria scored 0.")
        obj = {"criteria": []}
    elif raw_text and not raw_text.strip().startswith("{"):
        warnings.append(f"{prefix} LLM wrapped the JSON in extra text; JSON was extracted.")

    try:
        card = RawScorecard.model_validate(obj)
    except ValidationError as exc:
        warnings.append(f"{prefix} scorecard failed schema validation ({exc.error_count()} errors); "
                        "using whatever criteria could be read.")
        items = obj.get("criteria") if isinstance(obj.get("criteria"), list) else []
        card = RawScorecard(criteria=[RawCriterionResult.model_validate(i) for i in items if isinstance(i, dict)])

    reported_name = str(card.supplier_name or "").strip()
    if reported_name and reported_name.lower() != supplier_name.lower():
        warnings.append(f"{prefix} LLM reported supplier name '{reported_name}'; the entered name was kept.")

    by_id = {c["criterion_id"]: c for c in criteria}
    by_name = {c["name"].strip().lower(): c for c in criteria}
    received: dict[int, RawCriterionResult] = {}
    for item in card.criteria:
        cid = _to_int_id(item.criterion_id)
        if cid not in by_id:
            alt_name = str(item.criterion_name or item.name or "").strip().lower()
            if alt_name in by_name:
                cid = by_name[alt_name]["criterion_id"]
            else:
                warnings.append(f"{prefix} ignored result for unknown criterion_id {item.criterion_id!r}.")
                continue
        if cid in received:
            warnings.append(f"{prefix} duplicate result for criterion {cid}; first one kept.")
            continue
        received[cid] = item

    results: list[CriterionResult] = []
    for c in criteria:
        cid, max_score = c["criterion_id"], int(c["max_score"])
        item = received.get(cid)
        if item is None:
            warnings.append(f"{prefix} '{c['name']}' missing from LLM output; scored 0 (no evidence, no points).")
            results.append(CriterionResult(
                criterion_id=cid, name=c["name"], weight=float(c["weight"]), max_score=max_score, score=0,
                justification="Not returned by the evaluator; defaulted to 0.", evidence="",
                evidence_verified=False, status="MISSING"))
            continue

        status = "OK"
        value, problem = _coerce_score(item.score)
        if value is None:
            warnings.append(f"{prefix} '{c['name']}' had {problem or 'no'} score; set to 0.")
            value, status = 0.0, "NORMALISED"
        if value != round(value):
            warnings.append(f"{prefix} '{c['name']}' score {value:g} rounded to {round(value)}.")
            value, status = float(round(value)), "NORMALISED"
        if value < 0 or value > max_score:
            clipped = min(max(value, 0), max_score)
            warnings.append(f"{prefix} '{c['name']}' score {value:g} outside 0-{max_score}; clipped to {clipped:g}.")
            value, status = clipped, "NORMALISED"
        reported_max = _to_int_id(item.max_score)
        if reported_max is not None and reported_max != max_score:
            warnings.append(f"{prefix} '{c['name']}' reported max_score {reported_max}; database value {max_score} used.")

        justification = str(item.justification or "").strip()
        if not justification:
            justification = "Not provided by evaluator."
            warnings.append(f"{prefix} '{c['name']}' has no justification.")
        evidence = str(item.evidence or "").strip()
        verified = bool(evidence) and evidence_in_document(evidence, document_text)
        if not evidence:
            warnings.append(f"{prefix} '{c['name']}' has no supporting evidence quote.")
        elif not verified:
            warnings.append(f"{prefix} '{c['name']}' evidence quote could not be found in the document - review manually.")

        results.append(CriterionResult(
            criterion_id=cid, name=c["name"], weight=float(c["weight"]), max_score=max_score, score=int(value),
            justification=justification, evidence=evidence, evidence_verified=verified, status=status))

    risks = card.risks if isinstance(card.risks, list) else [card.risks] if card.risks else []
    return ValidatedScorecard(
        supplier_name=supplier_name,
        criteria=results,
        risks=[str(r) for r in risks if str(r).strip()],
        overall_summary=str(card.overall_summary or "").strip(),
        warnings=warnings,
    )


# --------------------------------------------------------------------------------
# Supplier metadata validation (UI input)
# --------------------------------------------------------------------------------


def validate_supplier_inputs(suppliers: list[dict], today: date | None = None) -> list[str]:
    """Return blocking errors for the supplier metadata table."""
    today = today or date.today()
    errors: list[str] = []
    seen: set[str] = set()
    for i, s in enumerate(suppliers, start=1):
        label = f"Row {i} ({s.get('filename', 'file')})"
        name = str(s.get("supplier_name") or "").strip()
        if not name:
            errors.append(f"{label}: supplier name is required.")
        elif name.lower() in seen:
            errors.append(f"{label}: supplier name '{name}' is used more than once.")
        seen.add(name.lower())

        d = s.get("submission_date")
        if isinstance(d, str):
            try:
                d = datetime.strptime(d, "%Y-%m-%d").date()
            except ValueError:
                d = None
        if not isinstance(d, date):
            errors.append(f"{label}: submission date must be a valid date (YYYY-MM-DD).")
        elif d > today:
            errors.append(f"{label}: submission date {d.isoformat()} is in the future.")

        r = s.get("experience_rating")
        try:
            r = float(r)
        except (TypeError, ValueError):
            r = None
        if r is None or not (config.EXPERIENCE_RATING_MIN <= r <= config.EXPERIENCE_RATING_MAX):
            errors.append(
                f"{label}: historical experience rating must be between "
                f"{config.EXPERIENCE_RATING_MIN} and {config.EXPERIENCE_RATING_MAX}."
            )
    return errors
