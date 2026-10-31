"""Template Agent: reads the client's RFP / RFQ and drafts the evaluation scheme.

This is what makes the app work for any domain (software, vehicles, medicines...):
the criteria, weights and industry benchmarks come from the client's own document
instead of being hard-coded.

* With an LLM: the model reads the RFP and returns criteria as JSON.
* Without an LLM (offline mode): a rules-based parser reads an "Evaluation criteria"
  section written as lines like  "C1. Technical Capability (30%): description".
Either way the result is normalised (weights total 100) and shown to the analyst
for review before it is saved.
"""

from __future__ import annotations

import re

from . import config, db
from .llm_client import LLMClient, LLMError
from .validation_tool import parse_json_object


def normalise_weights(weights: list[float]) -> list[int]:
    """Scale to integers that total exactly 100 (largest-remainder method)."""
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0:
        weights, total = [1.0] * len(weights), float(len(weights))
    raw = [max(0.0, float(w)) * 100 / total for w in weights]
    floors = [int(x) for x in raw]
    shortfall = 100 - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - floors[i], -i), reverse=True)
    for i in order[:shortfall]:
        floors[i] += 1
    return floors


def clean_scheme(obj: dict, filename: str = "") -> tuple[dict, list[dict], list[str]]:
    """Validate/normalise a parsed scheme. Returns (template_info, criteria, notes)."""
    notes: list[str] = []
    items = obj.get("criteria") if isinstance(obj.get("criteria"), list) else []
    criteria, seen = [], set()
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()[:80]
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            weight = float(str(it.get("weight", 0)).replace("%", "").strip() or 0)
        except ValueError:
            weight = 0.0
        try:
            max_score = int(float(it.get("max_score") or config.DEFAULT_MAX_SCORE))
        except (TypeError, ValueError):
            max_score = config.DEFAULT_MAX_SCORE
        criteria.append({
            "name": name, "weight": weight, "max_score": max_score if 1 <= max_score <= 100 else config.DEFAULT_MAX_SCORE,
            "description": str(it.get("description") or f"Evidence relevant to {name}.").strip(),
            "industry_benchmark": str(it.get("industry_benchmark") or "").strip(), "is_active": 1,
        })
    if len(criteria) > config.MAX_CRITERIA:
        notes.append(f"Only the first {config.MAX_CRITERIA} criteria were kept.")
        criteria = criteria[: config.MAX_CRITERIA]
    if criteria:
        original = [c["weight"] for c in criteria]
        fixed = normalise_weights(original)
        if [round(w, 6) for w in original] != [float(f) for f in fixed]:
            notes.append(f"Weights were rescaled to total 100% (they totalled {sum(original):g}%).")
        for c, w in zip(criteria, fixed):
            c["weight"] = w
    title = str(obj.get("rfp_title") or "").strip() or (filename.rsplit(".", 1)[0] if filename else "New RFP template")
    info = {
        "name": title[:120],
        "buyer": str(obj.get("buyer") or "").strip(),
        "industry": str(obj.get("industry") or "").strip(),
        "summary": str(obj.get("summary") or "").strip(),
        "requirements": [str(r).strip() for r in (obj.get("key_requirements") or []) if str(r).strip()][:15],
    }
    return info, criteria, notes


# ------------------------------------------------------------------------------------
# Rules-based parser (no LLM)
# ------------------------------------------------------------------------------------
_CRIT = re.compile(r"^\s*C\d+[.)]\s*(?P<name>[^()\n]{3,80}?)\s*\(\s*(?P<w>\d+(?:\.\d+)?)\s*%\s*\)\s*[:\-–]\s*(?P<desc>.+)$")
_BENCH = re.compile(r"^\s*(?:industry\s+)?best\s+practice\s*[:\-–]\s*(?P<b>.+)$", re.I)


_NEW_ITEM = re.compile(r"^\s*(?:[CR]\d+[.)]|\d+[.)]\s|[A-Z][A-Za-z /&-]{2,30}:|(?:industry\s+)?best\s+practice\s*[:\-])", re.I)


def logical_lines(text: str) -> list[str]:
    """Re-join lines that PDF extraction wrapped, so each item becomes one line."""
    out: list[str] = []
    for raw in re.sub(r"\[Page \d+\]", "\n", text).splitlines():
        ln = raw.strip()
        if not ln:
            out.append("")
        elif out and out[-1] and not _NEW_ITEM.match(ln) and not out[-1].endswith(":"):
            out[-1] += " " + ln
        else:
            out.append(ln)
    return [l for l in out if l]


def parse_rules(text: str, filename: str = "") -> dict:
    lines = logical_lines(text)
    criteria = []
    for ln in lines:
        m = _CRIT.match(ln)
        if m:
            criteria.append({"name": m.group("name").strip(), "weight": float(m.group("w")),
                             "description": m.group("desc").strip(), "industry_benchmark": ""})
            continue
        b = _BENCH.match(ln)
        if b and criteria and not criteria[-1]["industry_benchmark"]:
            criteria[-1]["industry_benchmark"] = b.group("b").strip()

    def grab(*labels):
        for label in labels:
            for ln in lines:
                m = re.match(rf"^\s*{label}\s*[:\-]\s*(.+)$", ln, re.I)
                if m:
                    return m.group(1).strip()
        return ""

    reqs = [re.sub(r"^\s*R\d+[.)]\s*", "", ln) for ln in lines if re.match(r"^\s*R\d+[.)]", ln)]
    return {"rfp_title": grab("RFP title", "Title"), "buyer": grab("Issued by", "Buyer", "Client"),
            "industry": grab("Industry"), "summary": grab("Summary"), "key_requirements": reqs,
            "criteria": criteria}


# ------------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------------


def draft_scheme(document_text: str, filename: str, client: LLMClient | None, db_path=None) -> dict:
    """Return {"template": {...}, "criteria": [...], "notes": [...], "mode": "LLM"|"RULES"}."""
    notes: list[str] = []
    obj, mode = None, "RULES"
    if client is not None:
        system = db.get_prompt("template_system", db_path)["text"]
        user = db.get_prompt("template_user", db_path)["text"].format(document_text=document_text[: config.MAX_DOCUMENT_CHARS])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            raw = client.chat(messages, json_mode=True)
            obj, err = parse_json_object(raw, require_key="criteria")
            if obj is None:
                raw = client.chat(messages + [{"role": "assistant", "content": raw},
                                              {"role": "user", "content": f"That was not valid JSON ({err}). Return only the JSON object."}],
                                  json_mode=True)
                obj, err = parse_json_object(raw, require_key="criteria")
            if obj is None:
                notes.append(f"The AI's criteria could not be read ({err}); used the rules-based parser instead.")
            else:
                mode = "LLM"
        except LLMError as exc:
            notes.append(f"{exc} Used the rules-based parser instead.")
    if obj is None:
        obj = parse_rules(document_text, filename)
        if not obj["criteria"]:
            notes.append("No criteria were found in the document, so the standard five procurement criteria were "
                         "suggested. Review and edit them before saving.")
            obj["criteria"] = [{"name": n, "weight": w, "description": d, "industry_benchmark": b}
                               for n, w, d, b in db.SAMPLE_TEMPLATE["criteria"]]
    info, criteria, more = clean_scheme(obj, filename)
    notes += more
    if len(criteria) < config.MIN_CRITERIA:
        notes.append("Fewer than two usable criteria were produced; add criteria before saving.")
    return {"template": info, "criteria": criteria, "notes": notes, "mode": mode}
