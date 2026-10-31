"""Evaluation Agent: asks an LLM to score ONE supplier against the template's criteria.

* The prompt is built from the analyst-editable prompt versions stored in SQLite,
  the template's context and its *current* criteria (nothing hard-coded).
* JSON mode, temperature 0, one automatic "repair" request if the JSON is unusable.
* Evaluation DNA: every scorecard is stored under a fingerprint of
  (document SHA-256 + criteria + prompt text + model). If the same fingerprint is
  seen again, the stored scorecard is reused - identical inputs always get the
  identical evaluation, and no AI call is spent.

The agent only *judges content*. It never computes totals, benchmarks or ranks.

The OfflineDemoEvaluator is a transparent keyword heuristic so the whole workflow
can be shown without an API key. It is labelled clearly and is not an AI judgement.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from . import config, db
from .llm_client import LLMClient, LLMError
from .validation_tool import parse_llm_json


@dataclass
class EvaluationResponse:
    raw_text: str
    model: str
    from_dna: bool = False
    fingerprint: str = ""
    dna_origin: str = ""          # run id that first produced this scorecard
    attempts: int = 1           # 0 = reused from DNA, no AI call
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def criteria_block(criteria: list[dict]) -> str:
    lines = []
    for c in criteria:
        lines.append(
            f"- criterion_id {c['criterion_id']}: {c['name']} (max_score {c['max_score']}, weight {c['weight']:g}%)\n"
            f"  What to inspect: {c['description']}\n"
            f"  Industry best practice: {c.get('industry_benchmark') or 'Use general good practice for this industry.'}"
        )
    return "\n".join(lines)


def rfp_context(template: dict) -> str:
    parts = [f"RFP: {template.get('name', '')}"]
    if template.get("buyer"):
        parts.append(f"Client: {template['buyer']}")
    if template.get("industry"):
        parts.append(f"Industry: {template['industry']}")
    if template.get("summary"):
        parts.append(f"Summary: {template['summary']}")
    if template.get("requirements"):
        parts.append("Key requirements:\n" + "\n".join(f"- {r}" for r in template["requirements"]))
    return "\n".join(parts)


def build_messages(supplier_name: str, document_text: str, criteria: list[dict], template: dict,
                   db_path=None) -> list[dict]:
    system = db.get_prompt("evaluation_system", db_path)["text"]
    user = db.get_prompt("evaluation_user", db_path)["text"].format(
        rfp_context=rfp_context(template), criteria_block=criteria_block(criteria),
        supplier_name=supplier_name, document_text=document_text)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fingerprint_parts(document_sha: str, criteria: list[dict], template: dict, model: str, db_path=None) -> dict:
    crit = json.dumps([(c["criterion_id"], c["name"], c["description"], c.get("industry_benchmark", ""),
                        c["max_score"]) for c in criteria], sort_keys=True)
    prompt_text = (db.get_prompt("evaluation_system", db_path)["text"] + "\n----\n"
                   + db.get_prompt("evaluation_user", db_path)["text"] + "\n----\n" + rfp_context(template))
    criteria_hash, prompt_hash = _sha(crit), _sha(prompt_text)
    fp = _sha(f"{model}|{document_sha}|{criteria_hash}|{prompt_hash}")
    return {"fingerprint": fp, "document_sha": document_sha, "criteria_hash": criteria_hash,
            "prompt_hash": prompt_hash, "model": model}


class LLMEvaluator:
    """Scores one proposal with a real LLM. DNA reuse is handled by the orchestrator."""

    def __init__(self, client: LLMClient, db_path=None):
        self.client = client
        self.db_path = db_path
        self.label = client.label
        self.model = client.label

    def evaluate(self, supplier_name: str, document_text: str, document_sha: str, criteria: list[dict],
                 template: dict, run_id: str = "") -> EvaluationResponse:
        from .prompts import EVALUATION_REPAIR

        messages = build_messages(supplier_name, document_text, criteria, template, self.db_path)
        ids = [c["criterion_id"] for c in criteria]
        raw, error = "", ""
        for attempt in range(1, config.LLM_MAX_ATTEMPTS + 1):
            try:
                raw = self.client.chat(messages, json_mode=True)
            except LLMError as exc:
                return EvaluationResponse(raw_text="", model=self.label, attempts=attempt, error=str(exc))
            _, error = parse_llm_json(raw)
            if not error:
                return EvaluationResponse(raw_text=raw, model=self.label, attempts=attempt)
            messages = messages + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": EVALUATION_REPAIR.format(error=error, ids=ids)}]
        return EvaluationResponse(raw_text=raw, model=self.label, attempts=config.LLM_MAX_ATTEMPTS, error=error)


# ------------------------------------------------------------------------------------
# Offline demo evaluator (no API key)
# ------------------------------------------------------------------------------------
_GENERIC_STOP = set("the and for with that this from into over their there which what when where while "
                    "about after before under other such than then them they your will shall must should "
                    "including include includes relevant evidence quality overall".split())


def _keywords(criterion: dict) -> list[str]:
    text = f"{criterion['name']} {criterion['description']} {criterion.get('industry_benchmark', '')}".lower()
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", text) if w not in _GENERIC_STOP]
    seen, out = set(), []
    for w in words:
        stem = w[:6]
        if stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out[:18]


def _best_sentence(text: str, keywords: list[str]) -> str:
    flat = re.sub(r"\[Page \d+\]", " ", text)
    flat = re.sub(r"(?<![.:!?])\n(?!\n)", " ", flat)
    best, best_hits = "", 0
    for s in re.split(r"(?<=[.!?])\s+|\n", flat):
        s = s.strip()
        if len(s) < 25:
            continue
        hits = sum(k in s.lower() for k in keywords)
        if hits > best_hits:
            best, best_hits = s, hits
    return " ".join(best.split()[:40])


class OfflineDemoEvaluator:
    label = "OFFLINE_DEMO (keyword heuristic - no AI)"
    model = label

    def evaluate(self, supplier_name: str, document_text: str, document_sha: str, criteria: list[dict],
                 template: dict, run_id: str = "") -> EvaluationResponse:
        lower = document_text.lower()
        n_words = max(1, len(lower.split()))
        results = []
        for c in criteria:
            kws = _keywords(c)
            distinct = sum(1 for k in kws if k in lower)
            mentions = sum(lower.count(k) for k in kws)
            coverage = distinct / max(1, len(kws))
            density = min(1.0, mentions / (n_words / 60))
            raw = 1.5 + 6.0 * coverage + 2.5 * density
            score = int(max(0, min(c["max_score"], round(raw * c["max_score"] / 10))))
            industry = int(max(0, min(c["max_score"], round((raw - 1.0) * c["max_score"] / 10))))
            results.append({
                "criterion_id": c["criterion_id"], "score": score, "industry_score": industry,
                "max_score": c["max_score"],
                "justification": f"Offline heuristic: {distinct} of {len(kws)} indicator terms found "
                                 f"({mentions} mentions) for {c['name']}.",
                "evidence": _best_sentence(document_text, kws),
            })
        payload = {"supplier_name": supplier_name, "criteria": results,
                   "strengths": [], "weaknesses": [],
                   "risks": ["Offline demo mode: strengths, weaknesses and risks need an AI provider."],
                   "overall_summary": "Scored by the offline keyword heuristic for demonstration only."}
        return EvaluationResponse(raw_text=json.dumps(payload), model=self.label)


class FaultInjectingEvaluator:
    """Wraps any evaluator and corrupts the reply for chosen suppliers (validation demo)."""

    def __init__(self, inner, supplier_names: set[str]):
        self.inner = inner
        self.names = {n.lower() for n in supplier_names}
        self.label = f"{inner.label} + simulated faults"
        self.model = self.label
        self.dna_label = inner.label      # healthy suppliers still share DNA with normal runs

    def skip_dna(self, supplier_name: str) -> bool:
        return supplier_name.lower() in self.names

    def evaluate(self, supplier_name, document_text, document_sha, criteria, template, run_id=""):
        resp = self.inner.evaluate(supplier_name, document_text, document_sha, criteria, template, run_id)
        if supplier_name.lower() not in self.names:
            return resp
        obj, _ = parse_llm_json(resp.raw_text)
        crit = (obj or {}).get("criteria", [])
        broken = []
        if crit:
            broken.append({**crit[0], "score": 14})                                   # out of range -> clipped
        if len(crit) > 1:
            broken.append({**crit[1], "score": "eight"})                              # non-numeric -> 0
        if len(crit) > 2:
            broken.append({**crit[2], "score": 6, "evidence": "We guarantee zero defects forever."})  # invented quote
        broken.append({"criterion_id": 999, "score": 9, "justification": "Unknown criterion", "evidence": ""})
        body = json.dumps({"supplier_name": supplier_name, "criteria": broken})       # rest omitted -> 0
        resp.raw_text = "Sure! Here is the evaluation:\n```json\n" + body + "\n```"
        resp.from_dna = False
        resp.notes.append("Simulated malformed AI output (validation demo).")
        return resp
