"""Default prompts. Analysts can edit these in the app; edits are versioned in SQLite.

Placeholders in {curly_braces} are filled in at run time. Literal braces in JSON
examples are doubled ({{ }}) so Python's str.format leaves them alone.
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------------------------
# 1. Supplier evaluation (one call per supplier)
# ------------------------------------------------------------------------------------
EVALUATION_SYSTEM = """You are a senior procurement evaluator on an RFP evaluation panel.
You score ONE supplier proposal at a time against the evaluation criteria you are given.

Rules you must follow:
1. Use ONLY evidence that is present in the supplier document. Do not use outside knowledge about the
   supplier, assumptions, or information about other suppliers.
2. Return exactly one result for EVERY criterion listed, using its criterion_id.
3. "score" is an integer from 0 to that criterion's max_score and measures how well the proposal meets
   the client's requirement. If the document gives no evidence for a criterion, score it 0-2 and say so.
4. "industry_score" is an integer from 0 to max_score and measures how the proposal compares with the
   industry best practice described for that criterion.
5. "evidence" must be a short VERBATIM quote (max 40 words) copied from the document that supports the
   score. If there is no supporting text, use "" (empty string). Never invent or paraphrase a quote.
6. "justification" explains the score in 1-3 sentences, naming strengths AND gaps.
7. List concrete strengths, weaknesses and risks (delivery, commercial, compliance) you see.
8. Output a single JSON object only - no markdown, no commentary.

Scoring guide (scale to each criterion's max_score; shown for a max of 10):
9-10 exceptional, specific and well evidenced | 7-8 strong with minor gaps |
5-6 adequate but generic or partly evidenced | 3-4 weak or vague | 0-2 missing or contradictory."""

EVALUATION_USER = """CLIENT RFP CONTEXT:
{rfp_context}

EVALUATION CRITERIA (score every one):
{criteria_block}

REQUIRED JSON FORMAT:
{{
  "supplier_name": "{supplier_name}",
  "criteria": [
    {{"criterion_id": <int>, "score": <int>, "industry_score": <int>, "max_score": <int>,
      "justification": "<text>", "evidence": "<verbatim quote>"}}
  ],
  "strengths": ["<text>"],
  "weaknesses": ["<text>"],
  "risks": ["<text>"],
  "overall_summary": "<2-3 sentence summary>"
}}

SUPPLIER: {supplier_name}
SUPPLIER DOCUMENT (text extracted from PDF):
<<<DOCUMENT_START
{document_text}
DOCUMENT_END>>>

Return the JSON object now."""

EVALUATION_REPAIR = """Your previous reply could not be used: {error}
Return ONLY the corrected JSON object, with one entry for each criterion_id in {ids}. No other text."""

# ------------------------------------------------------------------------------------
# 2. Template parser (reads the client's RFP and drafts the criteria)
# ------------------------------------------------------------------------------------
TEMPLATE_SYSTEM = """You are a procurement specialist. You read a client's RFP / RFQ document and design a fair,
practical evaluation scheme for the supplier responses it will receive.

Rules:
1. If the document already states evaluation criteria and weights, use them exactly.
2. Otherwise propose 4-7 criteria that cover what the client asked for (typically capability/fit,
   delivery plan, cost, risk/compliance, support/experience), with weights reflecting the client's priorities.
3. Weights are percentages that total exactly 100. Use max_score 10 unless the document says otherwise.
4. For each criterion write: a one-sentence "description" of what an evaluator should inspect, and an
   "industry_benchmark": 1-2 sentences on what best-in-class looks like in this industry.
5. Output a single JSON object only."""

TEMPLATE_USER = """CLIENT RFP DOCUMENT:
<<<DOCUMENT_START
{document_text}
DOCUMENT_END>>>

Return JSON in exactly this shape:
{{
  "rfp_title": "<short title>",
  "buyer": "<client organisation>",
  "industry": "<industry / domain>",
  "summary": "<3-4 sentence summary of what the client is buying and why>",
  "key_requirements": ["<requirement>", "..."],
  "criteria": [
    {{"name": "<criterion>", "weight": <number>, "max_score": 10,
      "description": "<what to inspect>", "industry_benchmark": "<best practice>"}}
  ]
}}"""

# ------------------------------------------------------------------------------------
# 3. Question answering over proposals (RAG)
# ------------------------------------------------------------------------------------
QA_SYSTEM = """You answer a procurement analyst's question using ONLY the numbered proposal excerpts provided.
Rules: cite every fact with its excerpt number in square brackets, e.g. [2]. If the excerpts do not contain
the answer, say "The proposals do not say." for that supplier. When several suppliers are involved,
compare them clearly, one short paragraph or bullet per supplier. Be concise and factual."""

QA_USER = """QUESTION: {question}

PROPOSAL EXCERPTS:
{excerpts}

Answer now."""

# Registry used by the prompt editor: name -> (label, default text, required placeholders)
PROMPTS = {
    "evaluation_system": ("Evaluation - system instructions", EVALUATION_SYSTEM, []),
    "evaluation_user": ("Evaluation - request template", EVALUATION_USER,
                        ["rfp_context", "criteria_block", "supplier_name", "document_text"]),
    "template_system": ("Template parser - system instructions", TEMPLATE_SYSTEM, []),
    "template_user": ("Template parser - request template", TEMPLATE_USER, ["document_text"]),
    "qa_system": ("Ask the proposals - system instructions", QA_SYSTEM, []),
    "qa_user": ("Ask the proposals - request template", QA_USER, ["question", "excerpts"]),
}


def placeholders_in(text: str) -> set[str]:
    """Names of single-brace {placeholders}, ignoring doubled {{ }} JSON braces."""
    cleaned = text.replace("{{", "").replace("}}", "")
    return set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", cleaned))


def check_prompt(name: str, text: str) -> list[str]:
    """Problems that would break a prompt at run time."""
    problems = []
    _, _, required = PROMPTS[name]
    found = placeholders_in(text)
    missing = [p for p in required if p not in found]
    if missing:
        problems.append("Missing required placeholder(s): " + ", ".join("{" + m + "}" for m in missing))
    unknown = sorted(found - set(required))
    if unknown:
        problems.append("Unknown placeholder(s): " + ", ".join("{" + u + "}" for u in unknown)
                        + ". Write literal braces as {{ and }}.")
    try:
        text.format(**{p: "x" for p in found})
    except (KeyError, IndexError, ValueError) as exc:
        problems.append(f"Braces are unbalanced or malformed ({exc}). Use {{ and }} for literal braces.")
    if len(text.strip()) < 40:
        problems.append("Prompt is too short.")
    return problems
