"""Orchestrator Agent: a LangGraph state machine that calls each tool in order.

    START
      -> load_template        (SQLite: template + its ACTIVE criteria + active prompt versions)
      -> create_batch         (RFP_RUN_ID, run row + PENDING supplier rows)
      -> extract_documents    (Document Tool)
      -> [conditional] no readable documents?  -> fail_run -> END
      -> evaluate_suppliers   (Evaluation DNA lookup, else Evaluation Agent - one AI call per supplier)
      -> validate_scorecards  (Validation Tool)
      -> rank_suppliers       (Ranking Tool - deterministic Python only)
      -> persist_results      (SQLite, one transaction, one RFP_RUN_ID; proposal text saved for Q&A)
      -> END

Every step writes to an execution trace that is shown in the UI and exported.
`rescore_run` re-applies analyst overrides and re-ranks a saved run with no AI calls.
"""

from __future__ import annotations

import operator
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config, db
from .document_tool import extract_pdf_text
from .evaluation_agent import fingerprint_parts
from .ranking_tool import apply_overrides, compute_rankings
from .validation_tool import parse_llm_json, validate_scorecard

FORMULAS = {
    "Absolute weighted score": "sum((criterion score / max score) x weight) -> 0-100",
    "Criterion benchmark": "highest validated score for the criterion across evaluated suppliers",
    "Criterion gap": "supplier score - benchmark (0 for the leader, otherwise negative)",
    "Relative performance %": "supplier score / benchmark x 100 (0 for everyone if the benchmark is 0)",
    "Peer Performance Index (PPI)": "sum(relative % x weight) / sum(weights) -> used for ranking",
    "Industry Alignment Index": "sum((industry score / max score) x weight) -> 0-100, shown alongside",
    "Tie-break order": "1) higher PPI 2) earlier submission date 3) higher experience rating 4) supplier name A-Z",
}


class RunState(TypedDict, total=False):
    template_id: int
    use_dna: bool
    rfp_run_id: str
    created_at: str
    db_path: Any
    template: dict
    criteria: list[dict]
    prompt_snapshot: dict
    supplier_inputs: list[dict]
    documents: dict
    raw_evaluations: dict
    scorecards: list[dict]
    ranking: dict
    not_evaluated: list[dict]
    status: str
    result: dict
    warnings: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]


def _t(step: str, actor: str, detail: str) -> dict:
    return {"time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "step": step, "actor": actor, "detail": detail}


def new_run_id() -> str:
    return f"RFP-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"


def build_graph(evaluator, progress: Callable[[str, float], None] | None = None):
    report = progress or (lambda msg, frac: None)

    def load_template(state: RunState):
        dbp = state.get("db_path")
        template = db.get_template(state["template_id"], dbp)
        if not template:
            raise ValueError("The selected template no longer exists.")
        criteria = db.get_criteria(state["template_id"], dbp, active_only=True)
        problems = db.validate_criteria_set(criteria)
        if problems:
            raise ValueError("Criteria configuration is invalid: " + " ".join(problems))
        prompt_snapshot = {n: db.get_prompt(n, dbp)["version"] for n in ("evaluation_system", "evaluation_user")}
        report("Loaded template, criteria and prompt from SQLite", 0.05)
        return {"template": template, "criteria": criteria, "prompt_snapshot": prompt_snapshot,
                "trace": [_t("1 Setup", "Orchestrator -> SQLite",
                             f"Template '{template['name']}' with {len(criteria)} active criteria: "
                             + ", ".join(f"{c['name']} ({c['weight']:g}%)" for c in criteria)
                             + f"; evaluation prompt versions {prompt_snapshot}")]}

    def create_batch(state: RunState):
        run_id = new_run_id()
        db.create_run(run_id, state["template"], evaluator.label, state["criteria"], state["prompt_snapshot"],
                      state["supplier_inputs"], state.get("db_path"))
        report(f"Created batch {run_id}", 0.1)
        return {"rfp_run_id": run_id, "created_at": db.utc_now(), "status": "RUNNING",
                "trace": [_t("3 Batch", "Orchestrator -> SQLite",
                             f"Run {run_id} created with {len(state['supplier_inputs'])} PENDING supplier entries")]}

    def extract_documents(state: RunState):
        docs, not_eval, warnings, trace = {}, [], [], []
        for s in state["supplier_inputs"]:
            res = extract_pdf_text(s["file_bytes"], s["filename"])
            if res.ok:
                docs[s["supplier_name"]] = {"text": res.text, "sha256": res.sha256, "pages": res.pages,
                                            "chars": res.chars, "truncated": res.truncated,
                                            "extractor": res.extractor, "filename": s["filename"]}
                warnings += res.warnings
                trace.append(_t("4 Evaluate", "Document Tool",
                                f"{s['supplier_name']}: {res.pages} page(s), {res.chars:,} characters via {res.extractor}"))
            else:
                not_eval.append({"supplier_name": s["supplier_name"], "filename": s["filename"], "reason": res.error})
                warnings.append(f"{s['supplier_name']} NOT evaluated - {res.error}")
                trace.append(_t("4 Evaluate", "Document Tool", f"{s['supplier_name']}: extraction FAILED"))
        report(f"Read {len(docs)} document(s)", 0.2)
        return {"documents": docs, "not_evaluated": not_eval, "warnings": warnings, "trace": trace}

    def route_after_extraction(state: RunState):
        return "evaluate" if state["documents"] else "fail"

    def fail_run(state: RunState):
        msg = "No supplier documents could be read, so nothing was evaluated."
        db.fail_run(state["rfp_run_id"], state.get("warnings", []) + [msg], state.get("db_path"))
        return {"status": "FAILED", "warnings": [msg], "trace": [_t("Stop", "Orchestrator", msg)]}

    def evaluate_suppliers(state: RunState):
        dbp, criteria, template = state.get("db_path"), state["criteria"], state["template"]
        use_dna = state.get("use_dna", True)
        names = [s["supplier_name"] for s in state["supplier_inputs"] if s["supplier_name"] in state["documents"]]

        # Identical documents inside this batch are evaluated once and flagged.
        first_by_sha, duplicates = {}, {}
        for n in names:
            sha = state["documents"][n]["sha256"]
            if sha in first_by_sha:
                duplicates[n] = first_by_sha[sha]
            else:
                first_by_sha[sha] = n
        unique = [n for n in names if n not in duplicates]

        # Evaluation DNA: reuse stored scorecards for identical document + criteria + prompt + model.
        results, fps, todo = {}, {}, []
        dna_label = getattr(evaluator, "dna_label", evaluator.label)
        skip = getattr(evaluator, "skip_dna", lambda name: False)
        for n in unique:
            d = state["documents"][n]
            parts = fingerprint_parts(d["sha256"], criteria, template, dna_label, dbp)
            fps[n] = parts
            hit = db.dna_get(parts["fingerprint"], dbp) if use_dna and not skip(n) else None
            if hit:
                db.dna_touch(parts["fingerprint"], dbp)
                from .evaluation_agent import EvaluationResponse
                results[n] = EvaluationResponse(raw_text=hit["response"], model=dna_label, from_dna=True,
                                                fingerprint=parts["fingerprint"], dna_origin=hit["first_run_id"] or "",
                                                attempts=0)
            else:
                todo.append(n)

        def run_one(name):
            d = state["documents"][name]
            return evaluator.evaluate(name, d["text"], d["sha256"], criteria, template, state["rfp_run_id"])

        if todo:
            with ThreadPoolExecutor(max_workers=min(config.MAX_PARALLEL_LLM_CALLS, len(todo))) as pool:
                futures = {pool.submit(run_one, n): n for n in todo}
                for fut in as_completed(futures):
                    name = futures[fut]
                    r = fut.result()
                    r.fingerprint = fps[name]["fingerprint"]
                    results[name] = r
                    if use_dna and not skip(name) and not r.error and parse_llm_json(r.raw_text)[0] is not None:
                        db.dna_put({**fps[name], "response": r.raw_text, "first_run_id": state["rfp_run_id"],
                                    "first_supplier": name}, dbp)
                    report(f"Evaluation Agent scored {name}", 0.2 + 0.6 * len(results) / len(unique))

        warnings = []
        for dup, original in duplicates.items():
            results[dup] = results[original]
            warnings.append(f"{dup}: document is identical to {original}'s (same SHA-256); the same scorecard was "
                            "reused. Check for a duplicate submission.")

        raw, trace = {}, []
        for name in names:
            r = results[name]
            raw[name] = {"raw_text": r.raw_text, "model": r.model, "from_dna": r.from_dna, "fingerprint": r.fingerprint,
                         "dna_origin": r.dna_origin, "attempts": r.attempts, "error": r.error, "notes": r.notes,
                         "reused_from": duplicates.get(name)}
            if name in duplicates:
                how = f"identical document to {duplicates[name]}; scorecard shared"
            elif r.from_dna:
                how = f"DNA match {r.fingerprint[:10]} - reused evaluation from run {r.dna_origin or 'earlier'} (no AI call)"
            else:
                how = f"{r.attempts} AI call(s)"
            if r.error:
                how += f"; problem: {r.error}"
            trace.append(_t("4 Evaluate", f"Evaluation Agent ({r.model})", f"{name}: {how}"))
        return {"raw_evaluations": raw, "trace": trace, "warnings": warnings}

    def validate_scorecards(state: RunState):
        meta = {s["supplier_name"]: s for s in state["supplier_inputs"]}
        cards, warnings, trace = [], [], []
        for name, r in state["raw_evaluations"].items():
            v = validate_scorecard(r["raw_text"], name, state["criteria"], state["documents"][name]["text"], r["error"])
            if r.get("reused_from") or r.get("from_dna"):
                v.warnings = [w for w in v.warnings if "LLM reported supplier name" not in w]
            m = meta[name]
            cards.append({
                "supplier_name": name,
                "submission_date": str(m["submission_date"]),
                "experience_rating": float(m["experience_rating"]),
                "source_file": m["filename"],
                "document_sha256": state["documents"][name]["sha256"],
                "dna_fingerprint": r["fingerprint"],
                "dna_reused": bool(r["from_dna"]),
                "dna_origin_run": r["dna_origin"],
                "criteria": [c.model_dump() for c in v.criteria],
                "strengths": v.strengths,
                "weaknesses": v.weaknesses,
                "risks": v.risks,
                "overall_summary": v.overall_summary,
                "validation_warnings": v.warnings,
                "evaluation_model": r["model"],
            })
            warnings += v.warnings
            verified = sum(c.evidence_verified for c in v.criteria)
            trace.append(_t("5 Validate", "Validation Tool",
                            f"{name}: {len(v.warnings)} warning(s); evidence verified for {verified}/{len(v.criteria)} criteria"))
        report("Validated every scorecard", 0.85)
        return {"scorecards": cards, "warnings": warnings, "trace": trace}

    def rank_suppliers(state: RunState):
        ranking = compute_rankings(state["scorecards"], state["criteria"])
        order = " > ".join(f"#{s['final_rank']} {s['supplier_name']} (PPI {s['ppi']:.2f})" for s in ranking["suppliers"])
        trace = [
            _t("6 Score", "Ranking Tool", "Absolute weighted scores and Industry Alignment Index calculated"),
            _t("7 Benchmark", "Ranking Tool", "Benchmarks: " + "; ".join(
                f"{b['name']}={b['benchmark_score']:g} ({', '.join(b['leaders'])})" for b in ranking["benchmarks"])),
            _t("8 Rank", "Ranking Tool", order),
        ]
        tb = [t for t in ranking["tie_breaks"] if t["is_tie_break"]]
        if tb:
            trace.append(_t("8 Rank", "Ranking Tool", f"{len(tb)} tie-break(s) applied"))
        report("Ranked suppliers with deterministic tie-breaks", 0.92)
        return {"ranking": ranking, "warnings": ranking["warnings"], "trace": trace}

    def persist_results(state: RunState):
        t = state["template"]
        result = {
            "rfp_run_id": state["rfp_run_id"],
            "created_at": state["created_at"],
            "status": "COMPLETED",
            "revision": 1,
            "template": {k: t.get(k) for k in ("template_id", "name", "buyer", "industry", "summary", "requirements")},
            "rfp_title": t["name"],
            "evaluation_mode": evaluator.label,
            "prompt_versions": state["prompt_snapshot"],
            "criteria": state["criteria"],
            "formulas": FORMULAS,
            "scorecards": state["scorecards"],          # validated AI output, kept for re-scoring
            "suppliers": state["ranking"]["suppliers"],
            "benchmarks": state["ranking"]["benchmarks"],
            "tie_breaks": state["ranking"]["tie_breaks"],
            "not_evaluated": state.get("not_evaluated", []),
            "overrides": [],
            "warnings": state.get("warnings", []),
        }
        trace = state.get("trace", []) + [_t("9 Persist", "Orchestrator -> SQLite",
                                             f"Saved {len(result['suppliers'])} supplier result(s) and proposal text under {state['rfp_run_id']}")]
        result["execution_trace"] = trace
        db.save_run_documents(state["rfp_run_id"], state["documents"], state.get("db_path"))
        db.save_run_results(result, state.get("db_path"))
        report("Saved results to SQLite", 1.0)
        return {"status": "COMPLETED", "result": result, "trace": trace[-1:]}

    g = StateGraph(RunState)
    for name, fn in [("load_template", load_template), ("create_batch", create_batch),
                     ("extract_documents", extract_documents), ("fail_run", fail_run),
                     ("evaluate_suppliers", evaluate_suppliers), ("validate_scorecards", validate_scorecards),
                     ("rank_suppliers", rank_suppliers), ("persist_results", persist_results)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_template")
    g.add_edge("load_template", "create_batch")
    g.add_edge("create_batch", "extract_documents")
    g.add_conditional_edges("extract_documents", route_after_extraction,
                            {"evaluate": "evaluate_suppliers", "fail": "fail_run"})
    g.add_edge("fail_run", END)
    g.add_edge("evaluate_suppliers", "validate_scorecards")
    g.add_edge("validate_scorecards", "rank_suppliers")
    g.add_edge("rank_suppliers", "persist_results")
    g.add_edge("persist_results", END)
    return g.compile()


def run_evaluation(template_id: int, supplier_inputs: list[dict], evaluator, use_dna: bool = True, db_path=None,
                   progress: Callable[[str, float], None] | None = None) -> dict:
    graph = build_graph(evaluator, progress)
    initial: RunState = {"template_id": template_id, "supplier_inputs": supplier_inputs, "db_path": db_path,
                         "use_dna": use_dna, "warnings": [],
                         "trace": [_t("2 Input", "Streamlit UI", f"{len(supplier_inputs)} supplier PDF(s) received")]}
    return graph.invoke(initial)


def rescore_run(rfp_run_id: str, db_path=None) -> dict:
    """Re-apply all analyst overrides to the saved AI scorecards and re-rank. No AI calls."""
    run = db.load_run(rfp_run_id, db_path)
    if not run:
        raise ValueError(f"Run {rfp_run_id} not found.")
    overrides = db.list_overrides(rfp_run_id, db_path)
    cards = apply_overrides(run["scorecards"], overrides)
    ranking = compute_rankings(cards, run["criteria"])
    base_warnings = [w for w in run["warnings"] if not w.startswith("Analyst override:")]
    override_notes = [f"Analyst override: {o['supplier_name']} / {o['criterion_name']} {o['old_score']:g} -> "
                      f"{o['new_score']:g} ({o['reason']})" for o in overrides]
    run.update({
        "suppliers": ranking["suppliers"], "benchmarks": ranking["benchmarks"], "tie_breaks": ranking["tie_breaks"],
        "overrides": overrides, "revision": int(run.get("revision", 1)) + 1,
        "warnings": base_warnings + override_notes,
    })
    run["execution_trace"] = run.get("execution_trace", []) + [
        _t("Re-score", "Ranking Tool", f"Revision {run['revision']}: {len(overrides)} override(s) applied and re-ranked (no AI calls)")]
    db.save_run_results(run, db_path)
    return run
