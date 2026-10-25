"""Orchestrator Agent: a LangGraph state machine that calls each tool in order.

    START
      -> load_criteria        (SQLite: latest ACTIVE criteria, reloaded at run time)
      -> create_batch         (RFP_RUN_ID, run row in SQLite)
      -> extract_documents    (Document Tool)
      -> [conditional] no readable documents?  -> fail_run -> END
      -> evaluate_suppliers   (Evaluation Agent, one LLM call per supplier)
      -> validate_scorecards  (Validation Tool)
      -> rank_suppliers       (Ranking Tool - deterministic Python only)
      -> persist_results      (SQLite, single transaction, one RFP_RUN_ID)
      -> END

Every step appends to an execution trace that is shown in the UI and saved in
the exported JSON, so a reviewer can see exactly which tool did what.
"""

from __future__ import annotations

import operator
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import db
from .document_tool import extract_pdf_text
from .ranking_tool import compute_rankings
from .validation_tool import validate_scorecard

FORMULAS = {
    "absolute_weighted_score": "sum((criterion score / max score) x criterion weight); 0-100",
    "criterion_benchmark": "highest validated score for the criterion across all evaluated suppliers",
    "criterion_gap": "supplier score - benchmark score (0 for the leader, otherwise negative)",
    "relative_performance_pct": "supplier score / benchmark x 100; 0 for everyone if the benchmark is 0",
    "peer_performance_index": "sum(relative % x weight) / sum(weights)",
    "tie_break_order": "1) higher PPI 2) earlier submission date 3) higher experience rating 4) supplier name A-Z",
}


class RunState(TypedDict, total=False):
    rfp_title: str
    rfp_run_id: str
    created_at: str
    db_path: Any
    criteria: list[dict]
    supplier_inputs: list[dict]          # name, date, rating, filename, file_bytes
    documents: dict                      # supplier -> extraction info
    raw_evaluations: dict                # supplier -> EvaluationResponse fields
    scorecards: list[dict]               # validated scorecards + metadata
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
    """Compile the workflow. `evaluator` is an LLMEvaluator or OfflineDemoEvaluator."""
    report = progress or (lambda msg, frac: None)

    def load_criteria(state: RunState):
        report("Loading active criteria from SQLite", 0.05)
        criteria = db.get_criteria(state.get("db_path"), active_only=True)
        problems = db.validate_criteria_set(criteria)
        trace = [_t("1 Setup", "Orchestrator -> SQLite", f"Loaded {len(criteria)} active criteria: "
                    + ", ".join(f"{c['name']} ({c['weight']:g}%)" for c in criteria))]
        if problems:
            raise ValueError("Criteria configuration is invalid: " + " ".join(problems))
        return {"criteria": criteria, "trace": trace}

    def create_batch(state: RunState):
        run_id = new_run_id()
        db.create_run(run_id, state.get("rfp_title", ""), evaluator.label, state["criteria"], state.get("db_path"))
        report(f"Created batch {run_id}", 0.1)
        return {"rfp_run_id": run_id, "created_at": db.utc_now(), "status": "RUNNING",
                "trace": [_t("3 Batch", "Orchestrator -> SQLite",
                             f"Created run {run_id} for {len(state['supplier_inputs'])} supplier document(s)")]}

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
                                f"{s['supplier_name']}: {res.pages} pages, {res.chars:,} chars via {res.extractor}"))
            else:
                not_eval.append({"supplier_name": s["supplier_name"], "filename": s["filename"], "reason": res.error})
                warnings.append(f"{s['supplier_name']} NOT evaluated - {res.error}")
                trace.append(_t("4 Evaluate", "Document Tool", f"{s['supplier_name']}: extraction FAILED"))
        report(f"Extracted text from {len(docs)} document(s)", 0.2)
        return {"documents": docs, "not_evaluated": not_eval, "warnings": warnings, "trace": trace}

    def route_after_extraction(state: RunState):
        return "evaluate" if state["documents"] else "fail"

    def fail_run(state: RunState):
        msg = "No supplier documents could be read, so nothing was evaluated."
        db.fail_run(state["rfp_run_id"], state.get("warnings", []) + [msg], state.get("db_path"))
        return {"status": "FAILED", "warnings": [msg],
                "trace": [_t("Stop", "Orchestrator", msg)]}

    def evaluate_suppliers(state: RunState):
        criteria = state["criteria"]
        names = [s["supplier_name"] for s in state["supplier_inputs"] if s["supplier_name"] in state["documents"]]

        def run_one(name):
            d = state["documents"][name]
            return evaluator.evaluate(name, d["text"], d["sha256"], criteria)

        # LLM calls run in parallel threads; progress is reported from this (main) thread only,
        # and results are re-ordered by input order so the outcome is deterministic.
        results = {}
        with ThreadPoolExecutor(max_workers=min(4, len(names))) as pool:
            futures = {pool.submit(run_one, n): n for n in names}
            for fut in as_completed(futures):
                name = futures[fut]
                results[name] = fut.result()
                report(f"Evaluation Agent scored {name}", 0.2 + 0.6 * len(results) / len(names))

        raw, trace = {}, []
        for name in names:
            r = results[name]
            raw[name] = {"raw_text": r.raw_text, "model": r.model, "from_cache": r.from_cache,
                         "attempts": r.attempts, "error": r.error, "notes": r.notes}
            how = "cache hit (identical document + criteria)" if r.from_cache else f"{r.attempts} LLM call(s)"
            if r.error:
                how += f"; problem: {r.error}"
            trace.append(_t("4 Evaluate", f"Evaluation Agent ({r.model})", f"{name}: {how}"))
        return {"raw_evaluations": raw, "trace": trace}

    def validate_scorecards(state: RunState):
        meta = {s["supplier_name"]: s for s in state["supplier_inputs"]}
        cards, warnings, trace = [], [], []
        for name, r in state["raw_evaluations"].items():
            v = validate_scorecard(r["raw_text"], name, state["criteria"], state["documents"][name]["text"], r["error"])
            m = meta[name]
            cards.append({
                "supplier_name": name,
                "submission_date": str(m["submission_date"]),
                "experience_rating": float(m["experience_rating"]),
                "source_file": m["filename"],
                "document_sha256": state["documents"][name]["sha256"],
                "criteria": [c.model_dump() for c in v.criteria],
                "risks": v.risks,
                "overall_summary": v.overall_summary,
                "validation_warnings": v.warnings,
                "evaluation_model": r["model"],
                "from_cache": r["from_cache"],
            })
            warnings += v.warnings
            verified = sum(c.evidence_verified for c in v.criteria)
            trace.append(_t("5 Validate", "Validation Tool",
                            f"{name}: {len(v.warnings)} warning(s); evidence verified for {verified}/{len(v.criteria)} criteria"))
        report("Validated and normalised all scorecards", 0.85)
        return {"scorecards": cards, "warnings": warnings, "trace": trace}

    def rank_suppliers(state: RunState):
        ranking = compute_rankings(state["scorecards"], state["criteria"])
        order = " > ".join(f"#{s['final_rank']} {s['supplier_name']} (PPI {s['ppi']:.2f})" for s in ranking["suppliers"])
        trace = [
            _t("6 Score", "Ranking Tool", "Absolute weighted scores calculated for every supplier"),
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
        result = {
            "rfp_run_id": state["rfp_run_id"],
            "rfp_title": state.get("rfp_title", ""),
            "created_at": state["created_at"],
            "status": "COMPLETED",
            "evaluation_mode": evaluator.label,
            "criteria": state["criteria"],
            "formulas": FORMULAS,
            "suppliers": state["ranking"]["suppliers"],
            "benchmarks": state["ranking"]["benchmarks"],
            "tie_breaks": state["ranking"]["tie_breaks"],
            "not_evaluated": state.get("not_evaluated", []),
            "warnings": state.get("warnings", []),
        }
        trace = state.get("trace", []) + [_t("9 Persist", "Orchestrator -> SQLite",
                                             f"Saved {len(result['suppliers'])} supplier result(s) under {state['rfp_run_id']}")]
        result["execution_trace"] = trace
        db.save_run_results(result, state.get("db_path"))
        report("Saved results to SQLite", 1.0)
        return {"status": "COMPLETED", "result": result, "trace": trace[-1:]}

    g = StateGraph(RunState)
    g.add_node("load_criteria", load_criteria)
    g.add_node("create_batch", create_batch)
    g.add_node("extract_documents", extract_documents)
    g.add_node("fail_run", fail_run)
    g.add_node("evaluate_suppliers", evaluate_suppliers)
    g.add_node("validate_scorecards", validate_scorecards)
    g.add_node("rank_suppliers", rank_suppliers)
    g.add_node("persist_results", persist_results)

    g.add_edge(START, "load_criteria")
    g.add_edge("load_criteria", "create_batch")
    g.add_edge("create_batch", "extract_documents")
    g.add_conditional_edges("extract_documents", route_after_extraction,
                            {"evaluate": "evaluate_suppliers", "fail": "fail_run"})
    g.add_edge("fail_run", END)
    g.add_edge("evaluate_suppliers", "validate_scorecards")
    g.add_edge("validate_scorecards", "rank_suppliers")
    g.add_edge("rank_suppliers", "persist_results")
    g.add_edge("persist_results", END)
    return g.compile()


def run_evaluation(supplier_inputs: list[dict], evaluator, rfp_title: str = "", db_path=None,
                   progress: Callable[[str, float], None] | None = None) -> dict:
    """Run the whole workflow and return the final state."""
    graph = build_graph(evaluator, progress)
    initial: RunState = {"rfp_title": rfp_title, "supplier_inputs": supplier_inputs, "db_path": db_path,
                         "warnings": [], "trace": [_t("2 Input", "Streamlit UI",
                                                      f"{len(supplier_inputs)} supplier PDF(s) received")]}
    return graph.invoke(initial)
