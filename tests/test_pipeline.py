"""End-to-end test of the LangGraph workflow using the offline evaluator (no API key)."""

from pathlib import Path

import pytest

from rfp import db
from rfp.evaluation_agent import OfflineDemoEvaluator
from rfp.orchestrator import run_evaluation

ROOT = Path(__file__).resolve().parent.parent
PDFS = ROOT / "sample_data" / "supplier_pdfs"
EDGE = ROOT / "sample_data" / "edge_cases"

META = {
    "Apex_Systems_RFP_Response.pdf": ("Apex Systems", "2026-10-06", 4),
    "BrightPath_Tech_RFP_Response.pdf": ("BrightPath Tech", "2026-10-03", 2),
    "NexaWorks_RFP_Response.pdf": ("NexaWorks", "2026-10-07", 4),
    "Orbit_Digital_RFP_Response.pdf": ("Orbit Digital", "2026-10-07", 5),
}


@pytest.fixture()
def tmp_db(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


def inputs(extra=()):
    rows = []
    for f, (name, d, r) in META.items():
        rows.append({"supplier_name": name, "submission_date": d, "experience_rating": r,
                     "filename": f, "file_bytes": (PDFS / f).read_bytes()})
    for f in extra:
        rows.append({"supplier_name": Path(f).stem, "submission_date": "2026-10-01", "experience_rating": 3,
                     "filename": Path(f).name, "file_bytes": (EDGE / f).read_bytes()})
    return rows


def test_full_run_is_persisted_and_ranked(tmp_db):
    state = run_evaluation(inputs(), OfflineDemoEvaluator(), "Test RFP", db_path=tmp_db)
    assert state["status"] == "COMPLETED"
    result = state["result"]
    assert [s["final_rank"] for s in result["suppliers"]] == [1, 2, 3, 4]
    assert len(result["execution_trace"]) >= 10
    saved = db.load_run(result["rfp_run_id"], tmp_db)
    assert saved["suppliers"][0]["supplier_name"] == result["suppliers"][0]["supplier_name"]
    runs = db.list_runs(tmp_db)
    assert runs[0]["status"] == "COMPLETED" and runs[0]["suppliers"] == 4


def test_same_inputs_same_order(tmp_db):
    a = run_evaluation(inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    b = run_evaluation(inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    strip = lambda r: [(s["supplier_name"], s["absolute_score"], s["ppi"], s["final_rank"]) for s in r["suppliers"]]
    assert strip(a) == strip(b)


def test_bad_files_are_skipped_not_fatal(tmp_db):
    state = run_evaluation(inputs(["Scanned_Proposal_No_Text.pdf", "Corrupt_File.pdf"]),
                           OfflineDemoEvaluator(), db_path=tmp_db)
    result = state["result"]
    assert len(result["suppliers"]) == 4
    assert {n["filename"] for n in result["not_evaluated"]} == {"Scanned_Proposal_No_Text.pdf", "Corrupt_File.pdf"}


def test_all_files_bad_fails_cleanly(tmp_db):
    rows = [r for r in inputs(["Corrupt_File.pdf"]) if r["filename"] == "Corrupt_File.pdf"]
    state = run_evaluation(rows, OfflineDemoEvaluator(), db_path=tmp_db)
    assert state["status"] == "FAILED"
    assert db.list_runs(tmp_db)[0]["status"] == "FAILED"


def test_simulated_malformed_llm_output_is_normalised(tmp_db):
    ev = OfflineDemoEvaluator(simulate_malformed_for={"BrightPath Tech"})
    result = run_evaluation(inputs(), ev, db_path=tmp_db)["result"]
    bp = next(s for s in result["suppliers"] if s["supplier_name"] == "BrightPath Tech")
    assert len(bp["validation_warnings"]) >= 4
    assert all(0 <= c["score"] <= c["max_score"] for c in bp["criteria"])


def test_identical_document_ties_and_is_broken_by_rule_3(tmp_db):
    rows = inputs()
    nexa = next(r for r in rows if r["supplier_name"] == "NexaWorks")
    rows.append({**nexa, "supplier_name": "NexaWorks Twin", "experience_rating": 3,
                 "filename": "NexaWorks_Twin_Copy.pdf"})
    result = run_evaluation(rows, OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    order = [s["supplier_name"] for s in result["suppliers"]]
    a, b = (next(s for s in result["suppliers"] if s["supplier_name"] == n) for n in ("NexaWorks", "NexaWorks Twin"))
    assert a["ppi"] == b["ppi"] and order.index("NexaWorks") + 1 == order.index("NexaWorks Twin")
    tb = next(t for t in result["tie_breaks"] if t["lower"] == "NexaWorks Twin")
    assert tb["deciding_rule"] == "3. Higher experience rating"
    assert any("identical to NexaWorks" in w for w in result["warnings"])


def test_supplier_entries_created_at_batch_and_statuses_recorded(tmp_db):
    result = run_evaluation(inputs(["Corrupt_File.pdf"]), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    with db.connect(tmp_db) as conn:
        rows = conn.execute("SELECT supplier_name, status, final_rank FROM supplier_results WHERE rfp_run_id=?",
                            (result["rfp_run_id"],)).fetchall()
    status = {r["supplier_name"]: r["status"] for r in rows}
    assert status["Corrupt_File"] == "NOT_EVALUATED"
    assert sum(v == "RANKED" for v in status.values()) == 4 and len(rows) == 5
