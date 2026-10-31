"""End-to-end LangGraph workflow tests with the offline evaluator (no API key)."""

from rfp import db
from rfp.evaluation_agent import FaultInjectingEvaluator, OfflineDemoEvaluator
from rfp.orchestrator import rescore_run, run_evaluation

from conftest import proposal_inputs

SEEDED = 1  # the sample Kestrel template created by init_db


def test_full_run_is_persisted_and_ranked(tmp_db):
    state = run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)
    assert state["status"] == "COMPLETED"
    r = state["result"]
    assert [s["final_rank"] for s in r["suppliers"]] == [1, 2, 3, 4, 5]
    assert all(0 <= s["industry_index"] <= 100 for s in r["suppliers"])
    assert db.load_run(r["rfp_run_id"], tmp_db)["suppliers"][0]["supplier_name"] == r["suppliers"][0]["supplier_name"]
    assert len(db.get_run_documents(r["rfp_run_id"], tmp_db)) == 5
    assert db.list_runs(tmp_db)[0]["suppliers"] == 5


def test_same_inputs_same_order_and_dna_reuse(tmp_db):
    a = run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    b = run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    strip = lambda r: [(s["supplier_name"], s["absolute_score"], s["ppi"], s["final_rank"]) for s in r["suppliers"]]
    assert strip(a) == strip(b)
    assert all(s["dna_reused"] and s["dna_origin_run"] == a["rfp_run_id"] for s in b["suppliers"])
    assert all(d["reuse_count"] == 1 for d in db.list_dna(tmp_db))


def test_dna_disabled_means_fresh_evaluation(tmp_db):
    run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)
    b = run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), use_dna=False, db_path=tmp_db)["result"]
    assert not any(s["dna_reused"] for s in b["suppliers"])


def test_changed_criteria_invalidate_dna(tmp_db):
    run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)
    crit = db.get_criteria(SEEDED, tmp_db, active_only=False)
    crit[0]["description"] += " Include mobile access."
    db.save_criteria(SEEDED, crit, tmp_db)
    b = run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    assert not any(s["dna_reused"] for s in b["suppliers"])


def test_bad_files_are_skipped_not_fatal(tmp_db):
    r = run_evaluation(SEEDED, proposal_inputs(extra_edge=["Scanned_Proposal_No_Text.pdf", "Corrupt_File.pdf"]),
                       OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    assert len(r["suppliers"]) == 5
    assert {n["filename"] for n in r["not_evaluated"]} == {"Scanned_Proposal_No_Text.pdf", "Corrupt_File.pdf"}
    with db.connect(tmp_db) as conn:
        status = dict(conn.execute("SELECT supplier_name, status FROM supplier_results WHERE rfp_run_id=?",
                                   (r["rfp_run_id"],)).fetchall())
    assert status["Corrupt_File"] == "NOT_EVALUATED" and list(status.values()).count("RANKED") == 5


def test_all_files_bad_fails_cleanly(tmp_db):
    rows = [r for r in proposal_inputs(extra_edge=["Corrupt_File.pdf"]) if r["filename"] == "Corrupt_File.pdf"]
    state = run_evaluation(SEEDED, rows, OfflineDemoEvaluator(), db_path=tmp_db)
    assert state["status"] == "FAILED" and db.list_runs(tmp_db)[0]["status"] == "FAILED"


def test_simulated_malformed_output_is_normalised(tmp_db):
    ev = FaultInjectingEvaluator(OfflineDemoEvaluator(), {"BrightPath Tech"})
    r = run_evaluation(SEEDED, proposal_inputs(), ev, db_path=tmp_db)["result"]
    bp = next(s for s in r["suppliers"] if s["supplier_name"] == "BrightPath Tech")
    text = " ".join(bp["validation_warnings"])
    for expected in ["clipped", "non-numeric", "missing from LLM output", "unknown criterion_id", "could not be found"]:
        assert expected in text
    assert all(0 <= c["score"] <= c["max_score"] for c in bp["criteria"])


def test_identical_document_ties_and_rule_3_decides(tmp_db):
    rows = proposal_inputs()
    nexa = next(r for r in rows if r["supplier_name"] == "NexaWorks")
    rows.append({**nexa, "supplier_name": "NexaWorks Twin", "experience_rating": 3, "filename": "twin.pdf"})
    r = run_evaluation(SEEDED, rows, OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    tb = next(t for t in r["tie_breaks"] if t["lower"] == "NexaWorks Twin")
    assert tb["deciding_rule"] == "3. Higher experience rating"
    assert any("identical to NexaWorks" in w for w in r["warnings"])


def test_override_rescore_changes_rank_without_ai(tmp_db):
    r = run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    last = r["suppliers"][-1]
    for c in last["criteria"]:
        db.add_override(r["rfp_run_id"], last["supplier_name"], c["criterion_id"], c["name"], c["score"],
                        c["max_score"], "Panel re-review", db_path=tmp_db)
    new = rescore_run(r["rfp_run_id"], tmp_db)
    assert new["revision"] == 2 and new["suppliers"][0]["supplier_name"] == last["supplier_name"]
    assert any(w.startswith("Analyst override:") for w in new["warnings"])
    assert db.load_run(r["rfp_run_id"], tmp_db)["revision"] == 2


def test_second_domain_template_end_to_end(tmp_db):
    from pathlib import Path
    from rfp.document_tool import extract_pdf_text
    from rfp.template_agent import draft_scheme
    from conftest import SAMPLES
    data = (SAMPLES / "templates" / "Meridian_Fleet_Lease_RFQ.pdf").read_bytes()
    scheme = draft_scheme(extract_pdf_text(data, "rfq.pdf").text, "rfq.pdf", None, tmp_db)
    tid = db.save_template({**scheme["template"], "parse_mode": "RULES"}, scheme["criteria"], tmp_db)
    r = run_evaluation(tid, proposal_inputs("fleet_lease"), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    assert [c["name"] for c in r["criteria"]][0] == "Total Cost of Lease" and len(r["suppliers"]) == 3


def test_fault_injection_never_reads_or_writes_dna_for_faulted_supplier(tmp_db):
    run_evaluation(SEEDED, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)
    ev = FaultInjectingEvaluator(OfflineDemoEvaluator(), {"BrightPath Tech"})
    r = run_evaluation(SEEDED, proposal_inputs(), ev, db_path=tmp_db)["result"]
    by = {s["supplier_name"]: s for s in r["suppliers"]}
    assert not by["BrightPath Tech"]["dna_reused"] and by["BrightPath Tech"]["validation_warnings"]
    assert all(by[n]["dna_reused"] for n in by if n != "BrightPath Tech")
    assert len(db.list_dna(tmp_db)) == 5
