"""Templates, prompts, RAG, report and LLM client behaviour."""

import json
from types import SimpleNamespace

import pytest

from rfp import db, prompts, rag_tool
from rfp.document_tool import extract_pdf_text
from rfp.evaluation_agent import LLMEvaluator, OfflineDemoEvaluator
from rfp.llm_client import LLMClient, LLMError, friendly_error
from rfp.orchestrator import run_evaluation
from rfp.report_tool import build_run_report
from rfp.template_agent import clean_scheme, draft_scheme, normalise_weights

from conftest import SAMPLES, proposal_inputs


# ---------- templates ----------
def test_normalise_weights_totals_100():
    assert sum(normalise_weights([1, 1, 1])) == 100
    assert normalise_weights([30, 20, 20, 20, 10]) == [30, 20, 20, 20, 10]
    assert normalise_weights([0, 0]) == [50, 50]


def test_clean_scheme_rescales_and_dedupes():
    info, crit, notes = clean_scheme({"rfp_title": "X", "criteria": [
        {"name": "Cost", "weight": "40%"}, {"name": "cost", "weight": 10}, {"name": "Quality", "weight": 40}]})
    assert [c["name"] for c in crit] == ["Cost", "Quality"] and sum(c["weight"] for c in crit) == 100
    assert any("rescaled" in n for n in notes)            # 40+40 -> 50/50
    assert info["name"] == "X"


def test_rules_parser_reads_both_sample_templates(tmp_db):
    for f, first, n in [("Kestrel_HR_Service_Delivery_RFP.pdf", "Technical Capability", 5),
                        ("Meridian_Fleet_Lease_RFQ.pdf", "Total Cost of Lease", 5)]:
        text = extract_pdf_text((SAMPLES / "templates" / f).read_bytes(), f).text
        s = draft_scheme(text, f, None, tmp_db)
        assert s["mode"] == "RULES" and len(s["criteria"]) == n and s["criteria"][0]["name"] == first
        assert sum(c["weight"] for c in s["criteria"]) == 100
        assert all(c["industry_benchmark"] for c in s["criteria"])
        assert s["template"]["requirements"]


def test_template_without_criteria_falls_back_with_note(tmp_db):
    s = draft_scheme("We would like to buy office chairs for 200 staff. " * 10, "chairs.pdf", None, tmp_db)
    assert len(s["criteria"]) == 5 and any("No criteria were found" in n for n in s["notes"])


# ---------- prompts ----------
def test_prompt_checks_and_versioning(tmp_db):
    assert prompts.check_prompt("evaluation_user", prompts.EVALUATION_USER) == []
    bad = prompts.check_prompt("evaluation_user", "Score {supplier_name} please, this is long enough text.")
    assert any("Missing required" in p for p in bad)
    assert any("Unknown" in p for p in prompts.check_prompt("qa_user", prompts.QA_USER + " {oops}"))
    v = db.save_prompt_version("evaluation_system", prompts.EVALUATION_SYSTEM + "\nBe strict.", "stricter", tmp_db)
    assert v == 1 and db.get_prompt("evaluation_system", tmp_db)["version"] == 1
    db.activate_prompt_version("evaluation_system", 0, tmp_db)
    assert db.get_prompt("evaluation_system", tmp_db)["version"] == 0
    with pytest.raises(ValueError):
        db.save_prompt_version("qa_user", "no placeholders here at all, just words and more words", "", tmp_db)


def test_prompt_change_invalidates_dna(tmp_db):
    run_evaluation(1, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)
    db.save_prompt_version("evaluation_system", prompts.EVALUATION_SYSTEM + "\nBe strict.", "", tmp_db)
    r = run_evaluation(1, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    assert not any(s["dna_reused"] for s in r["suppliers"])
    assert r["prompt_versions"]["evaluation_system"] == 1


# ---------- RAG ----------
def test_chunking_keeps_pages_and_retrieval_covers_suppliers(tmp_db):
    r = run_evaluation(1, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    docs = {k: v["text"] for k, v in db.get_run_documents(r["rfp_run_id"], tmp_db).items()}
    chunks = rag_tool.build_index(docs)
    assert {c.page for c in chunks} >= {1, 2}
    hits = rag_tool.retrieve("ISO 27001 certification SOC 2", chunks)
    assert len({c.supplier for c, _ in hits}) >= 4
    assert any("27001" in c.text for c, _ in hits[:3])
    ans = rag_tool.answer("Which suppliers mention SOC 2?", r["rfp_run_id"], ["Apex Systems"], None, tmp_db)
    assert ans["mode"] == "extractive" and all(s["supplier"] == "Apex Systems" for s in ans["sources"])


# ---------- report ----------
def test_pdf_report_builds(tmp_db):
    r = run_evaluation(1, proposal_inputs(), OfflineDemoEvaluator(), db_path=tmp_db)["result"]
    pdf = build_run_report(r)
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000


# ---------- LLM client / evaluator with a fake API ----------
class FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])


def fake_client(replies):
    c = LLMClient("Cohere", "test-key")
    fake = FakeCompletions(replies)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return c, fake


GOOD = json.dumps({"supplier_name": "Apex", "criteria": [
    {"criterion_id": 1, "score": 8, "industry_score": 7, "max_score": 10, "justification": "ok", "evidence": "x"}]})


def test_cohere_defaults_and_json_mode(tmp_db):
    c, fake = fake_client([GOOD])
    assert c.label == "Cohere:command-a-03-2025"
    tmpl = db.get_template(1, tmp_db)
    crit = db.get_criteria(1, tmp_db)
    r = LLMEvaluator(c, tmp_db).evaluate("Apex", "doc", "sha", crit, tmpl)
    assert r.error == "" and fake.calls[0]["temperature"] == 0.0
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    user = fake.calls[0]["messages"][1]["content"]
    assert "Industry best practice" in user and "Kestrel" in user


def test_repair_retry_then_success(tmp_db):
    c, fake = fake_client(["not json", GOOD])
    r = LLMEvaluator(c, tmp_db).evaluate("Apex", "doc", "sha", db.get_criteria(1, tmp_db), db.get_template(1, tmp_db))
    assert r.attempts == 2 and r.error == ""


def test_api_error_is_friendly(tmp_db):
    c, fake = fake_client([RuntimeError("Error code: 401 - invalid api key")])
    r = LLMEvaluator(c, tmp_db).evaluate("Apex", "doc", "sha", db.get_criteria(1, tmp_db), db.get_template(1, tmp_db))
    assert r.raw_text == "" and "rejected" in r.error
    assert "Rate limit" in friendly_error(RuntimeError("429 Too Many Requests"))


def test_template_agent_uses_llm_when_available(tmp_db):
    reply = json.dumps({"rfp_title": "Office chairs", "buyer": "ACME", "industry": "Facilities",
                        "summary": "Chairs.", "key_requirements": ["200 chairs"],
                        "criteria": [{"name": "Price", "weight": 60, "description": "cost", "industry_benchmark": "b"},
                                     {"name": "Ergonomics", "weight": 40, "description": "fit", "industry_benchmark": "b"}]})
    c, fake = fake_client([reply])
    s = draft_scheme("RFP text " * 50, "chairs.pdf", c, tmp_db)
    assert s["mode"] == "LLM" and [x["name"] for x in s["criteria"]] == ["Price", "Ergonomics"]
