"""Unit tests for the Validation Tool."""

import json
from datetime import date

from rfp.validation_tool import (evidence_in_document, parse_llm_json, validate_scorecard,
                                 validate_supplier_inputs)

CRITERIA = [
    {"criterion_id": 1, "name": "Technical Capability", "description": "", "weight": 50, "max_score": 10},
    {"criterion_id": 2, "name": "Commercial Value", "description": "", "weight": 30, "max_score": 10},
    {"criterion_id": 3, "name": "Security & Compliance", "description": "", "weight": 20, "max_score": 10},
]
DOC = ("We provide a certified SAP SuccessFactors connector using OData v4 APIs. "
       "Three-year total cost of ownership is INR 2,48,42,500 excluding GST. "
       "ISO 27001:2022 certified and SOC 2 Type II report available under NDA.")


def test_parse_handles_fences_prose_and_trailing_commas():
    raw = 'Here you go:\n```json\n{"criteria": [{"criterion_id": 1, "score": 7,},],}\n```'
    obj, err = parse_llm_json(raw)
    assert err == "" and obj["criteria"][0]["score"] == 7


def test_parse_rejects_garbage():
    obj, err = parse_llm_json("I cannot help with that.")
    assert obj is None and err


def test_clean_scorecard_passes_with_verified_evidence():
    raw = json.dumps({"supplier_name": "Apex", "criteria": [
        {"criterion_id": 1, "score": 9, "max_score": 10, "justification": "Strong.",
         "evidence": "certified SAP SuccessFactors connector using OData v4 APIs"},
        {"criterion_id": 2, "score": 5, "max_score": 10, "justification": "Expensive.",
         "evidence": "Three-year total cost of ownership is INR 2,48,42,500 excluding GST"},
        {"criterion_id": 3, "score": 9, "max_score": 10, "justification": "Certified.",
         "evidence": "ISO 27001:2022 certified and SOC 2 Type II report available"},
    ], "risks": ["High cost"], "overall_summary": "Good."})
    v = validate_scorecard(raw, "Apex", CRITERIA, DOC)
    assert v.warnings == []
    assert [c.score for c in v.criteria] == [9, 5, 9]
    assert all(c.evidence_verified for c in v.criteria)


def test_malformed_scorecard_is_normalised_with_warnings():
    raw = json.dumps({"supplier_name": "Apex", "criteria": [
        {"criterion_id": 1, "score": 14, "justification": "x", "evidence": "made-up quote that is not in the text"},
        {"criterion_id": "2", "score": "eight", "justification": "", "evidence": ""},
        {"criterion_id": 99, "score": 5},
    ]})
    v = validate_scorecard(raw, "Apex", CRITERIA, DOC)
    scores = {c.criterion_id: c for c in v.criteria}
    assert scores[1].score == 10 and scores[1].status == "NORMALISED"   # clipped
    assert scores[1].evidence_verified is False                         # hallucinated quote flagged
    assert scores[2].score == 0 and scores[2].status == "NORMALISED"    # non-numeric
    assert scores[3].score == 0 and scores[3].status == "MISSING"       # omitted
    text = " ".join(v.warnings)
    for expected in ["clipped", "non-numeric", "missing from LLM output", "unknown criterion_id 99",
                     "could not be found in the document"]:
        assert expected in text


def test_unparseable_output_scores_zero_everywhere():
    v = validate_scorecard("not json at all", "Apex", CRITERIA, DOC)
    assert all(c.score == 0 for c in v.criteria)
    assert any("could not be parsed" in w for w in v.warnings)


def test_criterion_matched_by_name_when_id_missing():
    raw = json.dumps({"criteria": [{"criterion_name": "Commercial Value", "score": 6, "justification": "ok",
                                    "evidence": "Three-year total cost of ownership is INR 2,48,42,500"}]})
    v = validate_scorecard(raw, "Apex", CRITERIA, DOC)
    assert next(c for c in v.criteria if c.criterion_id == 2).score == 6


def test_evidence_fuzzy_match_tolerates_small_differences():
    assert evidence_in_document("ISO 27001:2022 certified, and SOC 2 Type-II report available", DOC)
    assert not evidence_in_document("We are CMMI level 5 appraised for all delivery centres", DOC)


def test_supplier_input_validation():
    rows = [
        {"filename": "a.pdf", "supplier_name": "Apex", "submission_date": date(2026, 10, 1), "experience_rating": 4},
        {"filename": "b.pdf", "supplier_name": "apex", "submission_date": date(2030, 1, 1), "experience_rating": 9},
        {"filename": "c.pdf", "supplier_name": "", "submission_date": "not-a-date", "experience_rating": None},
    ]
    errors = validate_supplier_inputs(rows, today=date(2026, 10, 25))
    joined = " ".join(errors)
    assert "used more than once" in joined
    assert "in the future" in joined
    assert "between 1 and 5" in joined
    assert "supplier name is required" in joined
    assert "valid date" in joined
    assert not any(e.startswith("Row 1") for e in errors)
