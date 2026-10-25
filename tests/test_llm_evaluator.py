"""LLMEvaluator behaviour with a fake OpenAI client (no network)."""

import json
from types import SimpleNamespace

import pytest

from rfp import db
from rfp.evaluation_agent import LLMEvaluator

CRITERIA = [{"criterion_id": 1, "name": "Technical", "description": "tech", "weight": 100, "max_score": 10}]
GOOD = json.dumps({"supplier_name": "Apex", "criteria": [
    {"criterion_id": 1, "score": 8, "max_score": 10, "justification": "ok", "evidence": "x"}]})


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make(tmp_path, replies, use_cache=True):
    path = tmp_path / "t.db"
    db.init_db(path)
    ev = LLMEvaluator("OpenRouter", "test-key", "test/model", use_cache=use_cache, db_path=path)
    fake = FakeCompletions(replies)
    ev.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return ev, fake


def test_json_mode_and_temperature_zero(tmp_path):
    ev, fake = make(tmp_path, [GOOD])
    r = ev.evaluate("Apex", "doc", "sha1", CRITERIA)
    assert r.error == "" and r.attempts == 1
    assert fake.calls[0]["temperature"] == 0.0
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert "criterion_id 1: Technical" in fake.calls[0]["messages"][1]["content"]


def test_repair_retry_after_bad_json(tmp_path):
    ev, fake = make(tmp_path, ["Sorry, here is my answer: not json", GOOD])
    r = ev.evaluate("Apex", "doc", "sha1", CRITERIA)
    assert r.attempts == 2 and r.error == ""
    assert "could not be parsed" in fake.calls[1]["messages"][-1]["content"]


def test_cache_returns_identical_result_without_calling_llm(tmp_path):
    ev, fake = make(tmp_path, [GOOD])
    first = ev.evaluate("Apex", "doc", "sha1", CRITERIA)
    second = ev.evaluate("Apex", "doc", "sha1", CRITERIA)
    assert second.from_cache and second.raw_text == first.raw_text and len(fake.calls) == 1


def test_changed_criteria_bypass_cache(tmp_path):
    ev, fake = make(tmp_path, [GOOD, GOOD])
    ev.evaluate("Apex", "doc", "sha1", CRITERIA)
    changed = [dict(CRITERIA[0], description="new wording")]
    r = ev.evaluate("Apex", "doc", "sha1", changed)
    assert not r.from_cache and len(fake.calls) == 2


def test_api_failure_is_reported_not_raised(tmp_path):
    ev, fake = make(tmp_path, [])
    def boom(**kw):
        raise RuntimeError("401 invalid key")
    fake.create = boom
    r = ev.evaluate("Apex", "doc", "sha1", CRITERIA)
    assert r.raw_text == "" and "401" in r.error
