"""Unit tests for the deterministic Ranking Tool."""

from rfp.ranking_tool import apply_overrides, compute_rankings

CRITERIA = [{"criterion_id": 1, "name": "Technical", "weight": 60, "max_score": 10},
            {"criterion_id": 2, "name": "Commercial", "weight": 40, "max_score": 10}]


def supplier(name, scores, date="2026-10-01", rating=3, industry=None):
    return {"supplier_name": name, "submission_date": date, "experience_rating": rating,
            "criteria": [{"criterion_id": cid, "score": s, "industry_score": (industry or {}).get(cid, s),
                          "justification": "", "evidence": "", "evidence_verified": True, "status": "OK"}
                         for cid, s in scores.items()]}


def by_name(r):
    return {s["supplier_name"]: s for s in r["suppliers"]}


def test_formulas_match_hand_calculation():
    r = compute_rankings([supplier("A", {1: 8, 2: 5}, industry={1: 6, 2: 5}), supplier("B", {1: 4, 2: 10})], CRITERIA)
    a, b = by_name(r)["A"], by_name(r)["B"]
    assert a["absolute_score"] == 68.0 and b["absolute_score"] == 64.0
    ta = next(c for c in a["criteria"] if c["criterion_id"] == 1)
    ca = next(c for c in a["criteria"] if c["criterion_id"] == 2)
    assert (ta["benchmark_score"], ta["gap"], ta["relative_pct"]) == (8, 0, 100)
    assert (ca["benchmark_score"], ca["gap"], ca["relative_pct"]) == (10, -5, 50)
    assert a["ppi"] == 80.0 and b["ppi"] == 70.0
    assert a["industry_index"] == 56.0        # 6/10*60 + 5/10*40
    assert ta["peer_average"] == 6.0
    assert a["final_rank"] == 1


def test_tie_break_order_date_then_rating_then_name():
    same = {1: 7, 2: 7}
    r = compute_rankings([supplier("Zeta", same, "2026-10-05", 5), supplier("Beta", same, "2026-10-02", 2),
                          supplier("Alpha", same, "2026-10-05", 5), supplier("Gamma", same, "2026-10-05", 4)],
                         CRITERIA)
    assert [s["supplier_name"] for s in r["suppliers"]] == ["Beta", "Alpha", "Zeta", "Gamma"]
    assert [t["deciding_rule"] for t in r["tie_breaks"]] == [
        "2. Earlier submission date", "4. Supplier name (A-Z)", "3. Higher experience rating"]


def test_ranking_independent_of_input_order():
    s = [supplier("A", {1: 9, 2: 3}), supplier("B", {1: 6, 2: 9}), supplier("C", {1: 6, 2: 9}, rating=4)]
    order = lambda x: [y["supplier_name"] for y in compute_rankings(x, CRITERIA)["suppliers"]]
    assert order(s) == order(list(reversed(s)))


def test_zero_benchmark_is_safe():
    r = compute_rankings([supplier("A", {1: 5, 2: 0}), supplier("B", {1: 4, 2: 0})], CRITERIA)
    for s in r["suppliers"]:
        comm = next(c for c in s["criteria"] if c["criterion_id"] == 2)
        assert comm["relative_pct"] == 0 and comm["gap"] == 0
    assert any("does not differentiate" in w for w in r["warnings"])


def test_absolute_and_ppi_can_disagree():
    crit = [{"criterion_id": 1, "name": "T", "weight": 50, "max_score": 10},
            {"criterion_id": 2, "name": "C", "weight": 50, "max_score": 10}]
    r = by_name(compute_rankings([supplier("X", {1: 10, 2: 2}), supplier("Y", {1: 5, 2: 6})], crit))
    assert r["X"]["absolute_score"] > r["Y"]["absolute_score"] and r["Y"]["ppi"] > r["X"]["ppi"]
    assert r["Y"]["final_rank"] == 1


def test_overrides_apply_latest_and_keep_original():
    cards = [supplier("A", {1: 5, 2: 5})]
    out = apply_overrides(cards, [{"supplier_name": "A", "criterion_id": 1, "new_score": 7, "reason": "x"},
                                  {"supplier_name": "A", "criterion_id": 1, "new_score": 8, "reason": "y"}])
    c = out[0]["criteria"][0]
    assert c["score"] == 8 and c["original_score"] == 5 and c["override_reason"] == "y"
    assert cards[0]["criteria"][0]["score"] == 5   # input untouched
