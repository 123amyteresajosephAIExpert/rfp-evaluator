"""Unit tests for the deterministic Ranking Tool."""

from rfp.ranking_tool import compute_rankings

CRITERIA = [
    {"criterion_id": 1, "name": "Technical", "weight": 60, "max_score": 10},
    {"criterion_id": 2, "name": "Commercial", "weight": 40, "max_score": 10},
]


def supplier(name, scores, date="2026-10-01", rating=3):
    return {
        "supplier_name": name,
        "submission_date": date,
        "experience_rating": rating,
        "criteria": [{"criterion_id": cid, "score": s, "justification": "", "evidence": "",
                      "evidence_verified": True, "status": "OK"} for cid, s in scores.items()],
    }


def by_name(result):
    return {s["supplier_name"]: s for s in result["suppliers"]}


def test_formulas_match_hand_calculation():
    r = compute_rankings([supplier("A", {1: 8, 2: 5}), supplier("B", {1: 4, 2: 10})], CRITERIA)
    a, b = by_name(r)["A"], by_name(r)["B"]
    # absolute = 8/10*60 + 5/10*40 = 48 + 20 = 68
    assert a["absolute_score"] == 68.0
    # B absolute = 24 + 40 = 64
    assert b["absolute_score"] == 64.0
    # benchmarks: technical 8 (A), commercial 10 (B)
    tech_a = next(c for c in a["criteria"] if c["criterion_id"] == 1)
    comm_a = next(c for c in a["criteria"] if c["criterion_id"] == 2)
    assert tech_a["benchmark_score"] == 8 and tech_a["gap"] == 0 and tech_a["relative_pct"] == 100
    assert comm_a["benchmark_score"] == 10 and comm_a["gap"] == -5 and comm_a["relative_pct"] == 50
    # PPI A = (100*60 + 50*40)/100 = 80 ; PPI B = (50*60 + 100*40)/100 = 70
    assert a["ppi"] == 80.0 and b["ppi"] == 70.0
    assert a["final_rank"] == 1 and b["final_rank"] == 2


def test_tie_break_order_date_then_rating_then_name():
    same = {1: 7, 2: 7}
    suppliers = [
        supplier("Zeta", same, date="2026-10-05", rating=5),
        supplier("Beta", same, date="2026-10-02", rating=2),      # earliest date -> 1st
        supplier("Alpha", same, date="2026-10-05", rating=5),     # same as Zeta -> name A-Z
        supplier("Gamma", same, date="2026-10-05", rating=4),     # lower rating than Alpha/Zeta
    ]
    r = compute_rankings(suppliers, CRITERIA)
    order = [s["supplier_name"] for s in r["suppliers"]]
    assert order == ["Beta", "Alpha", "Zeta", "Gamma"]
    rules = [t["deciding_rule"] for t in r["tie_breaks"]]
    assert rules == ["2. Earlier submission date", "4. Supplier name (A-Z)", "3. Higher experience rating"]
    assert [s["final_rank"] for s in r["suppliers"]] == [1, 2, 3, 4]


def test_ranking_is_independent_of_input_order():
    s = [supplier("A", {1: 9, 2: 3}), supplier("B", {1: 6, 2: 9}), supplier("C", {1: 6, 2: 9}, rating=4)]
    r1 = [x["supplier_name"] for x in compute_rankings(s, CRITERIA)["suppliers"]]
    r2 = [x["supplier_name"] for x in compute_rankings(list(reversed(s)), CRITERIA)["suppliers"]]
    assert r1 == r2


def test_zero_benchmark_is_safe():
    r = compute_rankings([supplier("A", {1: 5, 2: 0}), supplier("B", {1: 4, 2: 0})], CRITERIA)
    for s in r["suppliers"]:
        comm = next(c for c in s["criteria"] if c["criterion_id"] == 2)
        assert comm["relative_pct"] == 0 and comm["gap"] == 0
    assert any("does not differentiate" in w for w in r["warnings"])


def test_single_supplier_gets_full_ppi():
    r = compute_rankings([supplier("Solo", {1: 6, 2: 4})], CRITERIA)
    assert r["suppliers"][0]["ppi"] == 100.0 and r["suppliers"][0]["final_rank"] == 1


def test_absolute_and_ppi_can_disagree():
    """A supplier can lead on absolute score but not on PPI - the brief ranks on PPI."""
    crit = [{"criterion_id": 1, "name": "T", "weight": 50, "max_score": 10},
            {"criterion_id": 2, "name": "C", "weight": 50, "max_score": 10}]
    r = compute_rankings([supplier("X", {1: 10, 2: 2}), supplier("Y", {1: 5, 2: 6})], crit)
    x, y = by_name(r)["X"], by_name(r)["Y"]
    assert x["absolute_score"] > y["absolute_score"]  # 60 vs 55
    assert y["ppi"] > x["ppi"]                        # 75 vs 66.67
    assert y["final_rank"] == 1
