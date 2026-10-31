"""Ranking Tool: deterministic arithmetic, peer benchmarking, tie-breaks and ranks.

Pure Python, no LLM, no randomness, no I/O. Given the same validated
scorecards and metadata it always returns the same numbers and order.

Formulas (brief section 5)
--------------------------
absolute weighted score = sum over criteria of (score / max_score) x weight
                          -> 0..100 because active weights total 100
criterion benchmark     = highest validated score for that criterion across suppliers
criterion gap           = supplier score - benchmark (0 for the leader, otherwise negative)
relative performance %  = supplier score / benchmark x 100
                          if benchmark == 0 (nobody scored), relative % = 0 for everyone:
                          the criterion cannot differentiate suppliers, so it contributes
                          nothing to anyone's PPI and a warning is recorded
Peer Performance Index  = sum(relative % x weight) / sum(weights)
Industry Alignment Index = sum((industry score / max score) x weight) -> 0..100
                          how close the proposal is to industry best practice
                          (shown alongside; ranking stays on PPI as the brief requires)

Mandatory tie-break order
-------------------------
1) higher PPI  2) earlier submission date  3) higher historical experience rating
4) supplier name ascending (case-insensitive). Ranks 1..n are assigned only after
this single stable sort, so there are never shared ranks.
"""

from __future__ import annotations

from datetime import date

from . import config

R = config.SCORE_DECIMALS


def _r(x: float) -> float:
    return round(float(x) + 0.0, R)


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def sort_key(s: dict):
    return (
        -s["ppi"],
        _as_date(s["submission_date"]),
        -float(s["experience_rating"]),
        s["supplier_name"].casefold(),
    )


def compute_rankings(suppliers: list[dict], criteria: list[dict]) -> dict:
    """
    suppliers: [{supplier_name, submission_date (YYYY-MM-DD), experience_rating,
                 criteria: [{criterion_id, score, ...}], ...}]
    criteria:  active criteria rows from the database.

    Returns {"suppliers": ranked list, "benchmarks": {...}, "tie_breaks": [...], "warnings": [...]}
    """
    warnings: list[str] = []
    if not suppliers:
        return {"suppliers": [], "benchmarks": {}, "tie_breaks": [], "warnings": ["No suppliers to rank."]}

    total_weight = sum(float(c["weight"]) for c in criteria)
    if total_weight <= 0:
        raise ValueError("Active criteria weights must total more than zero.")
    if abs(total_weight - 100) > 1e-6:
        warnings.append(f"Active weights total {total_weight:g}%, not 100%; PPI is still normalised by total weight.")

    # 1. Benchmarks: highest validated score per criterion
    benchmarks: dict[int, dict] = {}
    for c in criteria:
        cid = c["criterion_id"]
        scores = []
        for s in suppliers:
            match = next((x for x in s["criteria"] if x["criterion_id"] == cid), None)
            scores.append((match["score"] if match else 0, s["supplier_name"]))
        best = max(score for score, _ in scores)
        leaders = sorted(name for score, name in scores if score == best)
        benchmarks[cid] = {"criterion_id": cid, "name": c["name"], "benchmark_score": best,
                           "peer_average": _r(sum(sc for sc, _ in scores) / len(scores)),
                           "max_score": c["max_score"], "weight": float(c["weight"]), "leaders": leaders}
        if best == 0:
            warnings.append(f"No supplier scored above 0 on '{c['name']}'; relative performance set to 0% "
                            "for all suppliers (criterion does not differentiate).")

    # 2. Per-supplier metrics
    ranked = []
    for s in suppliers:
        absolute = 0.0
        ppi_num = 0.0
        industry = 0.0
        rows = []
        for c in criteria:
            cid, weight, max_score = c["criterion_id"], float(c["weight"]), int(c["max_score"])
            item = next((x for x in s["criteria"] if x["criterion_id"] == cid), None)
            score = float(item["score"]) if item else 0.0
            ind_score = float(item.get("industry_score", score)) if item else 0.0
            bench = float(benchmarks[cid]["benchmark_score"])
            weighted_points = score / max_score * weight
            relative = (score / bench * 100.0) if bench > 0 else 0.0
            absolute += weighted_points
            ppi_num += relative * weight
            industry += ind_score / max_score * weight
            row = dict(item) if item else {"criterion_id": cid, "justification": "", "evidence": "",
                                           "evidence_verified": False, "status": "MISSING"}
            row.update({
                "name": c["name"], "weight": weight, "max_score": max_score, "score": score,
                "normalised_score_pct": _r(score / max_score * 100),
                "weighted_points": _r(weighted_points),
                "benchmark_score": bench,
                "gap": _r(score - bench),
                "relative_pct": _r(relative),
                "is_benchmark_leader": bench > 0 and score == bench,
                "industry_score": ind_score,
                "industry_pct": _r(ind_score / max_score * 100),
                "peer_average": benchmarks[cid]["peer_average"],
            })
            rows.append(row)
        out = dict(s)
        out["criteria"] = rows
        out["absolute_score"] = _r(absolute)
        out["ppi"] = _r(ppi_num / total_weight)
        out["industry_index"] = _r(industry * 100 / total_weight)
        out["experience_rating"] = float(s["experience_rating"])
        out["submission_date"] = _as_date(s["submission_date"]).isoformat()
        ranked.append(out)

    # 3. One stable sort with the mandatory tie-break order, then sequential ranks
    ranked.sort(key=sort_key)
    for i, s in enumerate(ranked, start=1):
        s["final_rank"] = i

    # 4. Explain every adjacent decision so the order is auditable
    tie_breaks = []
    for upper, lower in zip(ranked, ranked[1:]):
        tie_breaks.append(explain_pair(upper, lower))
    for s in ranked:
        s["rank_explanation"] = next(
            (t["explanation"] for t in tie_breaks if t["lower"] == s["supplier_name"]),
            "Highest Peer Performance Index in this run." if s["final_rank"] == 1 else "",
        )

    return {
        "suppliers": ranked,
        "benchmarks": list(benchmarks.values()),
        "tie_breaks": tie_breaks,
        "warnings": warnings,
    }


def explain_pair(upper: dict, lower: dict) -> dict:
    """Which rule placed `upper` above `lower`?"""
    u, l = upper, lower
    if u["ppi"] != l["ppi"]:
        rule, text = "1. Higher PPI", f"PPI {u['ppi']:.2f} vs {l['ppi']:.2f}"
    elif u["submission_date"] != l["submission_date"]:
        rule, text = "2. Earlier submission date", f"PPI tied at {u['ppi']:.2f}; submitted {u['submission_date']} vs {l['submission_date']}"
    elif u["experience_rating"] != l["experience_rating"]:
        rule, text = "3. Higher experience rating", (f"PPI and submission date tied; experience rating "
                                                    f"{u['experience_rating']:g} vs {l['experience_rating']:g}")
    else:
        rule, text = "4. Supplier name (A-Z)", "PPI, submission date and experience rating all tied; alphabetical order applied"
    return {
        "upper": u["supplier_name"], "upper_rank": u["final_rank"],
        "lower": l["supplier_name"], "lower_rank": l["final_rank"],
        "deciding_rule": rule,
        "is_tie_break": not rule.startswith("1."),
        "explanation": f"Placed below {u['supplier_name']} by rule {rule}: {text}.",
    }


def apply_overrides(scorecards: list[dict], overrides: list[dict]) -> list[dict]:
    """Return copies of the scorecards with analyst overrides applied (latest override wins)."""
    import copy

    cards = copy.deepcopy(scorecards)
    for o in overrides:
        for s in cards:
            if s["supplier_name"] != o["supplier_name"]:
                continue
            for c in s["criteria"]:
                if c["criterion_id"] == o["criterion_id"]:
                    c["original_score"] = c.get("original_score", c["score"])
                    c["score"] = float(o["new_score"])
                    c["overridden"] = True
                    c["override_reason"] = o["reason"]
    return cards
