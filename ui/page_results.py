"""Results: leaderboard, peer and industry benchmarks, scorecards, comparison, overrides and exports."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from rfp import db
from rfp.orchestrator import FORMULAS, rescore_run
from rfp.report_tool import build_run_report
from ui.common import BLUE, INK, ORANGE, fmt, leaderboard_df, page_header, pill, run_picker


def _bar(df, x, y, sort_field, title, domain=(0, 100), fmt_=".2f"):
    return alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, size=22).encode(
        x=alt.X(f"{x}:Q", scale=alt.Scale(domain=list(domain)), title=title),
        y=alt.Y(f"{y}:N", sort=alt.SortField(sort_field), title=None),
        tooltip=[y, alt.Tooltip(f"{x}:Q", format=fmt_)],
    ).properties(height=42 * len(df) + 24)


def _heat(df, value, title, order_sup, order_crit, fmt_=".0f"):
    base = alt.Chart(df).encode(
        x=alt.X("Criterion:N", sort=order_crit, title=None, axis=alt.Axis(labelAngle=-25, labelLimit=200, labelOverlap=False)),
        y=alt.Y("Supplier:N", sort=order_sup, title=None))
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme="blues", domain=[0, 100]), legend=alt.Legend(title=title)),
        tooltip=["Supplier", "Criterion", alt.Tooltip(f"{value}:Q", format=".1f"), "Score"])
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=fmt_),
        color=alt.condition(f"datum['{value}'] > 60", alt.value("white"), alt.value(INK)))
    return (cells + text).properties(height=44 * len(order_sup) + 50)


def render():
    ss = st.session_state
    page_header("🏆 Results", "Explore any completed evaluation run. Everything shown is read back from SQLite.")
    run = run_picker()
    if not run:
        return
    sups = run["suppliers"]
    top = sups[0]
    order_sup = [s["supplier_name"] for s in sups]
    order_crit = [c["name"] for c in run["criteria"]]
    reused = sum(s.get("dna_reused", False) for s in sups)
    st.markdown(" ".join([
        pill(f"Template: {run.get('rfp_title')}"), pill(f"Engine: {run['evaluation_mode']}"),
        pill(f"Revision {run.get('revision', 1)}"),
        pill(f"{len(run['warnings'])} warning(s)", "warn" if run["warnings"] else "ok"),
        pill(f"{reused} reused from DNA", "ok") if reused else "",
        pill(f"{len(run.get('overrides', []))} analyst override(s)", "warn") if run.get("overrides") else "",
    ]), unsafe_allow_html=True)
    if "OFFLINE" in run["evaluation_mode"]:
        st.caption("⚠ This run used the offline keyword heuristic, not AI. Use an AI provider for real evaluations.")

    m = st.columns([2.4, 1, 1.3, 1.3, 1])
    m[0].metric("Recommended supplier", top["supplier_name"])
    m[1].metric("Top PPI", fmt(top["ppi"]))
    m[2].metric("Absolute score", f"{fmt(top['absolute_score'])}/100")
    m[3].metric("Industry alignment", f"{fmt(top['industry_index'], 1)}/100")
    m[4].metric("Suppliers ranked", len(sups), help=f"{len(run['not_evaluated'])} file(s) could not be evaluated")

    tabs = st.tabs(["🏆 Leaderboard", "👥 Peer benchmark", "🏭 Industry benchmark", "🔍 Scorecards", "⚖️ Compare",
                    "✍️ Adjust scores", "🧾 Run details & export"])

    # ------------------------------------------------------------ leaderboard
    with tabs[0]:
        board = leaderboard_df(run)
        st.dataframe(board, hide_index=True, use_container_width=True, column_config={
            "PPI": st.column_config.ProgressColumn("PPI", min_value=0, max_value=100, format="%.2f"),
            "Absolute score": st.column_config.ProgressColumn("Absolute score", min_value=0, max_value=100, format="%.2f"),
            "Industry alignment": st.column_config.ProgressColumn("Industry alignment", min_value=0, max_value=100, format="%.1f"),
            "Why this position": st.column_config.TextColumn("Why this position", width="large")})
        st.caption("Ranked by PPI, then earlier submission date, then higher experience rating, then supplier name "
                   "A-Z. PPI = weighted average of each criterion score as a % of the best peer score.")
        l, r = st.columns(2)
        with l:
            st.markdown("**Peer Performance Index**")
            st.altair_chart(_bar(board, "PPI", "Supplier", "Rank", "PPI (0-100)"), use_container_width=True)
        with r:
            st.markdown("**Absolute weighted score**")
            st.altair_chart(_bar(board, "Absolute score", "Supplier", "Rank", "Score (0-100)"), use_container_width=True)
        if run["not_evaluated"]:
            st.warning("Not evaluated: " + "; ".join(f"{n['supplier_name']} ({n['reason'][:90]})" for n in run["not_evaluated"]))

    # ------------------------------------------------------------ peer
    with tabs[1]:
        st.markdown("How each supplier compares with **the best answer among its peers** on every criterion.")
        heat = pd.DataFrame([{"Supplier": s["supplier_name"], "Criterion": c["name"], "Relative %": c["relative_pct"],
                              "Score": c["score"]} for s in sups for c in s["criteria"]])
        st.altair_chart(_heat(heat, "Relative %", "% of best", order_sup, order_crit), use_container_width=True)
        bench = pd.DataFrame([{"Criterion": b["name"], "Weight %": b.get("weight"), "Best score": b["benchmark_score"],
                               "Max": b["max_score"], "Peer average": b.get("peer_average"),
                               "Leader(s)": ", ".join(b["leaders"])} for b in run["benchmarks"]])
        st.dataframe(bench, hide_index=True, use_container_width=True)
        pick = st.selectbox("Show gaps to the peer benchmark for", order_sup, key="gap_pick")
        s = next(x for x in sups if x["supplier_name"] == pick)
        gaps = pd.DataFrame([{"Criterion": c["name"], "Gap": c["gap"], "Score": c["score"],
                              "Benchmark": c["benchmark_score"]} for c in s["criteria"]])
        chart = alt.Chart(gaps).mark_bar(color=ORANGE, cornerRadiusEnd=4, size=20).encode(
            x=alt.X("Gap:Q", title="Points behind the best peer (0 = leader)"),
            y=alt.Y("Criterion:N", sort=order_crit, title=None), tooltip=["Criterion", "Score", "Benchmark", "Gap"]
        ).properties(height=40 * len(gaps) + 24)
        st.altair_chart(chart, use_container_width=True)

    # ------------------------------------------------------------ industry
    with tabs[2]:
        st.markdown("How each proposal compares with **industry best practice** for each criterion, judged by the "
                    "AI against the benchmark written into the template. This is shown alongside the peer view; "
                    "the ranking itself stays on PPI, as the brief requires.")
        ind = pd.DataFrame([{"Supplier": s["supplier_name"], "Criterion": c["name"], "Industry %": c.get("industry_pct", 0),
                             "Score": c.get("industry_score", 0)} for s in sups for c in s["criteria"]])
        st.altair_chart(_heat(ind, "Industry %", "% of best practice", order_sup, order_crit), use_container_width=True)
        idx = pd.DataFrame([{"Supplier": s["supplier_name"], "Rank": s["final_rank"], "Industry alignment": s["industry_index"]}
                            for s in sups])
        st.markdown("**Industry Alignment Index** (weighted, 0-100)")
        st.altair_chart(_bar(idx, "Industry alignment", "Supplier", "Rank", "Industry alignment (0-100)", fmt_=".1f"),
                        use_container_width=True)
        with st.expander("Industry best practice used for each criterion"):
            for c in run["criteria"]:
                st.markdown(f"**{c['name']}** - {c.get('industry_benchmark') or 'general good practice'}")

    # ------------------------------------------------------------ scorecards
    with tabs[3]:
        names = [f"#{s['final_rank']} {s['supplier_name']}" for s in sups]
        pick = st.selectbox("Supplier", names, key="card_pick")
        s = sups[names.index(pick)]
        c = st.columns(5)
        c[0].metric("Rank", s["final_rank"])
        c[1].metric("PPI", fmt(s["ppi"]))
        c[2].metric("Absolute", fmt(s["absolute_score"]))
        c[3].metric("Industry", fmt(s["industry_index"], 1))
        c[4].metric("Evidence verified", f"{sum(x['evidence_verified'] for x in s['criteria'])}/{len(s['criteria'])}")
        badges = [pill(f"File: {s.get('source_file', '')}")]
        if s.get("dna_fingerprint"):
            badges.append(pill(f"DNA {s['dna_fingerprint'][:10]}", "ok" if s.get("dna_reused") else ""))
        if s.get("dna_reused"):
            badges.append(pill(f"Reused from {s.get('dna_origin_run') or 'earlier run'}", "ok"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)
        if s.get("overall_summary"):
            st.markdown(f"**Summary.** {s['overall_summary']}")
        st.caption(s["rank_explanation"])
        k = st.columns(3)
        for col, label, key in zip(k, ["✅ Strengths", "⚠️ Weaknesses", "🚩 Risks"], ["strengths", "weaknesses", "risks"]):
            with col:
                st.markdown(f"**{label}**")
                st.markdown("\n".join(f"- {x}" for x in s.get(key, [])) or "_none listed_")
        card = pd.DataFrame([{
            "Criterion": x["name"], "Weight %": x["weight"], "Score": x["score"], "Max": x["max_score"],
            "Benchmark": x["benchmark_score"], "Gap": x["gap"], "Relative %": x["relative_pct"],
            "Industry score": x.get("industry_score"), "Weighted points": x["weighted_points"],
            "Evidence": "✔ verified" if x["evidence_verified"] else "⚠ check",
            "Status": ("OVERRIDDEN" if x.get("overridden") else x.get("status", "")),
        } for x in s["criteria"]])
        st.dataframe(card, hide_index=True, use_container_width=True, column_config={
            k_: st.column_config.NumberColumn(k_, format="%.2f") for k_ in ["Gap", "Relative %", "Weighted points"]})
        st.markdown("**Evidence and justification**")
        for x in s["criteria"]:
            flag = "✔ found in the proposal" if x["evidence_verified"] else "⚠ not found in the proposal - review"
            with st.expander(f"{x['name']} · {x['score']:g}/{x['max_score']} · {flag}"):
                if x.get("overridden"):
                    st.info(f"Analyst override: AI score {x.get('original_score'):g} → {x['score']:g}. "
                            f"Reason: {x.get('override_reason')}")
                st.markdown(f"**Justification:** {x.get('justification', '')}")
                st.markdown(f"**Quoted evidence:** " + (f"> {x['evidence']}" if x.get("evidence") else "_none_"))
                st.caption(f"Industry score {x.get('industry_score', 0):g}/{x['max_score']} · benchmark {x['benchmark_score']:g} "
                           f"· gap {x['gap']:g} · relative {x['relative_pct']:.1f}% · {x['weighted_points']:.2f} of "
                           f"{x['weight']:g} weighted points")
        if s.get("validation_warnings"):
            with st.expander(f"Validation warnings ({len(s['validation_warnings'])})"):
                st.markdown("\n".join(f"- {w}" for w in s["validation_warnings"]))

    # ------------------------------------------------------------ compare
    with tabs[4]:
        if len(sups) < 2:
            st.info("Need at least two suppliers to compare.")
        else:
            a1, a2 = st.columns(2)
            na = a1.selectbox("Supplier A", order_sup, index=0, key="cmp_a")
            nb = a2.selectbox("Supplier B", order_sup, index=1, key="cmp_b")
            A = next(x for x in sups if x["supplier_name"] == na)
            B = next(x for x in sups if x["supplier_name"] == nb)
            for col, X in ((a1, A), (a2, B)):
                col.metric(f"#{X['final_rank']} {X['supplier_name']}", f"PPI {fmt(X['ppi'])}",
                           f"absolute {fmt(X['absolute_score'])} · industry {fmt(X['industry_index'], 1)}", delta_color="off")
            rows = []
            for ca, cb in zip(A["criteria"], B["criteria"]):
                better = na if ca["score"] > cb["score"] else nb if cb["score"] > ca["score"] else "Tie"
                rows.append({"Criterion": ca["name"], "Weight %": ca["weight"], na: ca["score"], nb: cb["score"],
                             "Stronger": better})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            long = pd.DataFrame([{"Criterion": r["Criterion"], "Supplier": n, "Score": r[n]} for r in rows for n in (na, nb)])
            chart = alt.Chart(long).mark_bar(cornerRadiusEnd=3, size=12).encode(
                y=alt.Y("Criterion:N", sort=order_crit, title=None), x=alt.X("Score:Q", title="Score"),
                yOffset="Supplier:N",
                color=alt.Color("Supplier:N", scale=alt.Scale(domain=[na, nb], range=[BLUE, ORANGE]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Supplier", "Criterion", "Score"]).properties(height=60 * len(rows) + 30)
            st.altair_chart(chart, use_container_width=True)
            crit_pick = st.selectbox("Read the evidence side by side for", order_crit, key="cmp_crit")
            ea = next(x for x in A["criteria"] if x["name"] == crit_pick)
            eb = next(x for x in B["criteria"] if x["name"] == crit_pick)
            l, r = st.columns(2)
            for col, X, e in ((l, A, ea), (r, B, eb)):
                with col:
                    st.markdown(f"**{X['supplier_name']} · {e['score']:g}/{e['max_score']}**")
                    st.markdown(f"> {e.get('evidence') or 'No evidence quoted'}")
                    st.caption(e.get("justification", ""))

    # ------------------------------------------------------------ overrides
    with tabs[5]:
        st.markdown("A procurement analyst can change any AI score after review. A reason is required, every change "
                    "is logged, and the ranking is recalculated by the same deterministic rules. No AI call is made.")
        o1, o2 = st.columns(2)
        sup = o1.selectbox("Supplier", order_sup, key="ov_sup")
        S = next(x for x in sups if x["supplier_name"] == sup)
        crit_name = o2.selectbox("Criterion", order_crit, key="ov_crit")
        C = next(x for x in S["criteria"] if x["name"] == crit_name)
        st.caption(f"Current score {C['score']:g}/{C['max_score']}" +
                   (f" (AI originally {C.get('original_score'):g})" if C.get("overridden") else "") +
                   f" · evidence: {C.get('evidence') or 'none'}")
        new = st.slider("New score", 0, int(C["max_score"]), int(C["score"]), key=f"ov_new_{sup}_{crit_name}")
        reason = st.text_input("Reason (required)", key="ov_reason",
                               placeholder="e.g. Pricing confirmed in clarification call on 12 Oct")
        analyst = st.text_input("Analyst name", key="ov_who")
        if st.button("Apply override and re-rank", type="primary", disabled=not reason.strip() or new == C["score"]):
            ai_score = C.get("original_score", C["score"])
            db.add_override(run["rfp_run_id"], sup, C["criterion_id"], C["name"], ai_score, new, reason.strip(), analyst.strip())
            before = {x["supplier_name"]: x["final_rank"] for x in sups}
            new_run = rescore_run(run["rfp_run_id"])
            moved = [f"{x['supplier_name']} {before[x['supplier_name']]}→{x['final_rank']}" for x in new_run["suppliers"]
                     if before[x["supplier_name"]] != x["final_rank"]]
            st.success("Override saved; ranking recalculated." + (" Rank changes: " + ", ".join(moved) if moved else
                                                                     " No rank changes."))
            st.rerun()
        ovs = db.list_overrides(run["rfp_run_id"])
        if ovs:
            st.markdown("**Override log**")
            st.dataframe(pd.DataFrame([{"When": o["created_at"], "Supplier": o["supplier_name"], "Criterion": o["criterion_name"],
                                        "AI score": o["old_score"], "New score": o["new_score"], "Reason": o["reason"],
                                        "Analyst": o["analyst"]} for o in ovs]), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------ details & export
    with tabs[6]:
        tm = run.get("template", {})
        st.markdown(f"**RFP_RUN_ID:** `{run['rfp_run_id']}` · **created (UTC):** {run['created_at']} · "
                    f"**revision:** {run.get('revision', 1)}  \n**Template:** {run.get('rfp_title')} · **client:** "
                    f"{tm.get('buyer') or '-'}  \n**Engine:** {run['evaluation_mode']} · **prompt versions:** "
                    f"{run.get('prompt_versions')}")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ PDF report", build_run_report(run), f"{run['rfp_run_id']}_report.pdf", "application/pdf",
                           type="primary", use_container_width=True)
        export = {k: v for k, v in run.items()}
        d2.download_button("⬇ Complete result (JSON)", json.dumps(export, indent=2), f"{run['rfp_run_id']}.json",
                           "application/json", use_container_width=True)
        d3.download_button("⬇ Leaderboard (CSV)", leaderboard_df(run).to_csv(index=False), f"{run['rfp_run_id']}_leaderboard.csv",
                           "text/csv", use_container_width=True)
        st.markdown("#### Tie-break explanation")
        st.dataframe(pd.DataFrame([{"Position": f"#{t['upper_rank']} {t['upper']} above #{t['lower_rank']} {t['lower']}",
                                    "Deciding rule": t["deciding_rule"], "Tie-break used": "Yes" if t["is_tie_break"] else "No",
                                    "Explanation": t["explanation"]} for t in run["tie_breaks"]]),
                     hide_index=True, use_container_width=True)
        st.markdown(f"#### Warnings ({len(run['warnings'])})")
        if run["warnings"]:
            st.markdown("\n".join(f"- {w}" for w in run["warnings"]))
        else:
            st.caption("No warnings: every AI result was complete, in range and backed by evidence found in the PDF.")
        if run["not_evaluated"]:
            st.markdown("#### Files not evaluated")
            st.dataframe(pd.DataFrame(run["not_evaluated"]), hide_index=True, use_container_width=True)
        with st.expander("Formulas"):
            st.table(pd.DataFrame(list(run.get("formulas", FORMULAS).items()), columns=["Metric", "Formula"]))
        with st.expander("Criteria used in this run"):
            st.dataframe(pd.DataFrame(run["criteria"])[["name", "weight", "max_score", "description", "industry_benchmark"]],
                         hide_index=True, use_container_width=True)
        with st.expander("Agent execution trace"):
            st.dataframe(pd.DataFrame(run["execution_trace"]), hide_index=True, use_container_width=True)
    _ = ss
