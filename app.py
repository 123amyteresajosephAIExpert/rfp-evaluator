"""Streamlit UI for the Agentic RFP Evaluation and Supplier Ranking app.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from rfp import config, db
from rfp.evaluation_agent import FaultInjectingEvaluator, LLMEvaluator, OfflineDemoEvaluator
from rfp.orchestrator import FORMULAS, run_evaluation
from rfp.validation_tool import validate_supplier_inputs

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data" / "supplier_pdfs"
EDGE_DIR = ROOT / "sample_data" / "edge_cases"
SAMPLE_META = json.loads((ROOT / "sample_data" / "sample_metadata.json").read_text())
BAR_COLOR = "#2a6fdb"

st.set_page_config(page_title="RFP Evaluator", page_icon="📑", layout="wide")


@st.cache_resource
def _init_database() -> bool:
    db.init_db()
    return True


_init_database()
ss = st.session_state
ss.setdefault("extra_files", {})      # filename -> bytes (sample/edge files loaded from disk)
ss.setdefault("result", None)
ss.setdefault("last_state", None)


# ------------------------------------------------------------------------------------
# Sidebar: evaluation engine
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Evaluation engine")
    mode = st.radio(
        "Scoring mode",
        ["LLM (recommended)", "Offline demo (no API key)"],
        help="LLM mode reads each proposal with a large language model. Offline demo mode uses a transparent "
             "keyword heuristic so the workflow can be shown without an API key. It is not an AI judgement.",
    )
    evaluator = None
    engine_error = ""
    if mode.startswith("LLM"):
        provider = st.selectbox("Provider", list(config.PROVIDERS), index=list(config.PROVIDERS).index(config.DEFAULT_PROVIDER))
        p = config.PROVIDERS[provider]
        model = st.text_input("Model", value=config.get_setting("LLM_MODEL", p["default_model"]) if provider == config.DEFAULT_PROVIDER else p["default_model"])
        secret_key = config.get_setting(p["key_name"])
        if secret_key:
            st.success(f"{p['key_name']} loaded from app secrets.")
            api_key = secret_key
        else:
            api_key = st.text_input(f"{p['key_name']}", type="password",
                                    help="Used only for this session; never stored.")
        use_cache = st.toggle("Reuse cached scorecards for identical documents", value=True,
                              help="Same document + same criteria + same model returns the stored scorecard. "
                                   "Guarantees repeatable results and saves cost.")
        if api_key:
            try:
                evaluator = LLMEvaluator(provider, api_key, model.strip() or None, use_cache=use_cache)
            except Exception as exc:  # pragma: no cover
                engine_error = f"Could not create the LLM client: {exc}"
        else:
            engine_error = f"Add an API key ({p['key_name']}) or switch to offline demo mode."
    else:
        evaluator = OfflineDemoEvaluator()
        st.warning("Offline demo mode: scores come from keyword counts, not from an LLM. Use for demos and tests only.")

    with st.expander("Testing tools"):
        st.caption("Demonstrate the validation layer by deliberately corrupting the evaluator's reply for "
                   "selected suppliers (out-of-range, non-numeric, missing and invented-evidence results).")
        fault_names = st.text_input("Simulate malformed LLM output for (comma-separated supplier names)", "")
        if st.button("Add tie-break test file"):
            src = SAMPLE_DIR / "NexaWorks_RFP_Response.pdf"
            ss.extra_files["NexaWorks_Twin_Copy.pdf"] = src.read_bytes()
            st.toast("Added an identical copy of the NexaWorks proposal as 'NexaWorks Twin' (same date, "
                     "lower experience rating). Its PPI will tie with NexaWorks; rule 3 decides the order.")
        if st.button("Load bad files (scanned + corrupt PDF)"):
            for f in sorted(EDGE_DIR.glob("*.pdf")):
                ss.extra_files[f.name] = f.read_bytes()
            st.toast("Edge-case files added to the upload list.")

    faults = {n.strip() for n in fault_names.split(",") if n.strip()}
    if evaluator is not None and faults:
        evaluator = FaultInjectingEvaluator(evaluator, faults)

    st.divider()
    st.caption("The LLM only judges proposal content. All arithmetic, benchmarks, tie-breaks and ranks are "
               "computed by deterministic Python, so the same validated scorecards always give the same ranking.")


# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def default_meta(filename: str) -> dict:
    if filename == "NexaWorks_Twin_Copy.pdf":
        return {"supplier_name": "NexaWorks Twin", "submission_date": date(2026, 10, 7), "experience_rating": 3.0}
    if filename in SAMPLE_META:
        m = SAMPLE_META[filename]
        return {"supplier_name": m["supplier_name"], "submission_date": date.fromisoformat(m["submission_date"]),
                "experience_rating": float(m["experience_rating"])}
    stem = re.sub(r"(?i)[_\- ]*(rfp)?[_\- ]*(response|proposal)?$", "", Path(filename).stem)
    return {"supplier_name": stem.replace("_", " ").strip() or Path(filename).stem,
            "submission_date": date.today(), "experience_rating": 3.0}


def fmt(x, d=2):
    return f"{x:,.{d}f}"


def result_or_info():
    if ss.result is None:
        st.info("No evaluation loaded yet. Run one in the **Evaluate** tab or open a past run in **History**.")
        return None
    return ss.result


st.title("📑 Agentic RFP Evaluation and Supplier Ranking")
st.caption("Reads supplier proposals, scores them against configurable criteria with evidence, benchmarks "
           "suppliers against their peers and produces an explainable leaderboard.")
if ss.result:
    r = ss.result
    st.markdown(f"**Current run:** `{r['rfp_run_id']}` · {r.get('rfp_title') or 'Untitled RFP'} · "
                f"{len(r['suppliers'])} suppliers · {r['evaluation_mode']}")

tab_criteria, tab_eval, tab_board, tab_cards, tab_run, tab_hist = st.tabs(
    ["1 · Criteria", "2 · Evaluate", "3 · Leaderboard", "4 · Scorecards", "5 · Run details", "History"])


# ------------------------------------------------------------------------------------
# 1. Criteria
# ------------------------------------------------------------------------------------
with tab_criteria:
    st.subheader("Evaluation criteria")
    st.caption("Stored in SQLite. Change weights, switch criteria on or off, or add new ones. The prompt is built "
               "from this table at run time, so no code changes are needed. Active weights must total 100%.")
    active_now = db.get_criteria(active_only=True)
    st.markdown("**Active criteria used by the next evaluation**")
    st.dataframe(pd.DataFrame([{"Criterion": c["name"], "Weight %": c["weight"], "Max score": c["max_score"],
                                "What the LLM inspects": c["description"]} for c in active_now]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Edit criteria**")
    all_criteria = db.get_criteria(active_only=False)
    df = pd.DataFrame(all_criteria)
    df["is_active"] = df["is_active"].astype(bool)
    edited = st.data_editor(
        df,
        key="criteria_editor",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_order=["criterion_id", "is_active", "name", "weight", "max_score", "description"],
        column_config={
            "criterion_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "is_active": st.column_config.CheckboxColumn("Active", width="small"),
            "name": st.column_config.TextColumn("Criterion", required=True),
            "weight": st.column_config.NumberColumn("Weight %", min_value=0, max_value=100, step=1, format="%g"),
            "max_score": st.column_config.NumberColumn("Max score", min_value=1, max_value=100, step=1),
            "description": st.column_config.TextColumn("What the LLM should inspect", width="large", required=True),
        },
    )
    active_total = float(edited.loc[edited["is_active"] == True, "weight"].fillna(0).sum())  # noqa: E712
    c1, c2 = st.columns([1, 3])
    c1.metric("Active weight total", f"{active_total:g}%", delta=None if active_total == 100 else f"{active_total - 100:+g}%",
              delta_color="inverse")
    if c2.button("Save criteria", type="primary"):
        rows = []
        kept_ids = set()
        for rec in edited.to_dict("records"):
            if not str(rec.get("name") or "").strip():
                continue
            cid = rec.get("criterion_id")
            cid = int(cid) if pd.notna(cid) else None
            kept_ids.add(cid)
            rows.append({"criterion_id": cid, "name": str(rec["name"]), "description": str(rec.get("description") or ""),
                         "weight": float(rec.get("weight") or 0), "max_score": int(rec.get("max_score") or 10),
                         "is_active": 1 if rec.get("is_active") else 0})
        for c in all_criteria:  # deleted rows are deactivated, never destroyed (audit trail)
            if c["criterion_id"] not in kept_ids:
                rows.append({**c, "is_active": 0})
        try:
            db.save_criteria(rows)
            st.success("Criteria saved. The next evaluation will use them.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if active_total != 100:
        st.warning(f"Active weights total {active_total:g}%. Evaluation is blocked until they total 100%.")


# ------------------------------------------------------------------------------------
# 2. Evaluate
# ------------------------------------------------------------------------------------
with tab_eval:
    st.subheader("Supplier proposals")
    rfp_title = st.text_input("RFP title", value="KML/HR/2026/014 - HR Service Delivery Platform")
    uploads = st.file_uploader("Upload supplier RFP responses (PDF)", type=["pdf"], accept_multiple_files=True)
    b1, b2, _ = st.columns([1.3, 1, 3])
    if b1.button("Load the 4 sample supplier PDFs"):
        for f in sorted(SAMPLE_DIR.glob("*.pdf")):
            ss.extra_files[f.name] = f.read_bytes()
    if b2.button("Clear loaded files"):
        ss.extra_files = {}

    files: dict[str, bytes] = dict(ss.extra_files)
    for u in uploads or []:
        files[u.name] = u.getvalue()

    if not files:
        st.info("Upload at least one PDF, or load the sample proposals to try the app.")
    else:
        st.markdown("**Supplier details** - edit any cell. Historical experience rating is 1 (low) to 5 (high).")
        meta_df = pd.DataFrame([{"filename": n, **default_meta(n)} for n in files])
        meta = st.data_editor(
            meta_df,
            key="meta_" + "|".join(sorted(files)),
            hide_index=True,
            use_container_width=True,
            column_config={
                "filename": st.column_config.TextColumn("File", disabled=True),
                "supplier_name": st.column_config.TextColumn("Supplier name", required=True),
                "submission_date": st.column_config.DateColumn("Submission date", format="YYYY-MM-DD"),
                "experience_rating": st.column_config.NumberColumn(
                    "Experience rating", min_value=config.EXPERIENCE_RATING_MIN,
                    max_value=config.EXPERIENCE_RATING_MAX, step=0.5),
            },
        )
        rows = meta.to_dict("records")
        for r_ in rows:
            if isinstance(r_["submission_date"], pd.Timestamp):
                r_["submission_date"] = r_["submission_date"].date()
            if pd.isna(r_.get("experience_rating")):
                r_["experience_rating"] = None
        errors = validate_supplier_inputs(rows)
        crit_problems = db.validate_criteria_set(db.get_criteria(active_only=True))
        for e in errors + crit_problems:
            st.error(e)
        if engine_error:
            st.warning(engine_error)

        can_run = not errors and not crit_problems and evaluator is not None
        if st.button(f"Evaluate {len(rows)} supplier(s)", type="primary", disabled=not can_run):
            supplier_inputs = [{"supplier_name": r_["supplier_name"].strip(),
                                "submission_date": r_["submission_date"].isoformat(),
                                "experience_rating": float(r_["experience_rating"]),
                                "filename": r_["filename"], "file_bytes": files[r_["filename"]]} for r_ in rows]
            with st.status("Running the agentic evaluation workflow...", expanded=True) as status:
                bar = st.progress(0.0)
                def progress(msg, frac):
                    bar.progress(min(frac, 1.0))
                    status.write(msg)
                try:
                    state = run_evaluation(supplier_inputs, evaluator, rfp_title, progress=progress)
                except Exception as exc:
                    status.update(label="Evaluation failed", state="error")
                    st.exception(exc)
                    state = None
                if state:
                    ss.last_state = {"status": state.get("status"), "warnings": state.get("warnings", []),
                                     "trace": state.get("trace", [])}
                    if state.get("status") == "COMPLETED":
                        ss.result = state["result"]
                        status.update(label=f"Completed run {ss.result['rfp_run_id']}", state="complete", expanded=False)
                    else:
                        status.update(label="Run stopped - no readable documents", state="error")
            if ss.last_state and ss.last_state["status"] == "COMPLETED":
                st.success("Done. Open the **Leaderboard** tab to see the ranking.")
                if ss.result["warnings"]:
                    st.warning(f"{len(ss.result['warnings'])} validation warning(s) - see **Run details**.")
            elif ss.last_state:
                for w in ss.last_state["warnings"]:
                    st.error(w)


# ------------------------------------------------------------------------------------
# 3. Leaderboard
# ------------------------------------------------------------------------------------
with tab_board:
    r = result_or_info()
    if r:
        sups = r["suppliers"]
        top = sups[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended supplier", top["supplier_name"])
        m2.metric("Top PPI", fmt(top["ppi"]))
        m3.metric("Suppliers ranked", len(sups), help=f"{len(r['not_evaluated'])} file(s) could not be evaluated")
        m4.metric("Validation warnings", len(r["warnings"]))

        board = pd.DataFrame([{
            "Rank": s["final_rank"], "Supplier": s["supplier_name"],
            "Absolute score (/100)": s["absolute_score"], "PPI": s["ppi"],
            "Submission date": s["submission_date"], "Experience rating": s["experience_rating"],
            "Why this position": s["rank_explanation"],
        } for s in sups])
        st.dataframe(board, hide_index=True, use_container_width=True,
                     column_config={"Absolute score (/100)": st.column_config.ProgressColumn(
                         "Absolute score (/100)", min_value=0, max_value=100, format="%.2f"),
                                    "PPI": st.column_config.ProgressColumn("PPI", min_value=0, max_value=100, format="%.2f")})
        st.caption("Ranking order: 1) higher PPI, 2) earlier submission date, 3) higher experience rating, "
                   "4) supplier name A-Z. PPI = weighted average of each criterion score as a % of the best peer score.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Peer Performance Index by supplier**")
            chart = alt.Chart(board).mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, size=26).encode(
                x=alt.X("PPI:Q", scale=alt.Scale(domain=[0, 100]), title="PPI (0-100)"),
                y=alt.Y("Supplier:N", sort=alt.SortField("Rank"), title=None),
                tooltip=["Rank", "Supplier", alt.Tooltip("PPI:Q", format=".2f"),
                         alt.Tooltip("Absolute score (/100):Q", format=".2f")],
            ).properties(height=48 * len(board) + 20)
            st.altair_chart(chart, use_container_width=True)
        with right:
            st.markdown("**Relative performance by criterion** (% of best peer score)")
            heat = pd.DataFrame([{"Supplier": s["supplier_name"], "Rank": s["final_rank"], "Criterion": c["name"],
                                  "Relative %": c["relative_pct"], "Score": c["score"], "Benchmark": c["benchmark_score"]}
                                 for s in sups for c in s["criteria"]])
            base = alt.Chart(heat).encode(
                x=alt.X("Criterion:N", title=None, sort=[c["name"] for c in sups[0]["criteria"]],
                       axis=alt.Axis(labelAngle=-30, labelLimit=200, labelOverlap=False)),
                y=alt.Y("Supplier:N", sort=alt.SortField("Rank"), title=None))
            cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
                color=alt.Color("Relative %:Q", scale=alt.Scale(scheme="blues", domain=[0, 100]),
                                legend=alt.Legend(title="Relative %")),
                tooltip=["Supplier", "Criterion", "Score", "Benchmark", alt.Tooltip("Relative %:Q", format=".1f")])
            labels = base.mark_text(fontSize=11).encode(
                text=alt.Text("Relative %:Q", format=".0f"),
                color=alt.condition("datum['Relative %'] > 60", alt.value("white"), alt.value("#1d2733")))
            st.altair_chart((cells + labels).properties(height=48 * len(board) + 40), use_container_width=True)

        if r["not_evaluated"]:
            st.warning("Not evaluated: " + "; ".join(f"{n['supplier_name']} ({n['filename']})" for n in r["not_evaluated"]))


# ------------------------------------------------------------------------------------
# 4. Detailed scorecards
# ------------------------------------------------------------------------------------
with tab_cards:
    r = result_or_info()
    if r:
        names = [f"#{s['final_rank']} {s['supplier_name']}" for s in r["suppliers"]]
        pick = st.selectbox("Supplier", names)
        s = r["suppliers"][names.index(pick)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rank", s["final_rank"])
        c2.metric("Absolute score", fmt(s["absolute_score"]))
        c3.metric("PPI", fmt(s["ppi"]))
        c4.metric("Evidence verified", f"{sum(c['evidence_verified'] for c in s['criteria'])}/{len(s['criteria'])}")
        if s.get("overall_summary"):
            st.markdown(f"**Summary:** {s['overall_summary']}")
        st.caption(s["rank_explanation"])

        card = pd.DataFrame([{
            "Criterion": c["name"], "Weight %": c["weight"], "Score": c["score"], "Max": c["max_score"],
            "Benchmark": c["benchmark_score"], "Gap": c["gap"], "Relative %": c["relative_pct"],
            "Weighted points": c["weighted_points"], "Leader": "★" if c["is_benchmark_leader"] else "",
            "Evidence verified": "✔" if c["evidence_verified"] else "⚠", "Status": c["status"],
        } for c in s["criteria"]])
        st.dataframe(card, hide_index=True, use_container_width=True, column_config={
            c: st.column_config.NumberColumn(c, format="%.2f") for c in ["Gap", "Relative %", "Weighted points"]})

        st.markdown("**Justification and evidence by criterion**")
        for c in s["criteria"]:
            flag = "✔ verified in document" if c["evidence_verified"] else "⚠ not verified - review manually"
            with st.expander(f"{c['name']} - {c['score']:g}/{c['max_score']} ({flag})"):
                st.markdown(f"**Justification:** {c['justification']}")
                st.markdown(f"**Evidence quoted from the proposal:** {('> ' + c['evidence']) if c['evidence'] else '_none provided_'}")
                st.caption(f"Benchmark {c['benchmark_score']:g} · gap {c['gap']:g} · relative {c['relative_pct']:.1f}% · "
                           f"contributes {c['weighted_points']:.2f} of {c['weight']:g} weighted points")
        if s.get("risks"):
            st.markdown("**Risks identified**")
            for risk in s["risks"]:
                st.markdown(f"- {risk}")
        if s.get("validation_warnings"):
            with st.expander(f"Validation warnings for this supplier ({len(s['validation_warnings'])})"):
                for w in s["validation_warnings"]:
                    st.markdown(f"- {w}")

        st.markdown("**Side-by-side score comparison**")
        compare = pd.DataFrame({x["supplier_name"]: {c["name"]: c["score"] for c in x["criteria"]} for x in r["suppliers"]})
        compare.loc["Absolute score"] = [x["absolute_score"] for x in r["suppliers"]]
        compare.loc["PPI"] = [x["ppi"] for x in r["suppliers"]]
        st.dataframe(compare, use_container_width=True)


# ------------------------------------------------------------------------------------
# 5. Run details
# ------------------------------------------------------------------------------------
with tab_run:
    r = result_or_info()
    if r:
        st.markdown(f"**RFP_RUN_ID:** `{r['rfp_run_id']}`  \n**Created (UTC):** {r['created_at']}  \n"
                    f"**Evaluation engine:** {r['evaluation_mode']}  \n**RFP:** {r.get('rfp_title') or '-'}")
        st.download_button("⬇ Download complete result (JSON)", data=json.dumps(r, indent=2),
                           file_name=f"{r['rfp_run_id']}.json", mime="application/json", type="primary")

        st.markdown("#### Tie-break explanation")
        tb = pd.DataFrame([{"Position": f"#{t['upper_rank']} {t['upper']}  above  #{t['lower_rank']} {t['lower']}",
                            "Deciding rule": t["deciding_rule"], "Tie-break used": "Yes" if t["is_tie_break"] else "No",
                            "Explanation": t["explanation"]} for t in r["tie_breaks"]])
        if len(tb):
            st.dataframe(tb, hide_index=True, use_container_width=True)
        else:
            st.caption("Only one supplier was ranked.")

        st.markdown(f"#### Warnings ({len(r['warnings'])})")
        if r["warnings"]:
            for w in r["warnings"]:
                st.markdown(f"- {w}")
        else:
            st.caption("No warnings - every LLM result was complete, in range and backed by verified evidence.")
        if r["not_evaluated"]:
            st.markdown("#### Files not evaluated")
            st.dataframe(pd.DataFrame(r["not_evaluated"]), hide_index=True, use_container_width=True)

        st.markdown("#### Benchmarks")
        st.dataframe(pd.DataFrame([{"Criterion": b["name"], "Benchmark score": b["benchmark_score"],
                                    "Max": b["max_score"], "Leader(s)": ", ".join(b["leaders"])} for b in r["benchmarks"]]),
                     hide_index=True, use_container_width=True)
        with st.expander("Formulas and criteria used in this run"):
            st.table(pd.DataFrame(list(r.get("formulas", FORMULAS).items()), columns=["Metric", "Formula"]))
            st.dataframe(pd.DataFrame(r["criteria"])[["criterion_id", "name", "weight", "max_score", "description"]],
                         hide_index=True, use_container_width=True)
        with st.expander("Agent execution trace"):
            st.dataframe(pd.DataFrame(r["execution_trace"]), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------------------
# History
# ------------------------------------------------------------------------------------
with tab_hist:
    runs = db.list_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)
        done = [x["rfp_run_id"] for x in runs if x["status"] == "COMPLETED"]
        if done:
            pick = st.selectbox("Open a completed run", done)
            if st.button("Load run"):
                ss.result = db.load_run(pick)
                st.rerun()
