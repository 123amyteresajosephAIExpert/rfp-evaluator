"""Evaluate: upload supplier proposals, enter metadata and run the agentic workflow."""

import pandas as pd
import streamlit as st

from rfp import config, db
from rfp.orchestrator import run_evaluation
from rfp.validation_tool import validate_supplier_inputs
from ui.common import (PAGES, SAMPLE_SETS, SAMPLES, ai_ready, default_meta, get_evaluator, need_ai_message,
                       page_header, pill, template_picker)


def render():
    ss = st.session_state
    ss.setdefault("extra_files", {})
    page_header("📤 Evaluate proposals",
                "Upload as many supplier proposals as you like, check their details, and run the evaluation.")
    t = template_picker("Evaluate against template")
    if not t:
        return
    crit = db.get_criteria(t["template_id"])
    st.markdown("Criteria: " + " ".join(pill(f"{c['name']} · {c['weight']:g}%") for c in crit), unsafe_allow_html=True)

    st.markdown("#### 1 · Proposals")
    uploads = st.file_uploader("Supplier proposals (PDF, any number)", type=["pdf"], accept_multiple_files=True)
    c1, c2, c3 = st.columns([2.2, 1, 1])
    sample_label = c1.selectbox("Sample proposals", list(SAMPLE_SETS), label_visibility="collapsed")
    if c2.button("Load samples", use_container_width=True):
        folder = SAMPLE_SETS[sample_label][0]
        for p in sorted((SAMPLES / "proposals" / folder).glob("*.pdf")):
            ss.extra_files[p.name] = p.read_bytes()
    if c3.button("Clear loaded", use_container_width=True):
        ss.extra_files = {}

    with st.expander("🧪 Testing tools (demonstrate validation and tie-breaks)"):
        st.caption("Use these during a demo to show how the app handles bad inputs and ties.")
        faults = st.text_input("Corrupt the AI's answer for these suppliers (comma-separated)",
                               help="Simulates out-of-range, non-numeric, missing and invented-evidence results.")
        e1, e2 = st.columns(2)
        if e1.button("Add bad files (scanned + corrupt PDF)", use_container_width=True):
            for p in sorted((SAMPLES / "edge_cases").glob("*.pdf")):
                ss.extra_files[p.name] = p.read_bytes()
        if e2.button("Add tie-break test file", use_container_width=True,
                     help="An identical copy of the NexaWorks proposal: it ties on PPI and the tie-break rules decide."):
            ss.extra_files["NexaWorks_Twin_Copy.pdf"] = (SAMPLES / "proposals" / "hr_platform" / "NexaWorks_RFP_Response.pdf").read_bytes()

    files = dict(ss.extra_files)
    for u in uploads or []:
        files[u.name] = u.getvalue()
    if not files:
        st.info("Upload proposals or load a sample set to continue.")
        return

    st.markdown("#### 2 · Supplier details")
    st.caption(f"Experience rating: {config.EXPERIENCE_RATING_MIN} (low) to {config.EXPERIENCE_RATING_MAX} (high), "
               "from your supplier history. Used only as a tie-breaker.")
    rows0 = []
    for n in files:
        m = default_meta(n)
        if n == "NexaWorks_Twin_Copy.pdf":
            m = {**default_meta("NexaWorks_RFP_Response.pdf"), "supplier_name": "NexaWorks Twin", "experience_rating": 3.0}
        rows0.append({"filename": n, **m, "size_kb": round(len(files[n]) / 1024, 1)})
    meta = st.data_editor(
        pd.DataFrame(rows0), key="meta_" + "|".join(sorted(files)), hide_index=True, use_container_width=True,
        column_config={
            "filename": st.column_config.TextColumn("File", disabled=True),
            "supplier_name": st.column_config.TextColumn("Supplier name", required=True),
            "submission_date": st.column_config.DateColumn("Submission date", format="YYYY-MM-DD"),
            "experience_rating": st.column_config.NumberColumn("Experience rating", min_value=config.EXPERIENCE_RATING_MIN,
                                                               max_value=config.EXPERIENCE_RATING_MAX, step=0.5),
            "size_kb": st.column_config.NumberColumn("Size (KB)", disabled=True),
        })
    rows = meta.to_dict("records")
    for r in rows:
        if isinstance(r["submission_date"], pd.Timestamp):
            r["submission_date"] = r["submission_date"].date()
        if pd.isna(r.get("experience_rating")):
            r["experience_rating"] = None
    errors = validate_supplier_inputs(rows) + db.validate_criteria_set(crit)
    for e in errors:
        st.error(e)

    st.markdown("#### 3 · Run")
    use_dna = st.toggle("Reuse previous evaluations when the document, criteria, prompt and model are identical "
                        "(evaluation DNA)", value=True)
    if not ai_ready():
        need_ai_message()
    can_run = not errors and ai_ready()
    if st.button(f"🚀 Evaluate {len(rows)} proposal(s)", type="primary", disabled=not can_run):
        fault_names = {n.strip() for n in (faults or "").split(",") if n.strip()}
        evaluator = get_evaluator(fault_names)
        inputs = [{"supplier_name": r["supplier_name"].strip(), "submission_date": r["submission_date"].isoformat(),
                   "experience_rating": float(r["experience_rating"]), "filename": r["filename"],
                   "file_bytes": files[r["filename"]]} for r in rows]
        with st.status("Running the agentic evaluation workflow...", expanded=True) as status:
            bar = st.progress(0.0)

            def progress(msg, frac):
                bar.progress(min(frac, 1.0))
                status.write(msg)
            try:
                state = run_evaluation(t["template_id"], inputs, evaluator, use_dna=use_dna, progress=progress)
            except Exception as exc:  # noqa: BLE001
                status.update(label="Evaluation failed", state="error")
                st.exception(exc)
                return
            if state.get("status") == "COMPLETED":
                ss.run_id = state["result"]["rfp_run_id"]
                status.update(label=f"Completed {ss.run_id}", state="complete", expanded=False)
            else:
                status.update(label="Stopped - no readable documents", state="error")
                for w in state.get("warnings", []):
                    st.error(w)
                return
        r = state["result"]
        reused = sum(s["dna_reused"] for s in r["suppliers"])
        st.success(f"Done. Winner: **{r['suppliers'][0]['supplier_name']}**. {len(r['suppliers'])} supplier(s) ranked"
                   + (f", {reused} reused from evaluation DNA" if reused else "") + ".")
        if r["warnings"]:
            st.warning(f"{len(r['warnings'])} validation warning(s) - see Results › Run details.")
        if "results" in PAGES:
            st.page_link(PAGES["results"], label="Open results", icon="🏆")
        with st.expander("Agent execution trace"):
            st.dataframe(pd.DataFrame(r["execution_trace"]), hide_index=True, use_container_width=True)
