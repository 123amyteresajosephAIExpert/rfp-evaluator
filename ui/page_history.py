"""History & DNA: every past run, and the library of stored evaluation fingerprints."""

import pandas as pd
import streamlit as st

from rfp import db
from ui.common import PAGES, page_header


def render():
    ss = st.session_state
    page_header("🗂️ History & evaluation DNA",
                "Every run is stored with its inputs, criteria, prompt versions, AI scorecards and results, so an "
                "analyst can reopen and analyse it later.")
    runs = db.list_runs()
    dna = db.list_dna()
    m = st.columns(4)
    m[0].metric("Runs", len(runs))
    m[1].metric("Completed", sum(r["status"] == "COMPLETED" for r in runs))
    m[2].metric("Stored evaluations (DNA)", len(dna))
    m[3].metric("AI calls saved by DNA", sum(d["reuse_count"] for d in dna))

    tab_runs, tab_dna = st.tabs(["Past runs", "Evaluation DNA library"])
    with tab_runs:
        if not runs:
            st.info("No runs yet.")
        else:
            df = pd.DataFrame([{"Run ID": r["rfp_run_id"], "Created (UTC)": r["created_at"], "Status": r["status"],
                                "Template": r["template_name"], "Winner": r["winner"], "Suppliers": r["suppliers"],
                                "Revision": r["revision"], "Engine": r["evaluation_mode"]} for r in runs])
            q = st.text_input("Filter", placeholder="Search by run ID, template, winner or engine")
            if q:
                mask = df.apply(lambda row: q.lower() in " ".join(map(str, row.values)).lower(), axis=1)
                df = df[mask]
            st.dataframe(df, hide_index=True, use_container_width=True)
            done = [r["rfp_run_id"] for r in runs if r["status"] == "COMPLETED"]
            if done:
                c1, c2 = st.columns([3, 1])
                pick = c1.selectbox("Open a run", done)
                if c2.button("Open in Results", type="primary", use_container_width=True):
                    ss.run_id = pick
                    st.switch_page(PAGES["results"])
    with tab_dna:
        st.markdown("A **fingerprint** combines the proposal file's SHA-256 hash, the criteria, the prompt text and "
                    "the AI model. If all four are identical, the stored scorecard is reused instead of asking the AI "
                    "again, so the same proposal always gets the same evaluation and no AI call is spent. Change any "
                    "of the four and a fresh evaluation is made.")
        if not dna:
            st.info("No stored evaluations yet.")
        else:
            st.dataframe(pd.DataFrame([{"Fingerprint": d["fingerprint"][:16], "Document SHA-256": d["document_sha"][:16],
                                        "First supplier": d["first_supplier"], "First run": d["first_run_id"],
                                        "Model": d["model"], "Reused": d["reuse_count"], "Last used": d["last_used_at"]}
                                       for d in dna]), hide_index=True, use_container_width=True)
            with st.expander("Clear the DNA library"):
                st.caption("Forces fresh AI evaluations next time. Past runs are not affected.")
                if st.button("Clear all stored evaluations"):
                    db.clear_dna()
                    st.rerun()
