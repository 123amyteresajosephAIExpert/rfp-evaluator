"""Criteria & prompt: edit the active template's criteria and the versioned AI prompts."""

import altair as alt
import pandas as pd
import streamlit as st

from rfp import db, prompts
from rfp.evaluation_agent import build_messages
from ui.common import BLUE, page_header, template_picker


def render():
    page_header("⚖️ Criteria & prompt",
                "Everything the AI is asked comes from here, stored in SQLite. Change it without touching code. "
                "Any change creates a new evaluation fingerprint, so previous DNA is not reused for it.")
    t = template_picker("Template")
    if not t:
        return
    tab_c, tab_p = st.tabs(["Criteria & weights", "AI prompt"])

    # ---------------------------------------------------------------- criteria
    with tab_c:
        all_c = db.get_criteria(t["template_id"], active_only=False)
        df = pd.DataFrame(all_c)
        df["is_active"] = df["is_active"].astype(bool)
        edited = st.data_editor(
            df, key=f"crit_editor_{t['template_id']}", num_rows="dynamic", hide_index=True, use_container_width=True,
            column_order=["is_active", "name", "weight", "max_score", "description", "industry_benchmark"],
            column_config={
                "is_active": st.column_config.CheckboxColumn("Active", width="small"),
                "name": st.column_config.TextColumn("Criterion", required=True),
                "weight": st.column_config.NumberColumn("Weight %", min_value=0, max_value=100, step=1, format="%g"),
                "max_score": st.column_config.NumberColumn("Max", min_value=1, max_value=100, step=1),
                "description": st.column_config.TextColumn("What the AI should inspect", width="large"),
                "industry_benchmark": st.column_config.TextColumn("Industry best practice", width="large"),
            })
        active = edited[edited["is_active"] == True]  # noqa: E712
        total = float(active["weight"].fillna(0).sum())
        c1, c2 = st.columns([1, 2])
        c1.metric("Active weight total", f"{total:g}%", delta=None if total == 100 else f"{total - 100:+g}%",
                  delta_color="inverse")
        if len(active):
            chart = alt.Chart(active.assign(name=active["name"].astype(str))).mark_bar(
                color=BLUE, cornerRadiusEnd=4, size=18).encode(
                x=alt.X("weight:Q", title="Weight %"), y=alt.Y("name:N", sort="-x", title=None),
                tooltip=["name", "weight", "max_score"]).properties(height=34 * len(active) + 20)
            c2.altair_chart(chart, use_container_width=True)
        rows, kept = [], set()
        for r in edited.to_dict("records"):
            if not str(r.get("name") or "").strip():
                continue
            cid = int(r["criterion_id"]) if pd.notna(r.get("criterion_id")) else None
            kept.add(cid)
            rows.append({"criterion_id": cid, "name": str(r["name"]), "description": str(r.get("description") or ""),
                         "industry_benchmark": str(r.get("industry_benchmark") or ""),
                         "weight": float(r.get("weight") or 0), "max_score": int(r.get("max_score") or 10),
                         "is_active": 1 if r.get("is_active") else 0})
        for c in all_c:   # deleted rows are deactivated, never destroyed (audit trail)
            if c["criterion_id"] not in kept:
                rows.append({**c, "is_active": 0})
        problems = db.validate_criteria_set(rows)
        for p in problems:
            st.error(p)
        if st.button("💾 Save criteria", type="primary", disabled=bool(problems)):
            db.save_criteria(t["template_id"], rows)
            st.success("Criteria saved. The next evaluation uses them.")
            st.rerun()
        st.caption("Tip: untick Active to exclude a criterion without losing it. Deleted rows are deactivated.")

    # ---------------------------------------------------------------- prompt
    with tab_p:
        st.markdown("The Evaluation Agent is guided by two prompts: **system instructions** (the rules) and the "
                    "**request template** (what is sent for each supplier). Edit and save to create a new version; "
                    "every run records which versions it used.")
        labels = {k: v[0] for k, v in prompts.PROMPTS.items()}
        name = st.selectbox("Prompt", list(labels), format_func=labels.get)
        current = db.get_prompt(name)
        _, default_text, required = prompts.PROMPTS[name]
        st.markdown(f"Active version: **v{current['version']}**"
                    + (" (built-in default)" if current["version"] == 0 else f" · {current.get('note') or ''}"))
        if required:
            st.caption("Required placeholders: " + ", ".join("{" + r + "}" for r in required)
                       + ". Write literal braces as {{ and }}.")
        text = st.text_area("Prompt text", current["text"], height=380, key=f"prompt_{name}_{current['version']}")
        note = st.text_input("What did you change? (saved with the version)", key=f"note_{name}")
        problems = prompts.check_prompt(name, text) if text != current["text"] else []
        for p in problems:
            st.error(p)
        b1, b2, _ = st.columns([1.2, 1.2, 3])
        if b1.button("💾 Save as new version", type="primary", disabled=text == current["text"] or bool(problems)):
            v = db.save_prompt_version(name, text, note)
            st.success(f"Saved and activated version {v}.")
            st.rerun()
        if b2.button("↩ Restore built-in default", disabled=current["version"] == 0):
            db.activate_prompt_version(name, 0)
            st.rerun()

        versions = db.list_prompt_versions(name)
        if versions:
            with st.expander(f"Version history ({len(versions)})"):
                st.dataframe(pd.DataFrame([{"Version": v["version"], "Active": "✔" if v["is_active"] else "",
                                            "Saved": v["created_at"], "Note": v["note"]} for v in versions]),
                             hide_index=True, use_container_width=True)
                pick = st.selectbox("Activate a previous version", [0] + [v["version"] for v in versions],
                                    format_func=lambda v: "Built-in default" if v == 0 else f"v{v}")
                if st.button("Activate selected version"):
                    db.activate_prompt_version(name, pick)
                    st.rerun()

        with st.expander("👁 Preview the full evaluation prompt for this template"):
            crit = db.get_criteria(t["template_id"])
            msgs = build_messages("<Supplier name>", "<the supplier's proposal text is inserted here>", crit, t)
            st.markdown("**System message**")
            st.code(msgs[0]["content"], language=None)
            st.markdown("**User message**")
            st.code(msgs[1]["content"], language=None)
