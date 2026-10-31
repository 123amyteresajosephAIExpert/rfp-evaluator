"""Templates: pick a saved RFP template or create one from a client RFP document."""

import pandas as pd
import streamlit as st

from rfp import config, db
from rfp.document_tool import extract_pdf_text
from rfp.template_agent import draft_scheme
from ui.common import PAGES, SAMPLE_SETS, SAMPLES, ai_ready, get_client, need_ai_message, page_header


def _criteria_editor(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return st.data_editor(
        df, key=key, num_rows="dynamic", hide_index=True, use_container_width=True,
        column_order=["name", "weight", "max_score", "description", "industry_benchmark"],
        column_config={
            "name": st.column_config.TextColumn("Criterion", required=True, width="medium"),
            "weight": st.column_config.NumberColumn("Weight %", min_value=0, max_value=100, step=1, format="%g"),
            "max_score": st.column_config.NumberColumn("Max score", min_value=1, max_value=100, step=1),
            "description": st.column_config.TextColumn("What the AI should inspect", width="large"),
            "industry_benchmark": st.column_config.TextColumn("Industry best practice", width="large"),
        })


def render():
    ss = st.session_state
    page_header("📄 RFP templates",
                "A template is the client's RFP turned into an evaluation scheme: what is being bought, the key "
                "requirements, and the criteria, weights and industry benchmarks every proposal is scored against.")

    tab_pick, tab_new = st.tabs(["Choose a saved template", "Create from a client RFP"])

    # ---------------------------------------------------------------- choose
    with tab_pick:
        templates = db.list_templates()
        if not templates:
            st.info("No templates yet - create one in the next tab.")
        else:
            names = [t["name"] for t in templates]
            current = next((t for t in templates if t["template_id"] == ss.get("template_id")), templates[0])
            pick = st.selectbox("Template to use for evaluations", names, index=names.index(current["name"]))
            t = templates[names.index(pick)]
            ss.template_id = t["template_id"]
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"**Client:** {t.get('buyer') or '-'}  \n**Industry:** {t.get('industry') or '-'}  \n"
                            f"**Created:** {t['created_at']} · **How:** {t.get('parse_mode')}"
                            + (f" · from `{t['source_filename']}`" if t.get("source_filename") else ""))
                if t.get("summary"):
                    st.markdown(f"**Summary:** {t['summary']}")
            with c2:
                if t.get("requirements"):
                    st.markdown("**Key requirements**")
                    st.markdown("\n".join(f"- {r}" for r in t["requirements"]))
            crit = db.get_criteria(t["template_id"])
            st.dataframe(pd.DataFrame([{"Criterion": c["name"], "Weight %": c["weight"], "Max": c["max_score"],
                                        "What the AI inspects": c["description"],
                                        "Industry best practice": c["industry_benchmark"]} for c in crit]),
                         hide_index=True, use_container_width=True)
            b1, b2, _ = st.columns([1.2, 1.2, 3])
            if "criteria" in PAGES:
                b1.page_link(PAGES["criteria"], label="Edit criteria & prompt", icon="⚖️")
            if "evaluate" in PAGES:
                b2.page_link(PAGES["evaluate"], label="Evaluate proposals", icon="📤")
            with st.expander("Archive this template"):
                st.caption("Archived templates are hidden from lists; past runs that used them stay available.")
                if st.button("Archive", disabled=len(templates) < 2):
                    db.archive_template(t["template_id"])
                    ss.pop("template_id", None)
                    st.rerun()

    # ---------------------------------------------------------------- create
    with tab_new:
        st.markdown("Upload the client's RFP / RFQ document. The **Template Agent** reads it and drafts the "
                    "criteria, weights and industry benchmarks. You review everything before it is saved.")
        up = st.file_uploader("Client RFP document (PDF)", type=["pdf"], key="tpl_upload")
        st.caption("No RFP handy? Use one of the samples:")
        cols = st.columns(len(SAMPLE_SETS))
        for col, (label, (_, fname)) in zip(cols, SAMPLE_SETS.items()):
            if col.button(f"Use sample: {label.split(' (')[0]}", use_container_width=True):
                ss.tpl_sample = fname
        if up is not None:
            data, fname = up.getvalue(), up.name
        elif ss.get("tpl_sample"):
            fname = ss.tpl_sample
            data = (SAMPLES / "templates" / fname).read_bytes()
        else:
            data, fname = None, None

        if data:
            st.markdown(f"Selected: `{fname}`")
            extraction = extract_pdf_text(data, fname)
            if not extraction.ok:
                st.error(extraction.error)
            else:
                existing = db.find_template_by_sha(extraction.sha256)
                if existing:
                    st.info(f"This exact document was already turned into the template **{existing['name']}** "
                            "(same fingerprint). You can use it from the first tab, or draft a new version below.")
                with st.expander("Preview extracted text"):
                    st.text(extraction.text[:4000] + ("..." if len(extraction.text) > 4000 else ""))
                go = st.button("🤖 Read RFP and draft criteria", type="primary", disabled=not ai_ready())
                if not ai_ready():
                    need_ai_message()
                if go:
                    with st.spinner("Template Agent is reading the RFP..."):
                        draft = draft_scheme(extraction.text, fname, get_client())
                    draft["source"] = {"filename": fname, "sha": extraction.sha256, "text": extraction.text}
                    ss.tpl_draft = draft
                    ss.tpl_draft_n = ss.get("tpl_draft_n", 0) + 1

        draft = ss.get("tpl_draft")
        if draft:
            st.divider()
            mode = "AI" if draft["mode"] == "LLM" else "rules-based parser (no AI)"
            st.markdown(f"#### Review the draft · drafted by {mode}")
            for n in draft["notes"]:
                st.warning(n)
            info = draft["template"]
            c1, c2 = st.columns(2)
            name = c1.text_input("Template name", info["name"])
            buyer = c2.text_input("Client", info["buyer"])
            industry = c1.text_input("Industry", info["industry"])
            summary = c2.text_area("Summary", info["summary"], height=100)
            reqs = st.text_area("Key requirements (one per line)", "\n".join(info["requirements"]), height=120)
            df = pd.DataFrame(draft["criteria"] or [{"name": "", "weight": 0, "max_score": 10, "description": "",
                                                     "industry_benchmark": ""}])
            edited = _criteria_editor(df, key=f"tpl_crit_{ss.get('tpl_draft_n', 0)}")
            total = float(edited["weight"].fillna(0).sum())
            st.metric("Weight total", f"{total:g}%", delta=None if total == 100 else f"{total - 100:+g}%",
                      delta_color="inverse")
            rows = [{"name": str(r["name"]).strip(), "weight": float(r["weight"] or 0),
                     "max_score": int(r["max_score"] or config.DEFAULT_MAX_SCORE),
                     "description": str(r.get("description") or ""), "industry_benchmark": str(r.get("industry_benchmark") or ""),
                     "is_active": 1} for r in edited.to_dict("records") if str(r.get("name") or "").strip()]
            problems = db.validate_criteria_set(rows)
            for p in problems:
                st.error(p)
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("💾 Save template", type="primary", disabled=bool(problems) or not name.strip()):
                src = draft["source"]
                tid = db.save_template({"name": name, "buyer": buyer, "industry": industry, "summary": summary,
                                        "requirements": [r.strip() for r in reqs.splitlines() if r.strip()],
                                        "source_filename": src["filename"], "source_sha256": src["sha"],
                                        "template_text": src["text"], "parse_mode": draft["mode"]}, rows)
                ss.template_id = tid
                ss.pop("tpl_draft", None)
                ss.pop("tpl_sample", None)
                st.success("Template saved and set as active.")
                st.rerun()
            if b2.button("Discard draft"):
                ss.pop("tpl_draft", None)
                st.rerun()
