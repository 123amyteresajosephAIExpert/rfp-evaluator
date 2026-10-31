"""Shared UI helpers: styling, the AI connection panel, session state and small widgets."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from rfp import config, db
from rfp.evaluation_agent import FaultInjectingEvaluator, LLMEvaluator, OfflineDemoEvaluator
from rfp.llm_client import LLMClient

ROOT = config.PROJECT_ROOT
SAMPLES = config.SAMPLE_DIR
SAMPLE_META = json.loads((SAMPLES / "sample_metadata.json").read_text())
SAMPLE_SETS = {
    "HR service delivery platform (5 proposals)": ("hr_platform", "Kestrel_HR_Service_Delivery_RFP.pdf"),
    "Corporate fleet lease (3 quotes)": ("fleet_lease", "Meridian_Fleet_Lease_RFQ.pdf"),
}
BLUE, ORANGE, INK = "#2a6fdb", "#e8743b", "#1d2733"
PAGES: dict = {}   # filled by app.py: name -> st.Page

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #1f3a5f 0%, #2a6fdb 100%); color: #fff; padding: 26px 30px;
       border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0 0 6px 0; font-size: 2.0rem;}
.hero p {color: #e6eefc; margin: 0; font-size: 1.02rem;}
.card {border: 1px solid #dfe5ee; border-radius: 12px; padding: 16px 18px; background: #fbfcfe; height: 100%;}
.card h4 {margin: 0 0 6px 0; color: #1f3a5f; font-size: 1.02rem; line-height: 1.35;}
.step-num {display: inline-block; width: 28px; height: 28px; border-radius: 50%; background: #2a6fdb; color: #fff;
           text-align: center; line-height: 28px; font-weight: 700; margin-right: 8px;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.80rem; margin: 2px 4px 2px 0;
       border: 1px solid #c9d6ea; background: #eef3fb; color: #1f3a5f;}
.pill.ok {background: #e8f6ee; border-color: #b6e0c6; color: #1c6b3c;}
.pill.warn {background: #fff6e5; border-color: #f2d49b; color: #8a5a00;}
.pill.bad {background: #fdecec; border-color: #f2b8b8; color: #9b1c1c;}
.muted {color: #5b6676; font-size: 0.92rem;}
.principle {border-left: 4px solid #2a6fdb; background: #f3f7fe; padding: 10px 14px; border-radius: 6px;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f"<span class='pill {kind}'>{text}</span>"


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p class='muted'>{subtitle}</p>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------
# AI connection (key is kept only in this browser session's memory)
# ------------------------------------------------------------------------------------
def ai_panel():
    ss = st.session_state
    ss.setdefault("ai_mode", "AI")
    ss.setdefault("ai_connected", False)
    ss.setdefault("ai_label", "")
    with st.sidebar:
        st.markdown("### 🔑 AI connection")
        mode = st.radio("Mode", ["Use an AI provider", "Offline demo (no AI key)"],
                        index=0 if ss.ai_mode == "AI" else 1, key="ai_mode_radio",
                        help="Offline demo runs the whole workflow with a transparent keyword heuristic instead of an "
                             "AI. Use it to try the app; use an AI provider for real evaluations.")
        ss.ai_mode = "AI" if mode.startswith("Use") else "OFFLINE"
        if ss.ai_mode == "AI":
            provider = st.selectbox("Provider", list(config.PROVIDERS), key="ai_provider",
                                    index=list(config.PROVIDERS).index(config.DEFAULT_PROVIDER))
            p = config.PROVIDERS[provider]
            st.caption(p["key_hint"])
            key = st.text_input("API key", type="password", key="ai_key",
                                help="Kept only in this browser session's memory. Never saved to disk, the "
                                     "database or GitHub.")
            with st.expander("Model (optional)"):
                st.text_input("Model name", value=p["default_model"], key=f"ai_model_{provider}")
            if st.button("Connect", type="primary", use_container_width=True, disabled=not key):
                try:
                    client = LLMClient(provider, key, ss.get(f"ai_model_{provider}"))
                    ok, msg = client.test_connection()
                except Exception as exc:  # noqa: BLE001
                    ok, msg = False, str(exc)
                ss.ai_connected, ss.ai_label = ok, (client.label if ok else "")
                ss.ai_message = msg
            if ss.get("ai_connected") and ss.ai_label.startswith(provider):
                st.success(f"Connected · {ss.ai_label}")
            elif ss.get("ai_message") and not ss.get("ai_connected"):
                st.error(ss.ai_message)
            else:
                st.info("Paste a key and click Connect.")
            st.caption("🔒 Your key stays in this session only.")
        else:
            st.warning("Offline demo: scores come from keyword counts, not AI. For trying the app only.")
        st.divider()
        t = active_template()
        if t:
            st.markdown(f"**Active template**  \n{t['name']}")
        if ss.get("run_id"):
            st.markdown(f"**Open run**  \n`{ss.run_id}`")


def ai_ready() -> bool:
    ss = st.session_state
    return ss.get("ai_mode") == "OFFLINE" or get_client() is not None


def get_client() -> LLMClient | None:
    ss = st.session_state
    provider = ss.get("ai_provider", config.DEFAULT_PROVIDER)
    if ss.get("ai_mode") != "AI" or not ss.get("ai_connected") or not ss.get("ai_label", "").startswith(provider):
        return None
    return LLMClient(provider, ss.get("ai_key", ""), ss.get(f"ai_model_{provider}"))


def get_evaluator(fault_names: set[str] | None = None):
    client = get_client()
    ev = LLMEvaluator(client) if client else OfflineDemoEvaluator()
    return FaultInjectingEvaluator(ev, fault_names) if fault_names else ev


def need_ai_message():
    st.warning("Connect an AI provider in the sidebar (or switch to Offline demo) to continue.")


# ------------------------------------------------------------------------------------
# Template and run selection
# ------------------------------------------------------------------------------------
def active_template() -> dict | None:
    ss = st.session_state
    templates = db.list_templates()
    if not templates:
        return None
    ids = [t["template_id"] for t in templates]
    if ss.get("template_id") not in ids:
        ss.template_id = ids[0]
    return next(t for t in templates if t["template_id"] == ss.template_id)


def template_picker(label: str = "Template", key: str = "tpl_pick") -> dict | None:
    templates = db.list_templates()
    if not templates:
        st.info("No templates yet. Create one on the Templates page.")
        return None
    current = active_template()
    names = [t["name"] for t in templates]
    choice = st.selectbox(label, names, index=names.index(current["name"]), key=key)
    t = templates[names.index(choice)]
    st.session_state.template_id = t["template_id"]
    return t


def run_picker(key: str = "run_pick") -> dict | None:
    runs = [r for r in db.list_runs() if r["status"] == "COMPLETED"]
    if not runs:
        st.info("No completed evaluations yet. Run one on the **Evaluate** page.")
        if "evaluate" in PAGES:
            st.page_link(PAGES["evaluate"], label="Go to Evaluate", icon="📤")
        return None
    ids = [r["rfp_run_id"] for r in runs]
    ss = st.session_state
    if ss.get("run_id") not in ids:
        ss.run_id = ids[0]
    labels = {r["rfp_run_id"]: f"{r['rfp_run_id']} · {r['template_name']} · winner {r['winner']} · "
                               f"{r['suppliers']} suppliers{' · rev ' + str(r['revision']) if r['revision'] > 1 else ''}"
              for r in runs}
    pick = st.selectbox("Evaluation run", ids, index=ids.index(ss.run_id), format_func=labels.get, key=key)
    ss.run_id = pick
    return db.load_run(pick)


# ------------------------------------------------------------------------------------
# Misc helpers
# ------------------------------------------------------------------------------------
def default_meta(filename: str) -> dict:
    if filename in SAMPLE_META:
        m = SAMPLE_META[filename]
        return {"supplier_name": m["supplier_name"], "submission_date": date.fromisoformat(m["submission_date"]),
                "experience_rating": float(m["experience_rating"])}
    stem = Path(filename).stem
    for junk in ("_RFP_Response", "_Response", "_Proposal", "_Quote", "_RFP", " RFP", "_Copy"):
        stem = stem.replace(junk, "")
    return {"supplier_name": stem.replace("_", " ").strip() or Path(filename).stem,
            "submission_date": date.today(), "experience_rating": 3.0}


def sample_pack_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((SAMPLES / "templates").glob("*.pdf")):
            z.write(p, f"1_client_rfp_templates/{p.name}")
        for p in sorted((SAMPLES / "proposals").rglob("*.pdf")):
            z.write(p, f"2_supplier_proposals/{p.parent.name}/{p.name}")
        for p in sorted((SAMPLES / "edge_cases").glob("*.pdf")):
            z.write(p, f"3_edge_cases_for_validation/{p.name}")
    return buf.getvalue()


def leaderboard_df(run: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": s["final_rank"], "Supplier": s["supplier_name"], "PPI": s["ppi"],
        "Absolute score": s["absolute_score"], "Industry alignment": s["industry_index"],
        "Submitted": s["submission_date"], "Experience": s["experience_rating"],
        "Why this position": s["rank_explanation"],
    } for s in run["suppliers"]])


def fmt(x, d=2) -> str:
    return f"{x:,.{d}f}"
