"""Ask the proposals: RAG question answering over the supplier documents of a run."""

import streamlit as st

from rfp import rag_tool
from ui.common import get_client, page_header, run_picker

SUGGESTIONS = [
    "Which suppliers hold ISO 27001 or SOC 2 certification?",
    "What does each supplier say about integration with SuccessFactors?",
    "Compare the delivery timelines and milestones.",
    "What pricing assumptions or exclusions are stated?",
    "What support hours and response times are offered?",
]


def render():
    ss = st.session_state
    page_header("💬 Ask the proposals",
                "Ask any question across the supplier documents of a run. The answer is built only from retrieved "
                "passages and cites them, so you can check every claim.")
    run = run_picker(key="ask_run")
    if not run:
        return
    names = [s["supplier_name"] for s in run["suppliers"]]
    scope = st.multiselect("Suppliers in scope", names, default=names)
    if get_client() is None:
        st.caption("No AI connected: you'll see the most relevant passages instead of a written answer.")
    st.markdown("Suggested questions:")
    cols = st.columns(len(SUGGESTIONS))
    for col, q in zip(cols, SUGGESTIONS):
        if col.button(q, use_container_width=True):
            ss.ask_q = q
    q = st.text_input("Your question", value=ss.get("ask_q", ""), placeholder="e.g. Who offers 24x7 support?")
    history = ss.setdefault("ask_history", {}).setdefault(run["rfp_run_id"], [])
    if st.button("Ask", type="primary", disabled=not q.strip() or not scope):
        with st.spinner("Retrieving passages and drafting the answer..."):
            res = rag_tool.answer(q.strip(), run["rfp_run_id"], scope, get_client())
        history.insert(0, {"q": q.strip(), **res})
    for item in history:
        with st.chat_message("user"):
            st.markdown(item["q"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if item["sources"]:
                with st.expander(f"Sources ({len(item['sources'])} passages)"):
                    for s in item["sources"]:
                        st.markdown(f"**[{s['n']}] {s['supplier']} · page {s['page']}** · relevance {s['score']}")
                        st.caption(s["text"])
