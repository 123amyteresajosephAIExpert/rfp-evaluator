"""Home: what the app does, a live checklist and the sample pack."""

import streamlit as st

from rfp import db
from ui.common import PAGES, active_template, ai_ready, sample_pack_zip


def render():
    st.markdown(
        "<div class='hero'><h1>📑 AI Procurement Evaluator</h1>"
        "<p>Upload a client RFP, let AI draft the evaluation criteria, score every supplier proposal with quoted "
        "evidence, and get an explainable, repeatable ranking against peers and industry best practice.</p></div>",
        unsafe_allow_html=True)

    t = active_template()
    runs = [r for r in db.list_runs() if r["status"] == "COMPLETED"]
    steps = [
        ("Connect AI", ai_ready(), "Paste an API key in the sidebar (or choose Offline demo).", None),
        ("Choose or create a template", t is not None,
         f"Active: {t['name']}" if t else "Upload the client's RFP and let AI draft the criteria.", "templates"),
        ("Review criteria and prompt", t is not None, "Adjust weights, industry benchmarks or the AI prompt if needed.",
         "criteria"),
        ("Evaluate proposals", bool(runs), f"{len(runs)} completed run(s) so far." if runs else
         "Upload supplier PDFs and click Evaluate.", "evaluate"),
        ("Review, question and export", bool(runs), "Leaderboard, peer and industry views, Q&A, PDF report.", "results"),
    ]
    st.markdown("#### Your progress")
    cols = st.columns(len(steps))
    for i, (col, (title, done, hint, page)) in enumerate(zip(cols, steps), start=1):
        with col:
            st.markdown(f"<div class='card'><h4><span class='step-num'>{i}</span>{'✅ ' if done else ''}{title}</h4>"
                        f"<p class='muted'>{hint}</p></div>", unsafe_allow_html=True)
            if page and page in PAGES:
                st.page_link(PAGES[page], label="Open", icon="➡️")

    st.markdown("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### How it works")
        st.markdown(
            "1. **Template Agent** reads the client's RFP and drafts criteria, weights and an industry benchmark "
            "for each criterion. You review and edit before saving.\n"
            "2. **Document Tool** extracts clean text from each supplier PDF and rejects scanned or broken files.\n"
            "3. **Evaluation Agent** (the AI) scores each supplier on every criterion, quoting evidence from the "
            "proposal, and lists strengths, weaknesses and risks.\n"
            "4. **Validation Tool** checks every AI answer: missing or out-of-range scores are fixed, and every "
            "quote is checked against the PDF.\n"
            "5. **Ranking Tool** (plain Python) calculates weighted scores, peer benchmarks, the Peer Performance "
            "Index, industry alignment and the tie-breaks.\n"
            "6. Everything is saved in **SQLite**: past runs can be reopened, questioned, adjusted and exported, "
            "and identical documents reuse their stored **evaluation DNA**.")
        st.markdown("<div class='principle'><b>Design principle: AI reads, code decides.</b> The AI judges proposal "
                    "content. It never does the arithmetic, the benchmarks, the tie-breaks or the ranking, so the same "
                    "validated scores always give the same result, and every number traces back to evidence.</div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### Try it in two minutes")
        st.markdown("The app ships with two fictional procurement exercises: an **HR service delivery platform** "
                    "RFP with 5 supplier proposals, and a **corporate fleet lease** RFQ with 3 quotes. The pages "
                    "have buttons to load them directly, or download the pack:")
        st.download_button("⬇ Download sample documents (ZIP)", sample_pack_zip(), "sample_documents.zip",
                           "application/zip", use_container_width=True)
        st.markdown("#### What makes this different")
        st.markdown("- Criteria are drafted from **any** client RFP, so it works beyond one domain\n"
                    "- **Editable, versioned AI prompt**\n"
                    "- **Peer and industry** benchmarking\n"
                    "- **Evidence check** catches invented quotes\n"
                    "- **Ask questions** across all proposals\n"
                    "- **Analyst overrides** with a reason and audit trail\n"
                    "- **Evaluation DNA** reuses past scoring\n"
                    "- One-click **PDF report**")
