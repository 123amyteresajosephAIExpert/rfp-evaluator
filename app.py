"""AI Procurement Evaluator - Streamlit entry point.

Run locally:  streamlit run app.py
"""

import streamlit as st

from rfp import db
from ui import common, page_ask, page_criteria, page_evaluate, page_history, page_home, page_results, page_templates

st.set_page_config(page_title="AI Procurement Evaluator", page_icon="📑", layout="wide")


db.init_db()  # idempotent and fast: creates tables / sample template only if missing
common.inject_css()

pages = {
    "home": st.Page(page_home.render, title="Home", icon="🏠", url_path="home", default=True),
    "templates": st.Page(page_templates.render, title="RFP templates", icon="📄", url_path="templates"),
    "criteria": st.Page(page_criteria.render, title="Criteria & prompt", icon="⚖️", url_path="criteria"),
    "evaluate": st.Page(page_evaluate.render, title="Evaluate", icon="📤", url_path="evaluate"),
    "results": st.Page(page_results.render, title="Results", icon="🏆", url_path="results"),
    "ask": st.Page(page_ask.render, title="Ask the proposals", icon="💬", url_path="ask"),
    "history": st.Page(page_history.render, title="History & DNA", icon="🗂️", url_path="history"),
}
common.PAGES.update(pages)

nav = st.navigation({
    "Start": [pages["home"]],
    "Set up": [pages["templates"], pages["criteria"]],
    "Run": [pages["evaluate"]],
    "Review": [pages["results"], pages["ask"], pages["history"]],
})
common.ai_panel()
nav.run()
