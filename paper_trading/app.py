"""Streamlit entry point: the Learn tutorial plus the Lab research dashboard.

Run: quant-dashboard  (or: streamlit run paper_trading/app.py)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from paper_trading.tutorial import build_pages
from paper_trading.tutorial.common import PAGES

ASSETS = Path(__file__).with_name("assets")

st.set_page_config(page_title="Quant Lab", page_icon=":material/insights:", layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "mark.svg"), size="large")

lab = st.Page(
    str(Path(__file__).with_name("dashboard.py")),
    title="Research dashboard",
    icon=":material/science:",
    url_path="lab",
)
learn = build_pages()
PAGES["lab"] = lab  # lets tutorial pages link to the dashboard
st.navigation({"Learn": learn, "Lab": [lab]}, expanded=True).run()
