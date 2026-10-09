"""Streamlit entry point: the Learn tutorial plus the Lab research dashboard.

Run: quant-dashboard  (or: streamlit run paper_trading/app.py)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from paper_trading.tutorial import build_pages

st.set_page_config(page_title="Quant Lab", page_icon="📈", layout="wide")

lab = st.Page(
    str(Path(__file__).with_name("dashboard.py")),
    title="Research dashboard",
    icon="🔬",
    url_path="lab",
)
st.navigation({"Learn": build_pages(), "Lab": [lab]}, expanded=True).run()
