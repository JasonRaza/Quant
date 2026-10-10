"""Streamlit entry point: the Learn tutorial plus the Lab research dashboard.

Run: quant-dashboard  (or: streamlit run paper_trading/app.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo's packages importable without `pip install -e .` (hosted deployments
# run this file directly from a clone).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from paper_trading.live_data import live_settings, refresh_status  # noqa: E402
from paper_trading.tutorial import build_pages  # noqa: E402
from paper_trading.tutorial.common import PAGES, ohlcv  # noqa: E402

ASSETS = Path(__file__).with_name("assets")


def data_freshness_note() -> str:
    """One line telling visitors how recent the prices are."""
    bench = ohlcv(live_settings().benchmark)
    if bench.empty:
        return "No market data loaded."
    note = f"Market data through {bench.index[-1].date()}"
    status = refresh_status()
    if status is not None:
        note += " · refreshed automatically" if not status.failed else " · bundled snapshot"
    return note + "."

st.set_page_config(page_title="Quant Lab", page_icon=":material/insights:", layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "mark.svg"), size="large")

lab = st.Page(
    str(Path(__file__).with_name("dashboard.py")),
    title="Research dashboard",
    icon=":material/science:",
    url_path="lab",
)
st.sidebar.caption(data_freshness_note())
st.sidebar.caption(
    "[Source code on GitHub](https://github.com/JasonRaza/Quant) · "
    "Educational project, not investment advice."
)

learn = build_pages()
PAGES["lab"] = lab  # lets tutorial pages link to the dashboard
st.navigation({"Learn": learn, "Lab": [lab]}, expanded=True).run()
