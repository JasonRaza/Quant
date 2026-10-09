"""Interactive beginner tutorial: from "what is a stock" to ML and live trading."""

from __future__ import annotations

import streamlit as st

from paper_trading.tutorial import ch00_start, ch01_markets, ch02_returns_risk, ch03_compare
from paper_trading.tutorial.common import CHAPTERS, PAGES, Chapter

_CHAPTERS = [
    Chapter(
        ch00_start.SLUG,
        0,
        "Start here",
        "🧭",
        "What quant finance is and how to use this tutorial",
        5,
        ch00_start.render,
    ),
    Chapter(
        ch01_markets.SLUG,
        1,
        "Markets and prices",
        "🏛️",
        "Stocks, ETFs, tickers and how to read OHLCV price bars",
        10,
        ch01_markets.render,
    ),
    Chapter(
        ch02_returns_risk.SLUG,
        2,
        "Returns and risk",
        "🎢",
        "Returns, compounding, volatility and drawdowns",
        15,
        ch02_returns_risk.render,
    ),
    Chapter(
        ch03_compare.SLUG,
        3,
        "Comparing investments",
        "⚖️",
        "Sharpe ratio, correlation, diversification and beta",
        15,
        ch03_compare.render,
    ),
]

CHAPTERS[:] = _CHAPTERS


def build_pages(default_slug: str = "start") -> list:
    """Create one Streamlit page per chapter and register them for cross-links."""
    pages = []
    for ch in CHAPTERS:
        page = st.Page(
            ch.render,
            title=f"{ch.number}. {ch.title}",
            icon=ch.icon,
            url_path=ch.slug,
            default=ch.slug == default_slug,
        )
        PAGES[ch.slug] = page
        pages.append(page)
    return pages
