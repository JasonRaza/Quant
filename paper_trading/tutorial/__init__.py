"""Interactive beginner tutorial: from "what is a stock" to ML and live trading."""

from __future__ import annotations

import streamlit as st

from paper_trading.tutorial import (
    ch00_start,
    ch01_markets,
    ch02_returns_risk,
    ch03_compare,
    ch04_data,
    ch05_backtesting,
    ch06_pitfalls,
    ch07_pairs,
    ch08_factors,
    ch09_portfolio,
    ch10_ml,
    ch11_trading,
    ch12_glossary,
)
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
    Chapter(
        ch04_data.SLUG,
        4,
        "Market data",
        "🗄️",
        "Data pipelines, adjusted prices, data quality and survivorship bias",
        15,
        ch04_data.render,
    ),
    Chapter(
        ch05_backtesting.SLUG,
        5,
        "Strategies and backtesting",
        "⏪",
        "Trading signals, moving averages and how a backtest engine works",
        20,
        ch05_backtesting.render,
    ),
    Chapter(
        ch06_pitfalls.SLUG,
        6,
        "Backtesting pitfalls",
        "🪤",
        "Lookahead bias, overfitting and the multiple-testing trap",
        20,
        ch06_pitfalls.render,
    ),
    Chapter(
        ch07_pairs.SLUG,
        7,
        "Mean reversion and pairs",
        "🔁",
        "Trend vs mean reversion, spreads, z-scores and market neutrality",
        15,
        ch07_pairs.render,
    ),
    Chapter(
        ch08_factors.SLUG,
        8,
        "Factors",
        "🏷️",
        "Ranking stocks, the information coefficient and telling signal from noise",
        20,
        ch08_factors.render,
    ),
    Chapter(
        ch09_portfolio.SLUG,
        9,
        "Building portfolios",
        "🧺",
        "Diversification, optimizers, estimation noise and walk-forward tests",
        20,
        ch09_portfolio.render,
    ),
    Chapter(
        ch10_ml.SLUG,
        10,
        "Machine learning",
        "🤖",
        "Features and labels, overfitting, and why validation must not leak",
        20,
        ch10_ml.render,
    ),
    Chapter(
        ch11_trading.SLUG,
        11,
        "From research to trading",
        "📨",
        "Orders, spreads, paper trading and what production really requires",
        15,
        ch11_trading.render,
    ),
    Chapter(
        ch12_glossary.SLUG,
        12,
        "Glossary and final quiz",
        "🎓",
        "Every key term in one place, and a final check of what you learned",
        10,
        ch12_glossary.render,
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
