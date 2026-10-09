"""Shared building blocks for tutorial chapters: data access, layout and quizzes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from data_pipeline.storage import ParquetStore
from paper_trading.service import Settings


@dataclass(frozen=True)
class Chapter:
    slug: str
    number: int
    title: str
    icon: str
    summary: str
    minutes: int
    render: Callable[[], None]


# Filled by paper_trading.tutorial.__init__ so chapters can link to each other.
CHAPTERS: list[Chapter] = []
PAGES: dict[str, object] = {}


# ---- data ----------------------------------------------------------------------------


@st.cache_resource
def settings() -> Settings:
    return Settings.from_config()


def all_symbols() -> list[str]:
    cfg = settings()
    return [cfg.benchmark] + cfg.universe


@st.cache_data(show_spinner=False)
def ohlcv(symbol: str) -> pd.DataFrame:
    df = ParquetStore(settings().data_dir).load(symbol)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


@st.cache_data(show_spinner=False)
def closes(symbols: tuple[str, ...] | None = None) -> pd.DataFrame:
    symbols = symbols or tuple(all_symbols())
    frames = {s: ohlcv(s)["close"] for s in symbols}
    return pd.DataFrame({s: f for s, f in frames.items() if not f.empty}).sort_index()


def require_data() -> bool:
    """Show a friendly message instead of crashing when no data has been fetched."""
    if ohlcv(settings().benchmark).empty:
        st.warning(
            "No market data found. Run `quant-fetch --config config/watchlist.yaml` "
            "(or `python -m data_pipeline.cli`) and reload this page."
        )
        return False
    return True


# ---- layout ----------------------------------------------------------------------------


def chapter_header(slug: str) -> Chapter:
    ch = next(c for c in CHAPTERS if c.slug == slug)
    done = completed()
    st.caption(
        f"Chapter {ch.number} of {len(CHAPTERS) - 1} · about {ch.minutes} min · "
        f"{len(done)}/{len(CHAPTERS)} chapters completed"
    )
    st.title(f"{ch.icon} {ch.title}")
    st.markdown(f"*{ch.summary}*")
    st.progress(len(done) / len(CHAPTERS))
    return ch


def section(title: str) -> None:
    st.divider()
    st.header(title)


def try_it(title: str = "Try it yourself"):
    """Bordered box for interactive experiments."""
    box = st.container(border=True)
    box.markdown(f"#### 🧪 {title}")
    return box


def key_idea(text: str) -> None:
    st.info(f"**Key idea.** {text}", icon="💡")


def warning_box(text: str) -> None:
    st.warning(text, icon="⚠️")


def takeaways(points: Sequence[str]) -> None:
    with st.container(border=True):
        st.markdown("#### ✅ What to remember")
        st.markdown("\n".join(f"- {p}" for p in points))


def term(word: str, definition: str) -> None:
    """Inline glossary entry rendered as a small expander."""
    with st.expander(f"📖 **{word}**"):
        st.markdown(definition)


def quiz(
    key: str,
    question: str,
    options: Sequence[str],
    answer: int,
    explanation: str,
) -> Optional[bool]:
    """Multiple-choice question; returns True/False once answered, None before."""
    with st.container(border=True):
        st.markdown(f"**❓ {question}**")
        choice = st.radio(
            question, options, index=None, key=f"quiz_{key}", label_visibility="collapsed"
        )
        if choice is None:
            return None
        correct = options.index(choice) == answer
        if correct:
            st.success(f"Correct! {explanation}")
        else:
            st.error(f"Not quite. The answer is **{options[answer]}**. {explanation}")
        st.session_state.setdefault("quiz_results", {})[key] = correct
        return correct


def completed() -> set[str]:
    return st.session_state.setdefault("completed_chapters", set())


def chapter_footer(slug: str, quiz_keys: Sequence[str] = ()) -> None:
    """Mark the chapter done when its quizzes are answered, and link to neighbours."""
    results = st.session_state.get("quiz_results", {})
    if all(k in results for k in quiz_keys):
        completed().add(slug)

    st.divider()
    idx = next(i for i, c in enumerate(CHAPTERS) if c.slug == slug)
    prev_ch = CHAPTERS[idx - 1] if idx > 0 else None
    next_ch = CHAPTERS[idx + 1] if idx + 1 < len(CHAPTERS) else None
    left, mid, right = st.columns([1, 1, 1])
    if prev_ch and prev_ch.slug in PAGES:
        left.page_link(PAGES[prev_ch.slug], label=f"← {prev_ch.title}")
    if slug in completed():
        mid.success("Chapter completed", icon="🎉")
    elif quiz_keys:
        mid.caption("Answer the quiz questions to complete this chapter.")
    if next_ch and next_ch.slug in PAGES:
        right.page_link(PAGES[next_ch.slug], label=f"Next: {next_ch.title} →")


# ---- small math helpers used by several chapters ------------------------------------------


def daily_returns(prices: pd.Series | pd.DataFrame):
    return prices.pct_change(fill_method=None).dropna(how="all")


def drawdown(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def annualized_return(prices: pd.Series) -> float:
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(np.sqrt(252) * returns.mean() / std) if std > 0 else 0.0


def fmt_pct(x: float, signed: bool = False) -> str:
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def fmt_usd(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"
