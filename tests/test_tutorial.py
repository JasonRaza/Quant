"""Smoke tests: every tutorial chapter renders on synthetic data without errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from data_pipeline.storage import ParquetStore
from paper_trading.tutorial import CHAPTERS
from shared.config import load_yaml, resolve_path

PAGE_SCRIPT = """
import streamlit as st
from paper_trading.tutorial import build_pages
st.navigation(build_pages(default_slug="{slug}"), position="hidden").run()
"""


@pytest.fixture(scope="module")
def synthetic_market(tmp_path_factory):
    """Six-plus years of random-walk bars for every watchlist ticker."""
    data_dir = tmp_path_factory.mktemp("parquet")
    store = ParquetStore(data_dir)
    symbols = load_yaml(resolve_path("config/watchlist.yaml"))["symbols"]
    dates = pd.bdate_range("2020-01-02", "2026-06-30")
    rng = np.random.default_rng(0)
    for i, sym in enumerate(symbols):
        close = 50 * (i + 1) * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(dates))))
        store.save(
            sym,
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates.date,
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
                }
            ),
        )
    mp = pytest.MonkeyPatch()
    mp.setenv("QUANT_DATA_DIR", str(data_dir))
    mp.setenv("QUANT_PAPER_DB", str(tmp_path_factory.mktemp("db") / "paper.sqlite"))
    st.cache_data.clear()
    st.cache_resource.clear()
    yield data_dir
    mp.undo()
    st.cache_data.clear()
    st.cache_resource.clear()


def run_chapter(slug: str) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT.format(slug=slug), default_timeout=300)
    return at.run()


@pytest.mark.parametrize("slug", [ch.slug for ch in CHAPTERS])
def test_chapter_renders(synthetic_market, slug):
    at = run_chapter(slug)
    assert not at.exception, [e.value for e in at.exception]
    chapter = next(ch for ch in CHAPTERS if ch.slug == slug)
    assert at.title[0].value.endswith(chapter.title)


def test_quiz_feedback_and_completion(synthetic_market):
    at = run_chapter("start")
    quiz = at.radio(key="quiz_start_1")
    quiz.set_value(quiz.options[1]).run()
    assert any("Correct" in s.value for s in at.success)
    assert "start" in at.session_state["completed_chapters"]

    at.radio(key="quiz_start_1").set_value(quiz.options[0]).run()
    assert any("Not quite" in e.value for e in at.error)


def test_chapter_numbers_are_sequential():
    assert [ch.number for ch in CHAPTERS] == list(range(len(CHAPTERS)))
    assert len({ch.slug for ch in CHAPTERS}) == len(CHAPTERS)
