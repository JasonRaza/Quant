"""Smoke tests: every tutorial chapter renders on synthetic data without errors."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from paper_trading.tutorial import CHAPTERS

PAGE_SCRIPT = """
import streamlit as st
from paper_trading.tutorial import build_pages
st.navigation(build_pages(default_slug="{slug}"), position="hidden").run()
"""


def run_chapter(slug: str) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT.format(slug=slug), default_timeout=300)
    return at.run()


@pytest.mark.parametrize("slug", [ch.slug for ch in CHAPTERS])
def test_chapter_renders(streamlit_on_synthetic_data, slug):
    at = run_chapter(slug)
    assert not at.exception, [e.value for e in at.exception]
    chapter = next(ch for ch in CHAPTERS if ch.slug == slug)
    assert at.title[0].value.endswith(chapter.title)


def test_quiz_feedback_and_completion(streamlit_on_synthetic_data):
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
