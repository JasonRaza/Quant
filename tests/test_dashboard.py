"""Smoke tests for the Streamlit entry point and the Lab dashboard."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_app_entry_point_opens_tutorial(streamlit_on_synthetic_data):
    at = AppTest.from_file("../paper_trading/app.py", default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Start here"


def test_lab_dashboard_renders_and_switches_strategy(streamlit_on_synthetic_data):
    at = AppTest.from_file("../paper_trading/dashboard.py", default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label.split(": ")[-1] for t in at.tabs]
    assert labels == ["Market", "Backtest", "Factors", "Portfolio", "ML alpha", "Paper trading"]

    strategy = next(r for r in at.radio if r.label == "Strategy")
    strategy.set_value("mean_reversion").run()
    assert not at.exception, [e.value for e in at.exception]
