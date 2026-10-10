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


def test_public_demo_uses_snapshot_and_private_paper_account(tmp_path, monkeypatch):
    """With no local data, the app falls back to the committed snapshot in demo mode."""
    import streamlit as st

    from paper_trading.service import SNAPSHOT_DIR, Settings

    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("QUANT_PAPER_DB", str(tmp_path / "shared.sqlite"))
    cfg = Settings.from_config()
    assert cfg.data_dir == SNAPSHOT_DIR and cfg.public_demo

    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file("../paper_trading/dashboard.py", default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Public demo" in i.value for i in at.info)
    assert not (tmp_path / "shared.sqlite").exists()  # the shared account is never touched
    st.cache_data.clear()
    st.cache_resource.clear()


def test_local_data_is_not_demo_mode(synthetic_market, monkeypatch):
    from paper_trading.service import Settings

    monkeypatch.setenv("QUANT_DATA_DIR", str(synthetic_market["data_dir"]))
    monkeypatch.delenv("QUANT_PUBLIC_DEMO", raising=False)
    cfg = Settings.from_config()
    assert cfg.data_dir == synthetic_market["data_dir"] and not cfg.public_demo
    monkeypatch.setenv("QUANT_PUBLIC_DEMO", "1")
    assert Settings.from_config().public_demo
