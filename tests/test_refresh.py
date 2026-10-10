"""Tests for refreshing a copy of the data snapshot."""

from datetime import date

import pandas as pd
import pytest

from data_pipeline.refresh import refresh_copy
from data_pipeline.storage import ParquetStore


def bars(symbol, start, n, price=100.0):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"symbol": symbol, "date": dates.date, "open": price, "high": price, "low": price,
         "close": price, "volume": 1_000}
    )


@pytest.fixture
def snapshot(tmp_path):
    store = ParquetStore(tmp_path / "snapshot")
    store.save("AAA", bars("AAA", "2024-01-01", 5))
    store.save("BBB", bars("BBB", "2024-01-01", 5))
    return tmp_path / "snapshot"


def test_appends_missing_days_without_touching_snapshot(snapshot, tmp_path):
    calls = []

    def fake_fetch(symbol, start, **kwargs):
        calls.append((symbol, start, kwargs))
        return bars(symbol, start, 3, price=110.0)

    live = tmp_path / "live"
    report = refresh_copy(snapshot, live, fetch=fake_fetch, today=date(2024, 1, 31))

    assert report.updated == {"AAA": 3, "BBB": 3}
    assert not report.failed
    assert len(ParquetStore(live).load("AAA")) == 8
    assert len(ParquetStore(snapshot).load("AAA")) == 5  # the committed snapshot is read-only
    # Fetches start the day after the snapshot's last bar, with a single quick attempt.
    assert {c[1] for c in calls} == {date(2024, 1, 6)}
    assert all(c[2] == {"max_retries": 1, "retry_delay": 0} for c in calls)
    assert report.last_date == ParquetStore(live).load("AAA")["date"].max()


def test_failed_tickers_keep_their_snapshot_history(snapshot, tmp_path):
    def flaky_fetch(symbol, start, **kwargs):
        if symbol == "BBB":
            raise RuntimeError("429 Too Many Requests")
        return bars(symbol, start, 2)

    report = refresh_copy(snapshot, tmp_path / "live", fetch=flaky_fetch, today=date(2024, 1, 31))
    assert report.updated == {"AAA": 2}
    assert "429" in report.failed["BBB"]
    assert len(ParquetStore(tmp_path / "live").load("BBB")) == 5


def test_up_to_date_data_is_not_fetched(snapshot, tmp_path):
    last = ParquetStore(snapshot).load("AAA")["date"].max()

    def must_not_fetch(*args, **kwargs):
        raise AssertionError("fetch should not be called")

    report = refresh_copy(snapshot, tmp_path / "live", fetch=must_not_fetch, today=last)
    assert report.updated == {} and report.failed == {}


def test_newer_live_data_is_not_overwritten_by_snapshot(snapshot, tmp_path):
    live = tmp_path / "live"
    refresh_copy(snapshot, live, fetch=lambda s, start, **k: bars(s, start, 4),
                 today=date(2024, 1, 31))
    rows_after_first = len(ParquetStore(live).load("AAA"))

    refresh_copy(snapshot, live, fetch=lambda s, start, **k: bars(s, start, 0),
                 today=date(2024, 1, 31))
    assert len(ParquetStore(live).load("AAA")) == rows_after_first == 9


def test_live_settings_uses_refreshed_copy(tmp_path, monkeypatch):
    from paper_trading import live_data
    from paper_trading.service import SNAPSHOT_DIR

    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("QUANT_LIVE_REFRESH", "1")
    monkeypatch.setattr(live_data, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(
        live_data,
        "refresh_copy",
        lambda src, dst: refresh_copy(src, dst, fetch=lambda s, start, **k: bars(s, start, 0),
                                      today=date.today()),
    )
    live_data._refreshed.clear()
    cfg = live_data.live_settings()
    assert cfg.data_dir == tmp_path / "live" and cfg.public_demo
    assert {p.name for p in (tmp_path / "live").glob("*.parquet")} == {
        p.name for p in SNAPSHOT_DIR.glob("*.parquet")
    }
    assert live_data.refresh_status() is not None
    live_data._refreshed.clear()

    monkeypatch.setenv("QUANT_LIVE_REFRESH", "0")
    assert live_data.live_settings().data_dir == SNAPSHOT_DIR
