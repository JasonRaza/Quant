"""Tests for OHLCV normalization."""

import pandas as pd

from shared.types import normalize_ohlcv


def test_normalize_yfinance_format():
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1e6, 1.1e6],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    out = normalize_ohlcv(raw, "SPY")
    assert list(out.columns) == ["symbol", "date", "open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert out["symbol"].iloc[0] == "SPY"
    assert out["close"].iloc[1] == 102.0


def test_normalize_empty():
    out = normalize_ohlcv(pd.DataFrame(), "X")
    assert out.empty
