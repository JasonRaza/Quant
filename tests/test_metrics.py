"""Tests for backtest metrics."""

import pandas as pd

from backtester.metrics import max_drawdown, sharpe_ratio, summarize


def test_max_drawdown():
    equity = pd.Series([100, 110, 105, 90, 95])
    dd = max_drawdown(equity)
    assert dd < 0
    assert round(dd, 4) == round((90 - 110) / 110, 4)


def test_sharpe_positive_trend():
    returns = pd.Series([0.01] * 100)
    assert sharpe_ratio(returns) == 0.0  # zero std


def test_summarize():
    equity = pd.Series([100_000, 101_000, 100_500, 102_000])
    trades = pd.DataFrame({"trade_value": [1000.0, -500.0]})
    m = summarize(equity, trades)
    assert "sharpe" in m
    assert m["final_equity"] == 102_000.0
