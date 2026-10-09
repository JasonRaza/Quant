"""Golden-file style tests for backtest engine."""

import numpy as np
import pandas as pd
import pytest

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.strategies import DualMovingAverageCrossover, MeanReversionPair, pair_spread_prices


def test_dual_ma_backtest_deterministic(synthetic_bars):
    prices = synthetic_bars.rename(columns={"symbol": "sym"}).drop(columns=["sym"])
    strategy = DualMovingAverageCrossover(fast=3, slow=5)
    config = BacktestConfig(initial_capital=10_000, commission_bps=5, slippage_bps=5)
    engine = BacktestEngine(config)
    result = engine.run(prices, strategy)

    assert len(result.equity) == len(prices)
    assert result.metrics["final_equity"] > 0
    assert "benchmark_sharpe" in result.metrics
    # Signal shifted by 1 — no trades on first bar
    assert result.equity.iloc[0] == 10_000


def test_no_lookahead_signal_shift():
    """Fast MA crossover should not use same-bar close for execution."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="B"),
            "close": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13],
        }
    )
    strategy = DualMovingAverageCrossover(fast=2, slow=3)
    signals = strategy.generate_signals(df.set_index("date"))
    assert signals.iloc[0] == 0.0


def test_pair_spread_prices_track_long_short_return():
    pair = pd.DataFrame({"close_a": [100.0, 110.0, 99.0], "close_b": [50.0, 50.0, 55.0]})
    out = pair_spread_prices(pair)
    # Day 1: A +10%, B 0% -> +10%. Day 2: A -10%, B +10% -> -20%.
    assert out["close"].tolist() == pytest.approx([100.0, 110.0, 88.0])


def test_mean_reversion_pair_backtest_runs():
    rng = np.random.default_rng(0)
    n = 300
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    noise = np.cumsum(rng.normal(0, 0.5, n))
    noise -= pd.Series(noise).rolling(30, min_periods=1).mean().to_numpy()  # mean-reverting
    a = b + 5 + noise
    prices = pair_spread_prices(
        pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=n), "close_a": a, "close_b": b})
    )
    result = BacktestEngine().run(prices, MeanReversionPair())

    assert len(result.trades) > 0
    assert set(np.sign(result.positions.unique())) <= {-1.0, 0.0, 1.0}
    assert np.isfinite(result.metrics["final_equity"])
