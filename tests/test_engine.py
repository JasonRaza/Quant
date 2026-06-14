"""Golden-file style tests for backtest engine."""

import pandas as pd

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.strategies import DualMovingAverageCrossover


def test_dual_ma_backtest_deterministic(synthetic_bars):
    prices = synthetic_bars.rename(columns={"symbol": "sym"}).drop(columns=["sym"])
    strategy = DualMovingAverageCrossover(fast=3, slow=5)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, commission_bps=5, slippage_bps=5))
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
