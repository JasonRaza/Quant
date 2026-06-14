"""Event-driven backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtester.metrics import summarize
from backtester.strategies import Strategy


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict
    benchmark_equity: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        price_col: str = "close",
    ) -> BacktestResult:
        if prices.empty:
            raise ValueError("prices DataFrame is empty")

        df = prices.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        df = df.sort_index()

        signals = strategy.generate_signals(df)
        close = df[price_col].astype(float)

        cash = self.config.initial_capital
        shares = 0.0
        equity_vals = []
        position_vals = []
        trade_rows = []

        cost_rate = (self.config.commission_bps + self.config.slippage_bps) / 10_000

        for dt in df.index:
            price = float(close.loc[dt])
            target_signal = float(signals.loc[dt])
            equity = cash + shares * price
            target_value = equity * target_signal
            target_shares = target_value / price if price > 0 else 0.0
            delta_shares = target_shares - shares

            if abs(delta_shares) > 1e-9:
                trade_price = price * (1 + cost_rate * (1 if delta_shares > 0 else -1))
                trade_value = delta_shares * trade_price
                cost = abs(trade_value) * cost_rate
                cash -= trade_value + cost
                shares += delta_shares
                trade_rows.append(
                    {
                        "date": dt,
                        "shares": delta_shares,
                        "price": trade_price,
                        "trade_value": trade_value,
                        "cost": cost,
                    }
                )

            equity = cash + shares * price
            equity_vals.append(equity)
            position_vals.append(shares)

        equity_series = pd.Series(equity_vals, index=df.index, name="equity")
        position_series = pd.Series(position_vals, index=df.index, name="position")
        trades_df = pd.DataFrame(trade_rows)

        # Buy-and-hold benchmark
        first_price = float(close.iloc[0])
        bench_shares = self.config.initial_capital / first_price
        benchmark = bench_shares * close

        metrics = summarize(equity_series, trades_df)
        bench_metrics = summarize(benchmark, pd.DataFrame())
        metrics["benchmark_total_return"] = bench_metrics["total_return"]
        metrics["benchmark_sharpe"] = bench_metrics["sharpe"]
        metrics["benchmark_max_drawdown"] = bench_metrics["max_drawdown"]

        return BacktestResult(
            equity=equity_series,
            positions=position_series,
            trades=trades_df,
            metrics=metrics,
            benchmark_equity=benchmark,
        )
