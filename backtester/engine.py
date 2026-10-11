"""Event-driven backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    metrics: dict[str, Any]
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
        prev_signal = 0.0
        equity_vals = []
        position_vals = []
        trade_rows = []

        slippage = self.config.slippage_bps / 10_000
        commission = self.config.commission_bps / 10_000

        for dt in df.index:
            price = float(close.loc[dt])
            target_signal = float(signals.loc[dt])

            # Trade only when the signal changes; otherwise the position is held.
            if target_signal != prev_signal and price > 0:
                equity = cash + shares * price
                if target_signal > 0:
                    # Size longs net of slippage and commission so cash doesn't go negative.
                    all_in_price = price * (1 + slippage) * (1 + commission)
                    target_shares = equity * target_signal / all_in_price
                else:
                    target_shares = equity * target_signal / price
                delta_shares = target_shares - shares

                if abs(delta_shares) > 1e-9:
                    # Slippage worsens the fill price; commission is charged once on notional.
                    fill_price = price * (1 + slippage if delta_shares > 0 else 1 - slippage)
                    trade_value = delta_shares * fill_price
                    fee = abs(trade_value) * commission
                    cash -= trade_value + fee
                    shares += delta_shares
                    trade_rows.append(
                        {
                            "date": dt,
                            "shares": delta_shares,
                            "price": fill_price,
                            "trade_value": trade_value,
                            "cost": fee + abs(delta_shares) * price * slippage,
                        }
                    )
            prev_signal = target_signal

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
