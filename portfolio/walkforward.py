"""Walk-forward backtest of portfolio construction methods.

At each rebalance date t, an allocator sees only returns up to t's close,
trades to its target weights at t's close (paying costs on turnover), and the
weights then drift with prices until the next rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtester.metrics import summarize
from portfolio.estimators import factor_alpha, ledoit_wolf_covariance
from portfolio.optimizer import (
    Constraints,
    equal_weight,
    max_sharpe,
    mean_variance,
    min_variance,
    risk_parity,
)

Allocator = Callable[[pd.DataFrame, pd.Timestamp], pd.Series]


@dataclass
class WalkForwardConfig:
    lookback: int = 252
    rebalance_every: int = 21
    cost_bps: float = 10.0
    initial_capital: float = 100_000.0


@dataclass
class WalkForwardResult:
    equity: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def run_walkforward(
    returns: pd.DataFrame,
    allocator: Allocator,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    cfg = config or WalkForwardConfig()
    if len(returns) <= cfg.lookback:
        raise ValueError(f"need more than lookback={cfg.lookback} rows of returns")

    rebal_dates = set(returns.index[cfg.lookback - 1 :: cfg.rebalance_every])
    start = cfg.lookback - 1
    cost_rate = cfg.cost_bps / 10_000

    equity = cfg.initial_capital
    w = pd.Series(0.0, index=returns.columns)
    equity_vals, weight_rows, trade_rows = [], {}, []

    for i in range(start, len(returns)):
        dt = returns.index[i]
        if i > start:
            r = returns.iloc[i].fillna(0.0)
            growth = (w * (1 + r)).sum()
            equity *= growth if w.sum() > 0 else 1.0
            if growth > 0:
                w = w * (1 + r) / growth

        if dt in rebal_dates:
            window = returns.iloc[i - cfg.lookback + 1 : i + 1].dropna(axis=1)
            target = allocator(window, dt).reindex(returns.columns).fillna(0.0)
            turnover = float((target - w).abs().sum())
            cost = turnover * cost_rate * equity
            equity -= cost
            trade_rows.append(
                {"date": dt, "turnover": turnover, "trade_value": turnover * equity, "cost": cost}
            )
            w = target
            weight_rows[dt] = target

        equity_vals.append(equity)

    equity_series = pd.Series(equity_vals, index=returns.index[start:], name="equity")
    trades = pd.DataFrame(trade_rows)
    metrics = summarize(equity_series, trades)
    # Skip the initial cash -> portfolio trade so avg turnover reflects rebalancing.
    rebal_turnover = trades["turnover"].iloc[1:]
    metrics["avg_turnover"] = round(float(rebal_turnover.mean()), 4) if len(rebal_turnover) else 0.0
    metrics["total_costs"] = round(float(trades["cost"].sum()), 2)
    return WalkForwardResult(
        equity=equity_series,
        weights=pd.DataFrame(weight_rows).T,
        trades=trades,
        metrics=metrics,
    )


def default_allocators(
    constraints: Constraints,
    scores: pd.DataFrame | None = None,
    ic: float = 0.05,
    risk_aversion: float = 5.0,
) -> dict[str, Allocator]:
    """Allocators compared in the report.

    `scores` (date x symbol) feeds the factor-tilted mean-variance allocator;
    the row at the rebalance date only uses data up to that date.
    """

    def ew(window, dt):
        return equal_weight(window.cov())

    def minvar(window, dt):
        return min_variance(ledoit_wolf_covariance(window), constraints)

    def erc(window, dt):
        return risk_parity(ledoit_wolf_covariance(window))

    allocators: dict[str, Allocator] = {
        "equal_weight": ew,
        "min_variance": minvar,
        "risk_parity": erc,
    }

    if scores is not None:

        def factor_mv(window, dt):
            cov = ledoit_wolf_covariance(window)
            row = scores.loc[:dt].iloc[-1] if not scores.loc[:dt].empty else pd.Series(dtype=float)
            if row.dropna().empty:
                return min_variance(cov, constraints)
            mu = factor_alpha(row, cov, ic=ic)
            return mean_variance(mu, cov, risk_aversion, constraints)

        allocators["factor_mean_variance"] = factor_mv

    return allocators


def historical_max_sharpe(constraints: Constraints) -> Allocator:
    """Max-Sharpe on trailing mean returns, the classic error-maximizing setup.

    Kept as a cautionary baseline: trailing means are poor return forecasts and
    the optimizer concentrates in whatever just went up.
    """

    def allocator(window, dt):
        cov = ledoit_wolf_covariance(window)
        mu = window.mean() * 252
        if (mu <= 0).all():
            return min_variance(cov, constraints)
        return max_sharpe(mu, cov, constraints=constraints)

    return allocator


def weights_concentration(weights: pd.DataFrame) -> float:
    """Average effective number of holdings (1 / sum w^2)."""
    if weights.empty:
        return 0.0
    return float((1 / np.square(weights).sum(axis=1)).mean())
