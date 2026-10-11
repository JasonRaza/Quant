"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    std = returns.std()
    if std is None or std < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def win_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    if active.empty:
        return 0.0
    return float((active > 0).mean())


def turnover(trades: pd.DataFrame) -> float:
    if trades.empty or "trade_value" not in trades.columns:
        return 0.0
    return float(trades["trade_value"].abs().sum())


def summarize(
    equity: pd.Series, trades: pd.DataFrame, periods_per_year: int = 252
) -> dict[str, float]:
    rets = compute_returns(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    return {
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe_ratio(rets, periods_per_year), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "win_rate": round(win_rate(rets), 4),
        "turnover": round(turnover(trades), 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }
