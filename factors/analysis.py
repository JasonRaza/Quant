"""Cross-sectional factor evaluation: IC, quantile spreads, composites."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Return from close at t to close at t + horizon (NaN at the tail)."""
    return close.shift(-horizon) / close - 1


def rebalance_dates(index: pd.Index, horizon: int = 21) -> pd.Index:
    """Every `horizon`-th date, so forward-return windows don't overlap.

    Overlapping windows would make consecutive IC observations correlated and
    inflate the t-stat.
    """
    return index[::horizon]


def cross_sectional_zscore(factor: pd.DataFrame) -> pd.DataFrame:
    mean = factor.mean(axis=1)
    std = factor.std(axis=1).replace(0.0, np.nan)
    return factor.sub(mean, axis=0).div(std, axis=0)


def information_coefficient(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    dates: pd.Index | None = None,
    min_obs: int = 3,
) -> pd.Series:
    """Spearman rank IC between factor and forward returns, one value per date."""
    dates = factor.index if dates is None else dates
    values = {}
    for dt in dates:
        pair = pd.concat([factor.loc[dt], fwd.loc[dt]], axis=1, keys=["f", "r"]).dropna()
        if len(pair) < min_obs or pair["f"].nunique() < 2 or pair["r"].nunique() < 2:
            continue
        values[dt] = pair["f"].rank().corr(pair["r"].rank())
    return pd.Series(values, name="ic", dtype=float)


def ic_summary(ic: pd.Series) -> dict:
    n = int(ic.count())
    if n == 0:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "hit_rate": 0.0, "n": 0}
    mean = float(ic.mean())
    std = float(ic.std()) if n > 1 else 0.0
    ir = mean / std if std > 1e-12 else 0.0
    return {
        "ic_mean": round(mean, 4),
        "ic_std": round(std, 4),
        "ic_ir": round(ir, 4),
        "t_stat": round(ir * np.sqrt(n), 4),
        "hit_rate": round(float((ic > 0).mean()), 4),
        "n": n,
    }


def quantile_returns(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    dates: pd.Index | None = None,
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """Mean forward return per factor quantile (1 = lowest score) per date."""
    dates = factor.index if dates is None else dates
    rows = {}
    for dt in dates:
        pair = pd.concat([factor.loc[dt], fwd.loc[dt]], axis=1, keys=["f", "r"]).dropna()
        if len(pair) < n_quantiles:
            continue
        buckets = pd.qcut(pair["f"].rank(method="first"), n_quantiles, labels=False) + 1
        rows[dt] = pair["r"].groupby(buckets).mean()
    out = pd.DataFrame(rows).T.sort_index()
    out.columns = [f"q{int(c)}" for c in out.columns]
    return out


def long_short_spread(qrets: pd.DataFrame) -> pd.Series:
    """Top-minus-bottom quantile return per rebalance period."""
    if qrets.empty:
        return pd.Series(dtype=float, name="long_short")
    return (qrets.iloc[:, -1] - qrets.iloc[:, 0]).rename("long_short")


def composite_score(factors: dict[str, pd.DataFrame], weights: dict[str, float] | None = None):
    """Weighted average of cross-sectional z-scores (equal weight by default)."""
    weights = weights or {name: 1.0 for name in factors}
    total = sum(abs(w) for w in weights.values())
    if total == 0:
        raise ValueError("weights must not all be zero")
    score = None
    for name, w in weights.items():
        z = cross_sectional_zscore(factors[name]) * (w / total)
        score = z if score is None else score.add(z, fill_value=0.0)
    return score
