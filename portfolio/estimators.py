"""Expected-return and covariance estimators (annualized)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None).iloc[1:]


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Shrink the sample covariance toward a scaled identity (Ledoit & Wolf, 2004).

    The sample covariance is noisy when the window is short relative to the
    number of assets, and mean-variance optimizers amplify that noise.
    Shrinkage intensity is estimated from the data, with no tuning parameter.
    """
    x = returns.dropna().to_numpy()
    n, p = x.shape
    if n < 2:
        raise ValueError("need at least 2 observations")
    x = x - x.mean(axis=0)
    s = x.T @ x / n
    mu = np.trace(s) / p
    target = mu * np.eye(p)

    d2 = np.sum((s - target) ** 2) / p
    # sum_k ||x_k x_k' - S||_F^2 simplifies to sum_k ||x_k||^4 - n * ||S||_F^2
    b2_bar = (np.sum(np.sum(x**2, axis=1) ** 2) - n * np.sum(s**2)) / (n**2 * p)
    b2 = min(b2_bar, d2)
    shrinkage = b2 / d2 if d2 > 0 else 1.0

    cov = shrinkage * target + (1 - shrinkage) * s
    return pd.DataFrame(cov * TRADING_DAYS, index=returns.columns, columns=returns.columns)


def historical_mean(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * TRADING_DAYS


def factor_alpha(score: pd.Series, cov: pd.DataFrame, ic: float = 0.05) -> pd.Series:
    """Turn cross-sectional scores into expected returns via alpha = IC * vol * z.

    This is Grinold's "alpha = IC x volatility x score" rule. It scales a raw
    ranking (e.g. the factor composite from Project 3) into return units, so
    the optimizer doesn't treat a z-score of 1 as a 100% expected return.
    """
    score = score.reindex(cov.index)
    z = (score - score.mean()) / score.std(ddof=0) if score.std(ddof=0) > 0 else score * 0.0
    vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
    return (ic * vol * z.fillna(0.0)).rename("alpha")
