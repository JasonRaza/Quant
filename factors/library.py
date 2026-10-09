"""Price/volume factor definitions.

Every factor takes wide (date x symbol) panels and returns a wide frame of the
same shape. The value at date t only uses data up to and including t's close.
Sign convention: higher score = more attractive (expected to outperform).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from factors.panel import PricePanel

FactorFn = Callable[[PricePanel], pd.DataFrame]


def momentum_12_1(panel: PricePanel, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-month return excluding the most recent month (Jegadeesh-Titman)."""
    close = panel.close
    return close.shift(skip) / close.shift(lookback) - 1


def short_term_reversal(panel: PricePanel, window: int = 21) -> pd.DataFrame:
    """Negative 1-month return: recent losers tend to bounce."""
    return -(panel.close / panel.close.shift(window) - 1)


def low_volatility(panel: PricePanel, window: int = 63) -> pd.DataFrame:
    """Negative annualized volatility of daily returns."""
    rets = panel.close.pct_change(fill_method=None)
    return -(rets.rolling(window, min_periods=window).std() * np.sqrt(252))


def liquidity(panel: PricePanel, window: int = 63) -> pd.DataFrame:
    """Log of average daily dollar volume."""
    dollar_vol = panel.close * panel.volume
    avg = dollar_vol.rolling(window, min_periods=window).mean()
    return np.log(avg.where(avg > 0))


FACTORS: dict[str, FactorFn] = {
    "momentum_12_1": momentum_12_1,
    "reversal_1m": short_term_reversal,
    "low_vol": low_volatility,
    "liquidity": liquidity,
}


def compute_factors(panel: PricePanel, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    names = names or list(FACTORS)
    unknown = set(names) - set(FACTORS)
    if unknown:
        raise KeyError(f"Unknown factors: {', '.join(sorted(unknown))}")
    return {name: FACTORS[name](panel) for name in names}
