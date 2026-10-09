"""Cross-sectional feature/label dataset for return prediction.

One row per (date, symbol). Features at date t use data up to t's close only;
the label is the forward return from t to t + horizon, demeaned across symbols
so models learn relative performance (which name beats the others), not the
market's direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factors.library import compute_factors
from factors.panel import PricePanel


@dataclass
class Dataset:
    X: pd.DataFrame  # MultiIndex (date, symbol)
    y: pd.Series  # demeaned forward return
    label_end: pd.Series  # date the label's forward window closes (for purging)

    @property
    def dates(self) -> pd.Index:
        return self.X.index.get_level_values("date")


def raw_features(panel: PricePanel) -> dict[str, pd.DataFrame]:
    close, volume = panel.close, panel.volume
    rets = close.pct_change(fill_method=None)
    # 1-month return and 63-day volatility are already covered by the Project 3
    # reversal and low-vol factors (same ranks, opposite sign), so they're omitted.
    feats = {
        "ret_5d": close / close.shift(5) - 1,
        "ret_63d": close / close.shift(63) - 1,
        "ret_126d": close / close.shift(126) - 1,
        "vol_21d": rets.rolling(21, min_periods=21).std() * np.sqrt(252),
        "dist_ma50": close / close.rolling(50, min_periods=50).mean() - 1,
        "dist_ma200": close / close.rolling(200, min_periods=200).mean() - 1,
        "volume_z": (volume - volume.rolling(63).mean()) / volume.rolling(63).std(),
    }
    feats.update(compute_factors(panel))  # Project 3 factors
    return feats


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Map each date's values to [-0.5, 0.5] by rank: robust to outliers and regime scale."""
    ranks = frame.rank(axis=1, pct=True)
    counts = frame.notna().sum(axis=1)
    # pct ranks run from 1/n to 1; center them so the middle name sits at 0.
    return ranks.sub(0.5 + 0.5 / counts.replace(0, np.nan), axis=0)


def build_dataset(panel: PricePanel, horizon: int = 21, sample_every: int = 5) -> Dataset:
    """Stack features and labels, sampling every `sample_every` trading days."""
    feats = {name: cross_sectional_rank(f) for name, f in raw_features(panel).items()}

    fwd = panel.close.shift(-horizon) / panel.close - 1
    label = fwd.sub(fwd.mean(axis=1), axis=0)

    index = panel.close.index
    end_pos = np.minimum(np.arange(len(index)) + horizon, len(index) - 1)
    label_end = pd.Series(index[end_pos], index=index)

    sample_dates = index[::sample_every]
    stacked = pd.concat(
        {name: f.loc[sample_dates].stack(future_stack=True) for name, f in feats.items()}, axis=1
    )
    stacked.index.names = ["date", "symbol"]
    y = label.loc[sample_dates].stack(future_stack=True).rename("target")
    y.index.names = ["date", "symbol"]

    data = stacked.join(y, how="inner").dropna()
    X, y = data.drop(columns="target"), data["target"]
    ends = label_end.reindex(X.index.get_level_values("date"))
    return Dataset(X=X, y=y, label_end=pd.Series(ends.to_numpy(), index=X.index, name="label_end"))


def latest_features(panel: PricePanel) -> pd.DataFrame:
    """Feature rows for the most recent date (no label yet), indexed by symbol."""
    feats = {name: cross_sectional_rank(f) for name, f in raw_features(panel).items()}
    last = panel.close.index[-1]
    return pd.DataFrame({name: f.loc[last] for name, f in feats.items()}).dropna()
