"""Model zoo and out-of-sample evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from factors.analysis import ic_summary


class SingleFeatureBaseline(BaseEstimator, RegressorMixin):
    """Predict with one feature's rank (no fitting). The bar any model must clear."""

    def __init__(self, feature: str = "momentum_12_1"):
        self.feature = feature

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.feature].to_numpy()


def default_models(seed: int = 0) -> dict[str, BaseEstimator]:
    return {
        "momentum_baseline": SingleFeatureBaseline("momentum_12_1"),
        "ridge": Ridge(alpha=10.0),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


@dataclass
class CVResult:
    predictions: pd.Series
    ic: pd.Series
    summary: dict
    long_short: pd.Series = field(default_factory=pd.Series)


def daily_rank_ic(pred: pd.Series, y: pd.Series, min_obs: int = 3) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    out = {}
    for dt, g in df.groupby(level="date"):
        if len(g) >= min_obs and g["p"].nunique() > 1:
            out[dt] = g["p"].rank().corr(g["y"].rank())
    return pd.Series(out, name="ic", dtype=float)


def top_minus_bottom(pred: pd.Series, y: pd.Series, n_quantiles: int = 3) -> pd.Series:
    """Per-date return of the top prediction quantile minus the bottom one."""
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    out = {}
    for dt, g in df.groupby(level="date"):
        if len(g) < n_quantiles:
            continue
        q = pd.qcut(g["p"].rank(method="first"), n_quantiles, labels=False)
        out[dt] = g["y"][q == n_quantiles - 1].mean() - g["y"][q == 0].mean()
    return pd.Series(out, name="long_short", dtype=float)


def cross_validate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    overlap: int = 1,
) -> CVResult:
    """Fit a fresh clone per split and collect out-of-sample predictions.

    `overlap` = label horizon / sampling step. Consecutive ICs share most of
    their forward window, so the t-stat is computed on every `overlap`-th IC
    only; using all of them would overstate significance about sqrt(overlap)x.
    """
    preds = []
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.append(pd.Series(m.predict(X.iloc[test_idx]), index=X.index[test_idx]))
    pred = pd.concat(preds).sort_index() if preds else pd.Series(dtype=float)
    ic = daily_rank_ic(pred, y)
    return CVResult(
        predictions=pred,
        ic=ic,
        summary=ic_summary(ic.iloc[:: max(overlap, 1)]) | {"ic_mean_all_dates": _mean(ic)},
        long_short=top_minus_bottom(pred, y),
    )


def _mean(s: pd.Series) -> float:
    return round(float(s.mean()), 4) if len(s) else 0.0


def ridge_coefficients(model: Ridge, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    fitted = clone(model).fit(X, y)
    return pd.Series(fitted.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)

