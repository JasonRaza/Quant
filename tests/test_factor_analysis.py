"""Tests for factor IC, quantile and composite analysis."""

import numpy as np
import pandas as pd
import pytest

from factors.analysis import (
    composite_score,
    cross_sectional_zscore,
    forward_returns,
    ic_summary,
    information_coefficient,
    long_short_spread,
    quantile_returns,
    rebalance_dates,
)

DATES = pd.bdate_range("2024-01-01", periods=6)
SYMS = ["A", "B", "C", "D", "E", "F"]


def perfect_factor():
    """Forward returns are a monotonic function of the factor."""
    factor = pd.DataFrame([[1, 2, 3, 4, 5, 6]] * len(DATES), index=DATES, columns=SYMS, dtype=float)
    fwd = factor * 0.01
    return factor, fwd


def test_forward_returns_alignment():
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    fwd = forward_returns(close, horizon=1)
    assert fwd["A"].iloc[0] == pytest.approx(0.10)
    assert np.isnan(fwd["A"].iloc[-1])


def test_rebalance_dates_non_overlapping():
    idx = pd.bdate_range("2024-01-01", periods=50)
    reb = rebalance_dates(idx, horizon=21)
    assert list(reb) == [idx[0], idx[21], idx[42]]


def test_ic_perfect_and_inverse():
    factor, fwd = perfect_factor()
    assert information_coefficient(factor, fwd).tolist() == pytest.approx([1.0] * len(DATES))
    assert information_coefficient(-factor, fwd).tolist() == pytest.approx([-1.0] * len(DATES))


def test_ic_skips_thin_dates():
    factor, fwd = perfect_factor()
    factor.iloc[0, 1:] = np.nan  # only one name on the first date
    ic = information_coefficient(factor, fwd)
    assert DATES[0] not in ic.index
    assert len(ic) == len(DATES) - 1


def test_ic_summary():
    s = ic_summary(pd.Series([0.1, 0.2, -0.1, 0.2]))
    assert s["n"] == 4
    assert s["hit_rate"] == 0.75
    assert s["t_stat"] == pytest.approx(s["ic_ir"] * 2, abs=1e-3)
    assert ic_summary(pd.Series(dtype=float))["n"] == 0


def test_quantile_returns_monotonic_spread():
    factor, fwd = perfect_factor()
    q = quantile_returns(factor, fwd, n_quantiles=3)
    assert list(q.columns) == ["q1", "q2", "q3"]
    assert (q["q3"] > q["q2"]).all() and (q["q2"] > q["q1"]).all()
    assert long_short_spread(q).iloc[0] == pytest.approx(0.055 - 0.015)


def test_zscore_and_composite():
    factor, _ = perfect_factor()
    z = cross_sectional_zscore(factor)
    assert z.mean(axis=1).abs().max() < 1e-12
    comp = composite_score({"a": factor, "b": -factor})
    assert comp.abs().max().max() < 1e-12  # opposite factors cancel
    with pytest.raises(ValueError):
        composite_score({"a": factor}, {"a": 0.0})
