"""Tests for portfolio estimators and optimizers."""

import numpy as np
import pandas as pd
import pytest

from portfolio.estimators import (
    factor_alpha,
    historical_mean,
    ledoit_wolf_covariance,
    sample_covariance,
)
from portfolio.optimizer import (
    Constraints,
    equal_weight,
    max_sharpe,
    mean_variance,
    min_variance,
    risk_contributions,
    risk_parity,
)
from portfolio.walkforward import WalkForwardConfig, default_allocators, run_walkforward

ASSETS = ["A", "B", "C", "D"]


def diag_cov(vols=(0.1, 0.2, 0.3, 0.4)) -> pd.DataFrame:
    return pd.DataFrame(np.diag(np.square(vols)), index=ASSETS, columns=ASSETS)


def random_returns(n=500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (n, len(ASSETS))), columns=ASSETS)


def assert_valid(w: pd.Series, max_weight: float = 1.0):
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert (w <= max_weight + 1e-9).all()


def test_ledoit_wolf_is_between_sample_and_target():
    rets = random_returns(n=60)
    lw = ledoit_wolf_covariance(rets).to_numpy()
    sample = sample_covariance(rets).to_numpy() * (len(rets) - 1) / len(rets)
    off = ~np.eye(len(ASSETS), dtype=bool)
    assert np.all(np.abs(lw[off]) <= np.abs(sample[off]) + 1e-12)  # off-diagonals shrunk
    assert np.trace(lw) == pytest.approx(np.trace(sample))  # total variance preserved
    assert np.all(np.linalg.eigvalsh(lw) > 0)


def test_ledoit_wolf_matches_sklearn_if_available():
    sk = pytest.importorskip("sklearn.covariance")
    rets = random_returns(n=80, seed=3)
    ours = ledoit_wolf_covariance(rets).to_numpy() / 252
    ref = sk.LedoitWolf().fit(rets.to_numpy()).covariance_
    np.testing.assert_allclose(ours, ref, rtol=1e-8)


def test_min_variance_diagonal_closed_form():
    cov = diag_cov()
    w = min_variance(cov, Constraints(max_weight=1.0))
    inv = 1 / np.diag(cov.to_numpy())
    np.testing.assert_allclose(w.to_numpy(), inv / inv.sum(), atol=1e-5)
    assert_valid(w)


def test_max_weight_cap_binds():
    w = min_variance(diag_cov(), Constraints(max_weight=0.3))
    assert_valid(w, 0.3)
    assert w["A"] == pytest.approx(0.3, abs=1e-6)


def test_infeasible_cap_raises():
    with pytest.raises(ValueError):
        min_variance(diag_cov(), Constraints(max_weight=0.2))


def test_mean_variance_tilts_to_higher_return():
    cov = pd.DataFrame(np.eye(4) * 0.04, index=ASSETS, columns=ASSETS)
    mu = pd.Series([0.02, 0.05, 0.08, 0.11], index=ASSETS)
    w = mean_variance(mu, cov, risk_aversion=5.0, constraints=Constraints(max_weight=1.0))
    assert_valid(w)
    assert list(w.sort_values().index) == ASSETS


def test_max_sharpe_closed_form_diagonal():
    """Unconstrained tangency with diagonal cov: w ∝ mu / var."""
    cov = diag_cov()
    mu = pd.Series([0.05, 0.08, 0.10, 0.12], index=ASSETS)
    w = max_sharpe(mu, cov, constraints=Constraints(max_weight=1.0))
    target = mu / np.diag(cov.to_numpy())
    np.testing.assert_allclose(w.to_numpy(), (target / target.sum()).to_numpy(), atol=1e-4)


def test_risk_parity_equalizes_contributions():
    rets = random_returns()
    cov = ledoit_wolf_covariance(rets * np.array([1, 2, 3, 4]))
    w = risk_parity(cov)
    assert_valid(w)
    np.testing.assert_allclose(risk_contributions(w, cov).to_numpy(), 0.25, atol=1e-3)


def test_equal_weight():
    assert equal_weight(diag_cov()).tolist() == [0.25] * 4


def test_factor_alpha_scaling():
    cov = diag_cov()
    score = pd.Series([1.0, 1.0, -1.0, -1.0], index=ASSETS)
    alpha = factor_alpha(score, cov, ic=0.1)
    np.testing.assert_allclose(alpha.to_numpy(), [0.01, 0.02, -0.03, -0.04])
    assert historical_mean(random_returns()).shape == (4,)


def walkforward_returns(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (n, len(ASSETS))), index=idx, columns=ASSETS)


def test_walkforward_equal_weight_no_costs_matches_manual():
    rets = walkforward_returns()
    cfg = WalkForwardConfig(lookback=100, rebalance_every=1, cost_bps=0.0, initial_capital=1.0)
    res = run_walkforward(rets, lambda win, dt: pd.Series(0.25, index=win.columns), cfg)
    # Daily rebalanced equal weight = compounding the cross-sectional mean return.
    expected = (1 + rets.iloc[100:].mean(axis=1)).prod()
    assert res.equity.iloc[-1] == pytest.approx(expected)
    assert res.equity.index[0] == rets.index[99]


def test_walkforward_allocator_never_sees_future():
    rets = walkforward_returns()
    seen = []

    def spy(window, dt):
        seen.append((window.index[-1], dt, len(window)))
        return pd.Series(0.25, index=window.columns)

    run_walkforward(rets, spy, WalkForwardConfig(lookback=100, rebalance_every=21))
    assert seen and all(last == dt and n == 100 for last, dt, n in seen)


def test_walkforward_costs_reduce_equity():
    rets = walkforward_returns()

    def flipper(window, dt):
        """Alternate 100% A / 100% B each rebalance: turnover of 2 per trade."""
        held = "A" if (rets.index.get_loc(dt) // 21) % 2 == 0 else "B"
        return pd.Series([1.0 if a == held else 0.0 for a in ASSETS], index=ASSETS)

    free = run_walkforward(rets, flipper, WalkForwardConfig(lookback=100, cost_bps=0.0))
    costly = run_walkforward(rets, flipper, WalkForwardConfig(lookback=100, cost_bps=50.0))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.metrics["total_costs"] > 0
    assert costly.metrics["avg_turnover"] == pytest.approx(2.0)


def test_default_allocators_run_and_respect_cap():
    rets = walkforward_returns()
    scores = pd.DataFrame(
        np.tile([2.0, 1.0, -1.0, -2.0], (len(rets), 1)), index=rets.index, columns=ASSETS
    )
    cons = Constraints(max_weight=0.4)
    for name, alloc in default_allocators(cons, scores).items():
        w = run_walkforward(rets, alloc, WalkForwardConfig(lookback=100)).weights
        assert np.allclose(w.sum(axis=1), 1.0), name
        if name != "risk_parity":
            assert (w <= 0.4 + 1e-6).all().all(), name

    factor_mv = default_allocators(cons, scores)["factor_mean_variance"]
    tilt = run_walkforward(rets, factor_mv, WalkForwardConfig(lookback=100))
    assert (tilt.weights["A"] > tilt.weights["D"]).all()
