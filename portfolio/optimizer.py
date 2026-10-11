"""Constrained portfolio construction: min-variance, mean-variance, max-Sharpe, risk parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Constraints:
    """Fully invested (weights sum to 1), long-only, per-asset weight cap."""

    max_weight: float = 0.35
    min_weight: float = 0.0

    def bounds(self, n: int) -> list[tuple[float, float]]:
        if self.max_weight * n < 1 - 1e-9:
            raise ValueError(f"max_weight={self.max_weight} can't be fully invested in {n} assets")
        return [(self.min_weight, self.max_weight)] * n


def _solve(
    objective: Callable[[np.ndarray], float],
    n: int,
    constraints: Constraints,
    extra: Sequence[dict[str, Any]] = (),
) -> np.ndarray:
    x0 = np.full(n, 1.0 / n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra]
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=constraints.bounds(n),
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"optimizer failed: {res.message}")
    w: np.ndarray = np.clip(res.x, constraints.min_weight, constraints.max_weight)
    weights: np.ndarray = w / w.sum()
    return weights


def equal_weight(cov: pd.DataFrame) -> pd.Series:
    n = len(cov)
    return pd.Series(1.0 / n, index=cov.index, name="weight")


def min_variance(cov: pd.DataFrame, constraints: Constraints | None = None) -> pd.Series:
    c = cov.to_numpy()
    w = _solve(lambda w: w @ c @ w, len(c), constraints or Constraints())
    return pd.Series(w, index=cov.index, name="weight")


def mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 5.0,
    constraints: Constraints | None = None,
) -> pd.Series:
    """Maximize mu'w - (risk_aversion / 2) * w'Σw."""
    m, c = mu.reindex(cov.index).to_numpy(), cov.to_numpy()
    w = _solve(
        lambda w: -(m @ w) + 0.5 * risk_aversion * (w @ c @ w),
        len(c),
        constraints or Constraints(),
    )
    return pd.Series(w, index=cov.index, name="weight")


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
    constraints: Constraints | None = None,
) -> pd.Series:
    m, c = mu.reindex(cov.index).to_numpy() - risk_free, cov.to_numpy()

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ c @ w, 1e-16))
        return float(-(m @ w) / vol)

    w = _solve(neg_sharpe, len(c), constraints or Constraints())
    return pd.Series(w, index=cov.index, name="weight")


def risk_parity(cov: pd.DataFrame, constraints: Constraints | None = None) -> pd.Series:
    """Equal risk contribution: each asset contributes the same share of variance."""
    c = cov.to_numpy()
    n = len(c)

    def objective(w: np.ndarray) -> float:
        port_var = w @ c @ w
        contrib = w * (c @ w) / port_var
        return float(np.sum((contrib - 1.0 / n) ** 2))

    w = _solve(objective, n, constraints or Constraints(max_weight=1.0))
    return pd.Series(w, index=cov.index, name="weight")


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    w, c = weights.reindex(cov.index).to_numpy(), cov.to_numpy()
    port_var = w @ c @ w
    return pd.Series(w * (c @ w) / port_var, index=cov.index, name="risk_contribution")


def portfolio_stats(
    weights: pd.Series, mu: pd.Series, cov: pd.DataFrame
) -> dict[str, float]:
    w = weights.reindex(cov.index).to_numpy()
    ret = float(mu.reindex(cov.index).to_numpy() @ w)
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    return {
        "expected_return": round(ret, 4),
        "volatility": round(vol, 4),
        "sharpe": round(ret / vol, 4) if vol > 0 else 0.0,
    }
