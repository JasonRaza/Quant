"""Glue between the research modules and the paper account (used by API and dashboard)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_pipeline.storage import ParquetStore
from factors.analysis import composite_score
from factors.library import compute_factors
from factors.panel import build_panel
from paper_trading.broker import PaperBroker, ParquetPriceSource
from portfolio.estimators import daily_returns
from portfolio.optimizer import Constraints
from portfolio.walkforward import default_allocators, historical_max_sharpe
from shared.config import load_yaml, resolve_path

ALLOCATION_METHODS = [
    "equal_weight",
    "min_variance",
    "risk_parity",
    "factor_mean_variance",
    "max_sharpe_historical",
]


# Committed copy of the market data, used when no local data has been fetched
# (e.g. on a fresh clone or a hosted deployment). Refresh with `make snapshot`.
SNAPSHOT_DIR = resolve_path("data/snapshot")


@dataclass
class Settings:
    data_dir: Path
    paper_db: Path
    universe: list[str]
    benchmark: str = "SPY"
    # Public demo: every visitor gets a private, temporary paper account instead of
    # sharing one. On when QUANT_PUBLIC_DEMO=1 or when running on the bundled snapshot.
    public_demo: bool = False

    @classmethod
    def from_config(cls, config_path: str = "config/watchlist.yaml") -> "Settings":
        """Load from the watchlist config; QUANT_DATA_DIR / QUANT_PAPER_DB override paths."""
        cfg = load_yaml(resolve_path(config_path))
        benchmark = cfg.get("benchmark", "SPY")
        data_dir = os.environ.get("QUANT_DATA_DIR") or cfg.get("data_dir", "data/parquet")
        paper_db = os.environ.get("QUANT_PAPER_DB") or cfg.get(
            "paper_db", "data/paper_trading.sqlite"
        )
        data_path = resolve_path(data_dir)
        on_snapshot = False
        if not any(data_path.glob("*.parquet")) and any(SNAPSHOT_DIR.glob("*.parquet")):
            data_path, on_snapshot = SNAPSHOT_DIR, True
        return cls(
            data_dir=data_path,
            paper_db=resolve_path(paper_db),
            universe=[s.upper() for s in cfg.get("symbols", []) if s.upper() != benchmark],
            benchmark=benchmark,
            public_demo=on_snapshot or os.environ.get("QUANT_PUBLIC_DEMO") == "1",
        )


def make_broker(settings: Settings) -> PaperBroker:
    return PaperBroker(settings.paper_db, ParquetPriceSource(settings.data_dir))


def target_weights(
    settings: Settings,
    method: str,
    max_weight: float = 0.25,
    lookback: int = 252,
) -> pd.Series:
    """Weights the chosen allocator would hold today, from the trailing window."""
    if method not in ALLOCATION_METHODS:
        raise ValueError(f"unknown method '{method}'; choose from {', '.join(ALLOCATION_METHODS)}")

    panel = build_panel(ParquetStore(settings.data_dir), settings.universe)
    returns = daily_returns(panel.close)
    window = returns.iloc[-lookback:].dropna(axis=1)
    constraints = Constraints(max_weight=max_weight)

    if method == "max_sharpe_historical":
        allocator = historical_max_sharpe(constraints)
    else:
        scores = None
        if method == "factor_mean_variance":
            scores = composite_score(compute_factors(panel))
        allocator = default_allocators(constraints, scores)[method]
    return allocator(window, returns.index[-1]).sort_values(ascending=False)
