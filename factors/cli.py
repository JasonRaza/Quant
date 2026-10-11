"""CLI for factor research: IC report, quantile spreads and latest screen."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.storage import ParquetStore
from factors.analysis import (
    composite_score,
    forward_returns,
    ic_summary,
    information_coefficient,
    long_short_spread,
    quantile_returns,
    rebalance_dates,
)
from factors.library import FACTORS, compute_factors
from factors.panel import build_panel
from shared.config import load_yaml, resolve_path

logger = logging.getLogger(__name__)


def evaluate_factors(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    fwd = forward_returns(close, horizon)
    dates = rebalance_dates(close.index, horizon)
    periods_per_year = 252 / horizon

    report: dict[str, Any] = {}
    ic_series: dict[str, pd.Series] = {}
    for name, factor in factors.items():
        ic = information_coefficient(factor, fwd, dates)
        qrets = quantile_returns(factor, fwd, dates, n_quantiles)
        spread = long_short_spread(qrets)
        ic_series[name] = ic
        report[name] = {
            **ic_summary(ic),
            "quantile_mean_return": {c: round(float(v), 4) for c, v in qrets.mean().items()},
            "long_short_ann_return": round(float(spread.mean() * periods_per_year), 4)
            if not spread.empty
            else 0.0,
        }
    return report, ic_series


def latest_screen(factors: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Rank symbols on the most recent date with a full composite score."""
    score = composite_score(factors)
    valid = score.dropna(how="all")
    if valid.empty:
        return {"as_of": None, "rankings": []}
    dt = valid.index[-1]
    row = valid.loc[dt].dropna().sort_values(ascending=False)
    rankings = [
        {
            "rank": i + 1,
            "symbol": sym,
            "composite": round(float(val), 4),
            **{name: _round(factors[name].loc[dt, sym]) for name in factors},
        }
        for i, (sym, val) in enumerate(row.items())
    ]
    return {"as_of": str(dt.date()), "rankings": rankings}


def _round(x: Any) -> float | None:
    return None if pd.isna(x) else round(float(x), 4)


def save_ic_chart(ic_series: dict[str, pd.Series], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, ic in ic_series.items():
        if not ic.empty:
            ic.cumsum().plot(ax=ax, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative rank IC by factor")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Cumulative IC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional factor research")
    parser.add_argument("--config", default="config/watchlist.yaml")
    parser.add_argument("--symbols", nargs="+", help="Override the watchlist universe")
    parser.add_argument("--exclude", nargs="*", default=["SPY"], help="Drop from universe")
    parser.add_argument("--factors", nargs="+", choices=sorted(FACTORS), default=None)
    parser.add_argument("--horizon", type=int, default=21, help="Forward return days")
    parser.add_argument("--quantiles", type=int, default=3)
    parser.add_argument("--output-json", default="reports/factor_report.json")
    parser.add_argument("--output-chart", default="reports/factor_ic.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_yaml(resolve_path(args.config))
    symbols = args.symbols or cfg.get("symbols", [])
    excluded = {s.upper() for s in args.exclude or []}
    symbols = [s for s in symbols if s.upper() not in excluded]

    store = ParquetStore(resolve_path(cfg.get("data_dir", "data/parquet")))
    panel = build_panel(store, symbols)
    logger.info("Universe: %s (%d dates)", ", ".join(panel.symbols), len(panel.close))

    factors = compute_factors(panel, args.factors)
    report, ic_series = evaluate_factors(factors, panel.close, args.horizon, args.quantiles)
    screen = latest_screen(factors)

    payload = {
        "universe": panel.symbols,
        "horizon_days": args.horizon,
        "quantiles": args.quantiles,
        "factors": report,
        "screen": screen,
    }
    out_json = resolve_path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_ic_chart(ic_series, resolve_path(args.output_chart))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
