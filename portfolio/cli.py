"""CLI: current optimal weights plus a walk-forward comparison of allocators."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from backtester.metrics import summarize
from data_pipeline.storage import ParquetStore
from factors.analysis import composite_score
from factors.library import compute_factors
from factors.panel import build_panel
from portfolio.estimators import daily_returns, ledoit_wolf_covariance
from portfolio.optimizer import Constraints, portfolio_stats, risk_contributions
from portfolio.walkforward import (
    WalkForwardConfig,
    default_allocators,
    historical_max_sharpe,
    run_walkforward,
    weights_concentration,
)
from shared.config import load_yaml, resolve_path

logger = logging.getLogger(__name__)


def benchmark_equity(store: ParquetStore, symbol: str, index: pd.Index, capital: float):
    df = store.load(symbol)
    if df.empty:
        return None
    close = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
    close = close.reindex(index).ffill()
    return (capital * close / close.iloc[0]).rename(symbol)


def save_equity_chart(curves: dict[str, pd.Series], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, eq in curves.items():
        style = {"linestyle": "--", "color": "gray"} if name.startswith("benchmark") else {}
        eq.plot(ax=ax, label=name, **style)
    ax.set_title("Walk-forward portfolio equity (monthly rebalance, net of costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio optimization and walk-forward test")
    parser.add_argument("--config", default="config/watchlist.yaml")
    parser.add_argument("--symbols", nargs="+", help="Override the watchlist universe")
    parser.add_argument("--exclude", nargs="*", default=["SPY"], help="Drop from universe")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--ic", type=float, default=0.05, help="Assumed IC for factor alpha")
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--output-json", default="reports/portfolio_report.json")
    parser.add_argument("--output-chart", default="reports/portfolio_equity.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_yaml(resolve_path(args.config))
    excluded = {s.upper() for s in args.exclude or []}
    symbols = [s for s in (args.symbols or cfg.get("symbols", [])) if s.upper() not in excluded]

    store = ParquetStore(resolve_path(cfg.get("data_dir", "data/parquet")))
    panel = build_panel(store, symbols)
    returns = daily_returns(panel.close)
    scores = composite_score(compute_factors(panel))
    logger.info("Universe: %s (%d return days)", ", ".join(panel.symbols), len(returns))

    constraints = Constraints(max_weight=args.max_weight)
    wf_cfg = WalkForwardConfig(
        lookback=args.lookback, rebalance_every=args.rebalance_every, cost_bps=args.cost_bps
    )
    allocators = default_allocators(constraints, scores, args.ic, args.risk_aversion)
    allocators["max_sharpe_historical"] = historical_max_sharpe(constraints)

    walkforward, curves, current = {}, {}, {}
    last_window = returns.iloc[-args.lookback :].dropna(axis=1)
    cov = ledoit_wolf_covariance(last_window)
    mu_hist = last_window.mean() * 252

    for name, alloc in allocators.items():
        res = run_walkforward(returns, alloc, wf_cfg)
        res.metrics["avg_effective_holdings"] = round(weights_concentration(res.weights), 2)
        walkforward[name] = res.metrics
        curves[name] = res.equity

        w = alloc(last_window, returns.index[-1])
        current[name] = {
            "weights": {s: round(float(v), 4) for s, v in w.sort_values(ascending=False).items()},
            "ex_ante_volatility": portfolio_stats(w, mu_hist, cov)["volatility"],
            "risk_contribution": {
                s: round(float(v), 4) for s, v in risk_contributions(w, cov).items()
            },
        }

    any_curve = next(iter(curves.values()))
    bench = benchmark_equity(store, args.benchmark, any_curve.index, wf_cfg.initial_capital)
    if bench is not None:
        walkforward[f"benchmark_{args.benchmark}"] = summarize(bench, pd.DataFrame())
        curves[f"benchmark_{args.benchmark}"] = bench

    payload = {
        "universe": panel.symbols,
        "as_of": str(returns.index[-1].date()),
        "settings": {
            "max_weight": args.max_weight,
            "lookback_days": args.lookback,
            "rebalance_every_days": args.rebalance_every,
            "cost_bps": args.cost_bps,
            "factor_ic": args.ic,
            "risk_aversion": args.risk_aversion,
        },
        "walkforward": walkforward,
        "current_allocation": current,
    }
    out_json = resolve_path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_equity_chart(curves, resolve_path(args.output_chart))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
