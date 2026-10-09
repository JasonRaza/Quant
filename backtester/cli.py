"""CLI for running backtests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.strategies import (
    DualMovingAverageCrossover,
    MeanReversionPair,
    pair_spread_prices,
)
from data_pipeline.storage import ParquetStore
from shared.config import resolve_path

logger = logging.getLogger(__name__)


def load_prices(data_dir: Path, symbol: str) -> pd.DataFrame:
    store = ParquetStore(data_dir)
    df = store.load(symbol)
    if df.empty:
        raise FileNotFoundError(
            f"No data for {symbol}. Run: quant-fetch --config config/watchlist.yaml"
        )
    return df


def load_pair_prices(data_dir: Path, sym_a: str, sym_b: str) -> pd.DataFrame:
    a = load_prices(data_dir, sym_a)[["date", "close"]].rename(columns={"close": "close_a"})
    b = load_prices(data_dir, sym_b)[["date", "close"]].rename(columns={"close": "close_b"})
    merged = pd.merge(a, b, on="date", how="inner").sort_values("date")
    return pair_spread_prices(merged.reset_index(drop=True))


def save_equity_chart(result, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity.plot(ax=ax, label="Strategy")
    result.benchmark_equity.plot(ax=ax, label="Buy & Hold", alpha=0.7)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-driven backtest")
    parser.add_argument("--data-dir", default="data/parquet")
    parser.add_argument("--strategy", choices=["dual_ma", "mean_reversion"], default="dual_ma")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--pair", nargs=2, metavar=("SYM_A", "SYM_B"), default=["KO", "PEP"])
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-chart", default="reports/equity_curve.png")
    parser.add_argument("--output-json", default="reports/backtest_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = resolve_path(args.data_dir)

    # A pair trade has two legs, so each spread trade pays costs twice.
    legs = 2 if args.strategy == "mean_reversion" else 1
    config = BacktestConfig(
        initial_capital=args.capital,
        commission_bps=args.commission_bps * legs,
        slippage_bps=args.slippage_bps * legs,
    )
    engine = BacktestEngine(config)

    if args.strategy == "dual_ma":
        prices = load_prices(data_dir, args.symbol)
        strategy = DualMovingAverageCrossover(fast=args.fast, slow=args.slow)
    else:
        prices = load_pair_prices(data_dir, args.pair[0], args.pair[1])
        strategy = MeanReversionPair()

    result = engine.run(prices, strategy)

    out_json = resolve_path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": args.strategy,
        "symbol": args.symbol if args.strategy == "dual_ma" else list(args.pair),
        "metrics": result.metrics,
        "trade_count": len(result.trades),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    save_equity_chart(result, resolve_path(args.output_chart))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
