"""Data quality reporting for stored OHLCV bars."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data_pipeline.storage import ParquetStore
from shared.config import load_yaml, resolve_path


def _expected_trading_days(start: date, end: date) -> int:
    """Approximate US trading days (Mon-Fri) in range."""
    days = pd.bdate_range(start, end)
    return len(days)


def generate_quality_report(
    data_dir: str | Path,
    stale_threshold_days: int = 5,
) -> dict:
    store = ParquetStore(data_dir)
    symbols = store.list_symbols()
    today = date.today()
    report: dict = {"generated_at": today.isoformat(), "symbols": {}, "summary": {}}

    missing_pct_list = []
    stale_count = 0

    for symbol in symbols:
        df = store.load(symbol)
        if df.empty:
            report["symbols"][symbol] = {"status": "empty"}
            continue

        start = df["date"].min()
        end = df["date"].max()
        expected = _expected_trading_days(start, end)
        actual = len(df)
        missing_pct = max(0.0, (expected - actual) / expected * 100) if expected else 0.0
        days_since_update = (today - end).days
        is_stale = days_since_update > stale_threshold_days

        vol = df["volume"].astype(float)
        vol_mean = vol.mean()
        vol_std = vol.std()
        spike_threshold = vol_mean + 3 * vol_std if vol_std > 0 else float("inf")
        volume_spikes = int((vol > spike_threshold).sum())

        entry = {
            "rows": actual,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "missing_pct_approx": round(missing_pct, 2),
            "days_since_update": days_since_update,
            "stale": is_stale,
            "volume_spikes_3sigma": volume_spikes,
        }
        report["symbols"][symbol] = entry
        missing_pct_list.append(missing_pct)
        if is_stale:
            stale_count += 1

    report["summary"] = {
        "symbol_count": len(symbols),
        "avg_missing_pct": round(sum(missing_pct_list) / len(missing_pct_list), 2)
        if missing_pct_list
        else 0.0,
        "stale_symbols": stale_count,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OHLCV data quality report")
    parser.add_argument("--config", default="config/watchlist.yaml")
    parser.add_argument("--stale-days", type=int, default=5)
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    report = generate_quality_report(resolve_path(cfg["data_dir"]), args.stale_days)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
