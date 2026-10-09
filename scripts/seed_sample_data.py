"""Sample OHLCV bars for offline demos (no network required)."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data_pipeline.storage import ParquetStore
from shared.config import project_root


def generate_sample_prices(symbol: str = "SPY", days: int = 504) -> pd.DataFrame:
    """~2 years of synthetic daily bars with a mild upward drift."""
    start = date(2022, 1, 3)
    rows = []
    price = 400.0
    d = start
    while len(rows) < days:
        if d.weekday() < 5:
            drift = 0.0003
            shock = 0.01 * (((len(rows) * 7) % 11) - 5) / 5
            price = max(50.0, price * (1 + drift + shock))
            rows.append(
                {
                    "symbol": symbol,
                    "date": d,
                    "open": round(price * 0.999, 2),
                    "high": round(price * 1.005, 2),
                    "low": round(price * 0.995, 2),
                    "close": round(price, 2),
                    "volume": int(50_000_000 + (len(rows) % 20) * 1_000_000),
                }
            )
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def seed_sample_data(data_dir: str | Path | None = None, force: bool = False) -> list[str]:
    """Write synthetic bars, skipping symbols that already have data.

    Overwriting fetched data would leave the fetch metadata pointing at the old
    last_date, so the next incremental fetch would append live bars onto
    synthetic history. Use force=True only on a throwaway data directory.
    """
    root = project_root()
    store = ParquetStore(data_dir or root / "data" / "parquet")
    seeded = []
    for sym in ["SPY", "KO", "PEP"]:
        if store.symbol_path(sym).exists() and not force:
            continue
        store.save(sym, generate_sample_prices(sym))
        seeded.append(sym)
    return seeded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed synthetic OHLCV for offline demos")
    parser.add_argument("--force", action="store_true", help="Overwrite existing symbol files")
    args = parser.parse_args()
    seeded = seed_sample_data(force=args.force)
    if seeded:
        print(f"Seeded sample data for: {', '.join(seeded)}")
    else:
        print("All sample symbols already have data; nothing seeded (use --force to overwrite)")
