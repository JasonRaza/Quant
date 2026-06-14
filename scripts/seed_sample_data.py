"""Sample OHLCV bars for offline demos (no network required)."""

from __future__ import annotations

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


def seed_sample_data(data_dir: str | Path | None = None) -> list[str]:
    root = project_root()
    store = ParquetStore(data_dir or root / "data" / "parquet")
    symbols = ["SPY", "KO", "PEP"]
    for sym in symbols:
        store.save(sym, generate_sample_prices(sym))
    return symbols


if __name__ == "__main__":
    seeded = seed_sample_data()
    print(f"Seeded sample data for: {', '.join(seeded)}")
