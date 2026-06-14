"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
  """20 trading days of synthetic OHLCV for SPY."""
  start = date(2024, 1, 2)
  rows = []
  price = 100.0
  for i in range(20):
    d = start + timedelta(days=i)
    if d.weekday() >= 5:
      continue
    price += (-1) ** i * 0.5
    rows.append(
      {
        "symbol": "TEST",
        "date": d,
        "open": price - 0.2,
        "high": price + 0.3,
        "low": price - 0.3,
        "close": price,
        "volume": 1_000_000 + i * 1000,
      }
    )
  return pd.DataFrame(rows)


@pytest.fixture
def tmp_parquet_dir(tmp_path: Path) -> Path:
  return tmp_path / "parquet"
