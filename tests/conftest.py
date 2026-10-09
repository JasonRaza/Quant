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


@pytest.fixture(scope="session")
def synthetic_market(tmp_path_factory):
    """Six-plus years of random-walk bars for every watchlist ticker, plus a config file.

    Lets CLI, dashboard and tutorial tests run offline without the real data.
    """
    import numpy as np
    import yaml

    from data_pipeline.storage import ParquetStore
    from shared.config import load_yaml, resolve_path

    root = tmp_path_factory.mktemp("market")
    data_dir = root / "parquet"
    store = ParquetStore(data_dir)
    symbols = load_yaml(resolve_path("config/watchlist.yaml"))["symbols"]
    dates = pd.bdate_range("2020-01-02", "2026-06-30")
    rng = np.random.default_rng(0)
    for i, sym in enumerate(symbols):
        close = 50 * (i + 1) * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(dates))))
        store.save(
            sym,
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates.date,
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
                }
            ),
        )
    config = root / "watchlist.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "symbols": symbols,
                "start_date": "2020-01-01",
                "end_date": None,
                "data_dir": str(data_dir),
                "metadata_db": str(root / "metadata.sqlite"),
                "paper_db": str(root / "paper.sqlite"),
            }
        ),
        encoding="utf-8",
    )
    return {"root": root, "data_dir": data_dir, "config": config}


@pytest.fixture
def streamlit_on_synthetic_data(synthetic_market, monkeypatch):
    """Point Settings at the synthetic data and reset Streamlit caches around the test."""
    import streamlit as st

    monkeypatch.setenv("QUANT_DATA_DIR", str(synthetic_market["data_dir"]))
    monkeypatch.setenv("QUANT_PAPER_DB", str(synthetic_market["root"] / "paper.sqlite"))
    st.cache_data.clear()
    st.cache_resource.clear()
    yield synthetic_market
    st.cache_data.clear()
    st.cache_resource.clear()
