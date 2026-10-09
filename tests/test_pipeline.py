"""Tests for data pipeline orchestration."""

from datetime import date
from unittest.mock import patch

import pandas as pd

from data_pipeline.pipeline import run_pipeline


def test_pipeline_fetches_and_stores(tmp_path, monkeypatch):
    root = tmp_path
    cfg_dir = root / "config"
    cfg_dir.mkdir()
    data_dir = root / "data" / "parquet"
    db_path = root / "data" / "metadata.sqlite"
    cfg_dir.joinpath("watchlist.yaml").write_text(
        f"""
symbols: [FAKE]
start_date: "2024-01-01"
data_dir: "{data_dir.as_posix()}"
metadata_db: "{db_path.as_posix()}"
fetch:
  max_retries: 1
  retry_delay_seconds: 0
""",
        encoding="utf-8",
    )

    fake_bars = pd.DataFrame(
        {
            "symbol": ["FAKE", "FAKE"],
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "open": [10.0, 10.5],
            "high": [10.5, 11.0],
            "low": [9.5, 10.0],
            "close": [10.2, 10.8],
            "volume": [1000, 1100],
        }
    )

    with patch("data_pipeline.pipeline.fetch_symbol_bars", return_value=fake_bars):
        results = run_pipeline(str(cfg_dir / "watchlist.yaml"))

    assert results["symbols"]["FAKE"]["status"] == "ok"
    assert (data_dir / "FAKE.parquet").exists()
