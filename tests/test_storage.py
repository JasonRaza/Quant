"""Tests for Parquet storage."""

import pandas as pd

from data_pipeline.storage import ParquetStore
from scripts.seed_sample_data import seed_sample_data


def test_merge_and_save_idempotent(synthetic_bars, tmp_parquet_dir):
    store = ParquetStore(tmp_parquet_dir)
    store.merge_and_save("TEST", synthetic_bars)
    first_len = len(store.load("TEST"))

    store.merge_and_save("TEST", synthetic_bars)
    second_len = len(store.load("TEST"))
    assert first_len == second_len == len(synthetic_bars)


def test_incremental_merge(synthetic_bars, tmp_parquet_dir):
    store = ParquetStore(tmp_parquet_dir)
    half = synthetic_bars.iloc[:10]
    rest = synthetic_bars.iloc[10:]
    store.merge_and_save("TEST", half)
    store.merge_and_save("TEST", rest)
    loaded = store.load("TEST")
    assert len(loaded) == len(synthetic_bars)


def test_seed_does_not_overwrite_existing_data(tmp_parquet_dir):
    store = ParquetStore(tmp_parquet_dir)
    real = pd.DataFrame(
        [{"symbol": "KO", "date": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 60.0,
          "volume": 1}]
    )
    store.save("KO", real)

    seeded = seed_sample_data(tmp_parquet_dir)
    assert "KO" not in seeded
    assert len(store.load("KO")) == 1

    assert "KO" in seed_sample_data(tmp_parquet_dir, force=True)
    assert len(store.load("KO")) > 1
