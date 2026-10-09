"""Build wide (date x symbol) price panels from the Parquet store."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_pipeline.storage import ParquetStore


@dataclass
class PricePanel:
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def build_panel(store: ParquetStore, symbols: list[str]) -> PricePanel:
    """Pivot per-symbol OHLCV into aligned close/volume panels.

    Dates missing for a symbol stay NaN (no forward fill), so a factor is only
    computed where real data exists.
    """
    frames = []
    for sym in symbols:
        df = store.load(sym)
        if df.empty:
            continue
        frames.append(df[["symbol", "date", "close", "volume"]])
    if not frames:
        raise ValueError(f"No data found for any of: {', '.join(symbols)}")

    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    close = long.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = long.pivot(index="date", columns="symbol", values="volume").sort_index()
    return PricePanel(close=close.astype(float), volume=volume.astype(float))
