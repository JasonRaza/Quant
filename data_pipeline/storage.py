"""Parquet storage for OHLCV bars."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.types import OHLCV_COLUMNS


class ParquetStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def symbol_path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.upper()}.parquet"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self.symbol_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[OHLCV_COLUMNS].sort_values("date").reset_index(drop=True)

    def save(self, symbol: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        path = self.symbol_path(symbol)
        out = df[OHLCV_COLUMNS].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out.to_parquet(path, index=False)
        return len(out)

    def merge_and_save(self, symbol: str, new_bars: pd.DataFrame) -> pd.DataFrame:
        existing = self.load(symbol)
        if existing.empty:
            combined = new_bars
        elif new_bars.empty:
            combined = existing
        else:
            combined = (
                pd.concat([existing, new_bars], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        self.save(symbol, combined)
        return combined

    def list_symbols(self) -> list[str]:
        return sorted(p.stem for p in self.data_dir.glob("*.parquet"))
