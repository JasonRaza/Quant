"""Common types and schemas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

OHLCV_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


@dataclass
class Bar:
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize raw provider data to standard OHLCV schema."""
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    out = df.copy()
    out = out.reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out["date"] = pd.to_datetime(out[date_col]).dt.date
    out["symbol"] = symbol

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[OHLCV_COLUMNS].dropna(subset=["date", "close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)
