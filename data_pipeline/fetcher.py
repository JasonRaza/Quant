"""Download OHLCV data from Yahoo Finance with retries."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from shared.types import normalize_ohlcv

logger = logging.getLogger(__name__)


def fetch_symbol_bars(
    symbol: str,
    start: date,
    end: Optional[date] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Fetch daily OHLCV bars for a single symbol."""
    end_str = (end or date.today()).isoformat()
    start_str = start.isoformat()
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(start=start_str, end=end_str, auto_adjust=True)
            if raw.empty:
                logger.warning("No data returned for %s (%s to %s)", symbol, start_str, end_str)
                return normalize_ohlcv(raw, symbol)
            return normalize_ohlcv(raw, symbol)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Fetch attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                symbol,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    raise RuntimeError(f"Failed to fetch {symbol} after {max_retries} attempts") from last_error


def incremental_start(last_date: Optional[date], default_start: date) -> date:
    """Return start date for incremental fetch (day after last stored bar)."""
    if last_date is None:
        return default_start
    return last_date + timedelta(days=1)
