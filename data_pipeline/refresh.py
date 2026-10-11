"""Bring a copy of a data snapshot up to date, tolerating fetch failures.

Used by the hosted app: the committed snapshot is copied to a writable folder and only
the missing days are downloaded. Any ticker that can't be fetched keeps its snapshot
history, so the app always has data to show.
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from data_pipeline.storage import ParquetStore

logger = logging.getLogger(__name__)

Fetcher = Callable[..., pd.DataFrame]


@dataclass
class RefreshReport:
    updated: dict[str, int] = field(default_factory=dict)  # ticker -> new rows
    failed: dict[str, str] = field(default_factory=dict)  # ticker -> error message
    last_date: Optional[date] = None


def refresh_copy(
    source_dir: Path,
    target_dir: Path,
    fetch: Optional[Fetcher] = None,
    today: Optional[date] = None,
    max_workers: int = 10,
    include_today: Optional[bool] = None,
) -> RefreshReport:
    """Sync snapshot files into `target_dir`, then append bars after each last date."""
    if fetch is None:
        from data_pipeline.fetcher import fetch_symbol_bars

        fetch = fetch_symbol_bars
    fetch_bars: Fetcher = fetch

    target_dir.mkdir(parents=True, exist_ok=True)
    source, store = ParquetStore(source_dir), ParquetStore(target_dir)
    for src in source_dir.glob("*.parquet"):
        dst = target_dir / src.name
        # Take the snapshot copy unless the target already has more recent data.
        target_last = _last_date(store, src.stem) if dst.exists() else None
        source_last = _last_date(source, src.stem)
        if target_last is None or (source_last is not None and source_last >= target_last):
            shutil.copy2(src, dst)

    today = today or date.today()
    if include_today is None:
        include_today = us_session_closed()
    report = RefreshReport()

    def update(symbol: str) -> None:
        existing = store.load(symbol)
        start = existing["date"].max() + timedelta(days=1) if not existing.empty else None
        if start is None or start > today:
            return
        try:
            # `end` is exclusive: ask for today's bar too, then keep it only once the US
            # session has closed so an intraday price is never stored as a daily close.
            new_bars = fetch_bars(
                symbol, start=start, end=today + timedelta(days=1), max_retries=1, retry_delay=0
            )
        except Exception as exc:  # network errors, rate limits, provider changes
            report.failed[symbol] = str(exc)[:200]
            return
        bar_dates = pd.to_datetime(new_bars["date"]).dt.date
        keep = (bar_dates >= start) & (bar_dates <= today) & ((bar_dates < today) | include_today)
        new_bars = new_bars[keep]
        if not new_bars.empty:
            store.merge_and_save(symbol, new_bars)
            report.updated[symbol] = len(new_bars)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(update, store.list_symbols()))

    last_dates = [d for s in store.list_symbols() if (d := _last_date(store, s)) is not None]
    report.last_date = max(last_dates) if last_dates else None
    if report.failed:
        logger.warning("Live refresh failed for %s", ", ".join(sorted(report.failed)))
    return report


def _last_date(store: ParquetStore, symbol: str) -> Optional[date]:
    df = store.load(symbol)
    return None if df.empty else df["date"].max()


def us_session_closed(now: Optional[pd.Timestamp] = None) -> bool:
    """True once today's US equity session is over (4:30pm New York, after the close)."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    return now.hour * 60 + now.minute >= 16 * 60 + 30
