"""Orchestrate idempotent market data pipeline runs."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from data_pipeline.fetcher import fetch_symbol_bars, incremental_start
from data_pipeline.storage import ParquetStore
from shared.config import load_yaml, resolve_path
from shared.db import MetadataStore

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config/watchlist.yaml") -> dict[str, Any]:
    cfg = load_yaml(resolve_path(config_path))
    symbols: list[str] = cfg.get("symbols", [])
    default_start = date.fromisoformat(cfg["start_date"])
    end_date: Optional[date] = None
    if cfg.get("end_date"):
        end_date = date.fromisoformat(cfg["end_date"])

    fetch_cfg = cfg.get("fetch", {})
    max_retries = int(fetch_cfg.get("max_retries", 3))
    retry_delay = float(fetch_cfg.get("retry_delay_seconds", 2))

    store = ParquetStore(resolve_path(cfg["data_dir"]))
    metadata = MetadataStore(resolve_path(cfg["metadata_db"]))

    results: dict[str, Any] = {"symbols": {}, "errors": []}

    for symbol in symbols:
        symbol = symbol.upper()
        try:
            last_date = metadata.get_last_date(symbol)
            start = incremental_start(last_date, default_start)
            if end_date and start > end_date:
                logger.info("%s already up to date (last=%s)", symbol, last_date)
                results["symbols"][symbol] = {"status": "skipped", "rows_added": 0}
                continue

            new_bars = fetch_symbol_bars(
                symbol,
                start=start,
                end=end_date,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            combined = store.merge_and_save(symbol, new_bars)
            last = combined["date"].max() if not combined.empty else last_date
            metadata.update_symbol(symbol, last, len(combined))
            results["symbols"][symbol] = {
                "status": "ok",
                "rows_added": len(new_bars),
                "total_rows": len(combined),
                "last_date": last.isoformat() if last else None,
            }
            logger.info(
                "Updated %s: +%d bars, total=%d, last=%s",
                symbol,
                len(new_bars),
                len(combined),
                last,
            )
        except Exception as exc:
            logger.exception("Failed processing %s", symbol)
            metadata.update_symbol(symbol, metadata.get_last_date(symbol), 0, str(exc))
            results["errors"].append({"symbol": symbol, "error": str(exc)})
            results["symbols"][symbol] = {"status": "error", "error": str(exc)}

    return results
