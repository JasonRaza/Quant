"""Daily refresh of the bundled data snapshot for the hosted app.

When the app runs on the committed snapshot (a deployment or a fresh clone), it tries
to download the days missing since the snapshot, at most once every 12 hours for all
visitors. If the provider refuses, the app quietly keeps the snapshot. Local setups
with their own fetched data are left untouched. Set QUANT_LIVE_REFRESH=0 to disable.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import streamlit as st

from data_pipeline.refresh import RefreshReport, refresh_copy
from paper_trading.service import SNAPSHOT_DIR, Settings

LIVE_DIR = Path(tempfile.gettempdir()) / "quant_live_data"


@st.cache_resource(ttl=timedelta(hours=12), show_spinner="Fetching the latest market prices...")
def _refreshed() -> tuple[Path, RefreshReport]:
    report = refresh_copy(SNAPSHOT_DIR, LIVE_DIR)
    if report.updated:
        # Cached charts and models were computed on the older prices.
        st.cache_data.clear()
    return LIVE_DIR, report


def live_settings() -> Settings:
    """Settings pointing at freshly updated data when running on the snapshot."""
    cfg = Settings.from_config()
    if cfg.data_dir != SNAPSHOT_DIR or os.environ.get("QUANT_LIVE_REFRESH") == "0":
        return cfg
    try:
        live_dir, _ = _refreshed()
    except Exception:  # never let a refresh problem take the app down
        return cfg
    return replace(cfg, data_dir=live_dir)


def refresh_status() -> RefreshReport | None:
    """Result of the last refresh, or None if the app isn't using live refresh."""
    cfg = Settings.from_config()
    if cfg.data_dir != SNAPSHOT_DIR or os.environ.get("QUANT_LIVE_REFRESH") == "0":
        return None
    try:
        return _refreshed()[1]
    except Exception:
        return None
