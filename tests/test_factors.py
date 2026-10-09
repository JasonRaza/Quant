"""Tests for factor panel construction and factor definitions."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data_pipeline.storage import ParquetStore
from factors.library import compute_factors, low_volatility, momentum_12_1, short_term_reversal
from factors.panel import PricePanel, build_panel


def make_panel(n_days: int = 300, symbols=("A", "B", "C", "D")) -> PricePanel:
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(symbols))), axis=0)),
        index=idx,
        columns=list(symbols),
    )
    volume = pd.DataFrame(1_000_000.0, index=idx, columns=list(symbols))
    return PricePanel(close=close, volume=volume)


def test_build_panel_aligns_symbols(tmp_parquet_dir):
    store = ParquetStore(tmp_parquet_dir)
    for sym, n in [("AAA", 5), ("BBB", 3)]:
        rows = [
            {
                "symbol": sym,
                "date": date(2024, 1, 1) + timedelta(days=i),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 10.0 + i,
                "volume": 100,
            }
            for i in range(n)
        ]
        store.save(sym, pd.DataFrame(rows))

    panel = build_panel(store, ["AAA", "BBB", "MISSING"])
    assert panel.symbols == ["AAA", "BBB"]
    assert panel.close.shape == (5, 2)
    assert panel.close["BBB"].isna().sum() == 2  # no forward fill


def test_build_panel_empty_raises(tmp_parquet_dir):
    with pytest.raises(ValueError):
        build_panel(ParquetStore(tmp_parquet_dir), ["NOPE"])


def test_momentum_skips_recent_month():
    panel = make_panel()
    mom = momentum_12_1(panel)
    t = panel.close.index[-1]
    expected = panel.close["A"].iloc[-1 - 21] / panel.close["A"].iloc[-1 - 252] - 1
    assert mom.loc[t, "A"] == pytest.approx(expected)
    assert mom.iloc[:252].isna().all().all()


def test_factors_have_no_lookahead():
    """Changing future prices must not change today's factor values."""
    panel = make_panel()
    cut = 280
    altered = PricePanel(close=panel.close.copy(), volume=panel.volume.copy())
    altered.close.iloc[cut + 1 :] *= 3.0

    for name, base in compute_factors(panel).items():
        new = compute_factors(altered, [name])[name]
        pd.testing.assert_frame_equal(base.iloc[: cut + 1], new.iloc[: cut + 1])


def test_reversal_and_low_vol_signs():
    panel = make_panel()
    panel.close["A"] = np.linspace(100, 200, len(panel.close))  # steady riser
    rev = short_term_reversal(panel).iloc[-1]
    assert rev["A"] < 0  # recent winner gets a negative reversal score
    vol = low_volatility(panel).iloc[-1]
    assert vol.idxmax() == "A"  # smooth series is the least volatile


def test_unknown_factor_raises():
    with pytest.raises(KeyError):
        compute_factors(make_panel(), ["nope"])
