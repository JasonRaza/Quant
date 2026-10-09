"""Tests for ML features, purged CV and model evaluation."""

import numpy as np
import pandas as pd
import pytest

from factors.panel import PricePanel
from ml_alpha.cv import PurgedWalkForward, leaky_kfold
from ml_alpha.features import build_dataset, cross_sectional_rank
from ml_alpha.models import (
    SingleFeatureBaseline,
    cross_validate,
    daily_rank_ic,
    default_models,
    top_minus_bottom,
)

SYMBOLS = ["A", "B", "C", "D", "E", "F"]


def make_panel(n_days: int = 600, seed: int = 0) -> PricePanel:
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (n_days, len(SYMBOLS))), axis=0)),
        index=idx,
        columns=SYMBOLS,
    )
    volume = pd.DataFrame(rng.uniform(1e6, 2e6, (n_days, len(SYMBOLS))), index=idx, columns=SYMBOLS)
    return PricePanel(close=close, volume=volume)


def test_cross_sectional_rank_is_centered():
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [30.0, 10.0, np.nan]])
    ranked = cross_sectional_rank(frame)
    np.testing.assert_allclose(ranked.iloc[0], [-1 / 3, 0, 1 / 3])
    np.testing.assert_allclose(ranked.iloc[1, :2], [0.25, -0.25])


def test_dataset_shape_and_label():
    panel = make_panel()
    ds = build_dataset(panel, horizon=21, sample_every=5)
    assert ds.X.index.names == ["date", "symbol"]
    assert not ds.X.isna().any().any()
    # Labels are demeaned across symbols on each date.
    assert ds.y.groupby(level="date").mean().abs().max() < 1e-12
    # Every label window ends after its feature date.
    assert (ds.label_end.to_numpy() > ds.dates.to_numpy()).all()


def test_features_do_not_use_future_prices():
    panel = make_panel()
    cut = panel.close.index[450]
    altered = PricePanel(close=panel.close.copy(), volume=panel.volume.copy())
    altered.close.loc[altered.close.index > cut] *= 2.5

    base = build_dataset(panel).X
    new = build_dataset(altered).X
    mask = base.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(base[mask], new.loc[base.index[mask]])


def test_purged_walk_forward_has_no_leakage():
    ds = build_dataset(make_panel(), horizon=21, sample_every=5)
    splitter = PurgedWalkForward(n_splits=4, min_train_dates=30)
    folds = list(splitter.split(ds.dates, ds.label_end))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        test_start = ds.dates[test_idx].min()
        assert ds.dates[train_idx].max() < test_start
        assert ds.label_end.iloc[train_idx].max() < test_start  # purged
    # Test blocks cover consecutive, non-overlapping dates.
    starts = [ds.dates[t].min() for _, t in folds]
    assert starts == sorted(starts)


def test_purging_actually_removes_samples():
    ds = build_dataset(make_panel(), horizon=21, sample_every=5)
    splitter = PurgedWalkForward(n_splits=2, min_train_dates=30)
    train_idx, test_idx = next(splitter.split(ds.dates, ds.label_end))
    test_start = ds.dates[test_idx].min()
    before_test = np.flatnonzero(ds.dates < test_start)
    assert len(train_idx) < len(before_test)  # last few label windows overlap the test block


def test_embargo_shrinks_training_set():
    ds = build_dataset(make_panel(), horizon=21, sample_every=5)
    plain = next(PurgedWalkForward(2, 30).split(ds.dates, ds.label_end))[0]
    embargoed = next(PurgedWalkForward(2, 30, embargo_days=30).split(ds.dates, ds.label_end))[0]
    assert len(embargoed) < len(plain)


def test_leaky_kfold_covers_all_dates_once():
    ds = build_dataset(make_panel(), horizon=21, sample_every=5)
    tested = np.concatenate([t for _, t in leaky_kfold(ds.dates, 5)])
    assert sorted(tested) == list(range(len(ds.X)))


def test_ic_and_long_short_helpers():
    dates = pd.bdate_range("2024-01-01", periods=3)
    idx = pd.MultiIndex.from_product([dates, list("abc")], names=["date", "symbol"])
    y = pd.Series(np.tile([0.01, 0.02, 0.03], 3), index=idx)
    assert daily_rank_ic(y, y).tolist() == pytest.approx([1.0] * 3)
    assert daily_rank_ic(-y, y).tolist() == pytest.approx([-1.0] * 3)
    assert top_minus_bottom(y, y).tolist() == pytest.approx([0.02] * 3)


def test_models_learn_a_planted_signal():
    """If the label really depends on a feature, purged CV should find it."""
    ds = build_dataset(make_panel(n_days=700), horizon=21, sample_every=5)
    rng = np.random.default_rng(1)
    y = 0.05 * ds.X["ret_63d"] + rng.normal(0, 0.01, len(ds.y))
    y = y - y.groupby(level="date").transform("mean")
    splits = list(PurgedWalkForward(3, 40).split(ds.dates, ds.label_end))

    for name, model in default_models().items():
        if name == "momentum_baseline":
            continue
        res = cross_validate(model, ds.X, y, splits, overlap=4)
        assert res.summary["ic_mean"] > 0.5, name
        assert res.long_short.mean() > 0, name

    baseline = cross_validate(SingleFeatureBaseline("ret_63d"), ds.X, y, splits)
    assert baseline.summary["ic_mean"] > 0.5
