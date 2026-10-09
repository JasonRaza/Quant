"""Purged walk-forward cross-validation for overlapping financial labels.

A sample at date t has a label built from prices up to t + horizon. If such a
sample sits in the training set while t + horizon falls inside the test period,
the model has been trained on test-period prices: classic leakage that makes
backtested ML look far better than it is. Purging drops those training samples.
Splits are by date, so all symbols on a date land in the same fold.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


class PurgedWalkForward:
    """Expanding-window splits: train on the past, test on the next block of dates.

    Training samples whose label window ends on or after the test start
    (+ `embargo_days` of extra safety margin) are purged.
    """

    def __init__(self, n_splits: int = 5, min_train_dates: int = 52, embargo_days: int = 0):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.min_train_dates = min_train_dates
        self.embargo = pd.Timedelta(days=embargo_days)

    def split(
        self, dates: pd.Index, label_end: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = pd.DatetimeIndex(dates)
        label_end = pd.DatetimeIndex(label_end)
        unique = dates.unique().sort_values()
        if len(unique) <= self.min_train_dates + self.n_splits:
            raise ValueError("not enough dates for the requested splits")

        test_blocks = np.array_split(unique[self.min_train_dates :], self.n_splits)
        for block in test_blocks:
            test_start, test_end = block[0], block[-1]
            test_idx = np.flatnonzero((dates >= test_start) & (dates <= test_end))
            train_mask = (dates < test_start) & (label_end < test_start - self.embargo)
            yield np.flatnonzero(train_mask), test_idx


def leaky_kfold(dates: pd.Index, n_splits: int = 5, seed: int = 0):
    """Shuffled K-fold ignoring time. Only used to show how much leakage inflates scores."""
    rng = np.random.default_rng(seed)
    unique = pd.DatetimeIndex(dates).unique()
    folds = np.array_split(rng.permutation(len(unique)), n_splits)
    dates = pd.DatetimeIndex(dates)
    for fold in folds:
        test_mask = dates.isin(unique[fold])
        yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
