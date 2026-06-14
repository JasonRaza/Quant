"""Trading strategies for backtesting."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return position signal in [-1, 1] or {0, 1} aligned to price index."""
        ...


class DualMovingAverageCrossover(Strategy):
    """Long when fast MA > slow MA, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError("fast period must be less than slow period")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        signal = (fast_ma > slow_ma).astype(float)
        signal = signal.shift(1).fillna(0.0)  # trade next bar; no lookahead
        return signal


class MeanReversionPair(Strategy):
    """Mean reversion on spread z-score between two symbols."""

    def __init__(self, lookback: int = 20, entry_z: float = 2.0, exit_z: float = 0.5) -> None:
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if "close_a" not in prices.columns or "close_b" not in prices.columns:
            raise ValueError("MeanReversionPair requires close_a and close_b columns")
        spread = prices["close_a"] - prices["close_b"]
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        z = (spread - mean) / std

        position = pd.Series(0.0, index=prices.index)
        current = 0.0
        for i in range(len(z)):
            zi = z.iloc[i]
            if pd.isna(zi):
                position.iloc[i] = 0.0
                continue
            if current == 0.0:
                if zi > self.entry_z:
                    current = -1.0
                elif zi < -self.entry_z:
                    current = 1.0
            else:
                if abs(zi) < self.exit_z:
                    current = 0.0
            position.iloc[i] = current
        return position.shift(1).fillna(0.0)
