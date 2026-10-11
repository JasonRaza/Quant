"""Simulated broker: market orders filled at the latest close, persisted in SQLite."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from data_pipeline.storage import ParquetStore

PriceSource = Callable[[str], Optional[float]]


class ParquetPriceSource:
    """Latest close from the local Parquet store (refresh with quant-fetch)."""

    def __init__(self, data_dir: str | Path) -> None:
        self.store = ParquetStore(data_dir)

    def __call__(self, symbol: str) -> Optional[float]:
        df = self.store.load(symbol)
        if df.empty:
            return None
        return float(df["close"].iloc[-1])

    def as_of(self, symbol: str) -> Optional[str]:
        df = self.store.load(symbol)
        return None if df.empty else str(df["date"].iloc[-1])


@dataclass
class Order:
    id: Optional[int]
    timestamp: str
    symbol: str
    side: str
    quantity: int
    price: Optional[float]
    commission: float
    status: str
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperBroker:
    """Long-only cash account. Whole-share market orders, filled immediately.

    Fill price = last close +/- slippage; commission is charged on notional.
    """

    def __init__(
        self,
        db_path: str | Path,
        price_source: PriceSource,
        initial_cash: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.price_source = price_source
        self.commission_rate = commission_bps / 10_000
        self.slippage_rate = slippage_bps / 10_000
        self._init_db(initial_cash)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self, initial_cash: float) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL,
                    initial_cash REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL,
                    commission REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    reason TEXT
                );
                """
            )
            if conn.execute("SELECT 1 FROM account").fetchone() is None:
                conn.execute(
                    "INSERT INTO account (id, cash, initial_cash, created_at) VALUES (1, ?, ?, ?)",
                    (initial_cash, initial_cash, _now()),
                )
            conn.commit()

    def reset(self, initial_cash: float = 100_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        with self._connect() as conn:
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM orders")
            conn.execute(
                "UPDATE account SET cash = ?, initial_cash = ?, realized_pnl = 0, created_at = ?"
                " WHERE id = 1",
                (initial_cash, initial_cash, _now()),
            )
            conn.commit()

    # ---- reads -----------------------------------------------------------------

    def account(self) -> dict[str, Any]:
        with self._connect() as conn:
            acct = dict(conn.execute("SELECT * FROM account WHERE id = 1").fetchone())
        pos = self.positions()
        market_value = float(pos["market_value"].sum()) if not pos.empty else 0.0
        unrealized = float(pos["unrealized_pnl"].sum()) if not pos.empty else 0.0
        equity = acct["cash"] + market_value
        return {
            "cash": round(acct["cash"], 2),
            "market_value": round(market_value, 2),
            "equity": round(equity, 2),
            "initial_cash": round(acct["initial_cash"], 2),
            "total_return": round(equity / acct["initial_cash"] - 1, 4),
            "realized_pnl": round(acct["realized_pnl"], 2),
            "unrealized_pnl": round(unrealized, 2),
            "created_at": acct["created_at"],
        }

    def positions(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM positions ORDER BY symbol")]
        cols = ["symbol", "quantity", "avg_cost", "price", "market_value", "unrealized_pnl"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        df["price"] = [self.price_source(s) or 0.0 for s in df["symbol"]]
        df["market_value"] = df["quantity"] * df["price"]
        df["unrealized_pnl"] = (df["price"] - df["avg_cost"]) * df["quantity"]
        return df[cols]

    def orders(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as conn:
            rows = [
                dict(r)
                for r in conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
            ]
        return pd.DataFrame(rows, columns=list(Order.__dataclass_fields__))

    # ---- trading ---------------------------------------------------------------

    def submit_order(self, symbol: str, side: str, quantity: int) -> Order:
        symbol, side = symbol.upper().strip(), side.lower().strip()
        order = Order(None, _now(), symbol, side, int(quantity), None, 0.0, "rejected")

        if side not in ("buy", "sell"):
            return self._record(order, f"invalid side '{side}'")
        if quantity != int(quantity) or quantity <= 0:
            return self._record(order, "quantity must be a positive whole number")
        last = self.price_source(symbol)
        if last is None or last <= 0:
            return self._record(order, f"no price available for {symbol}")

        sign = 1 if side == "buy" else -1
        price = last * (1 + sign * self.slippage_rate)
        notional = price * order.quantity
        commission = notional * self.commission_rate

        with self._connect() as conn:
            cash = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()["cash"]
            pos = conn.execute(
                "SELECT quantity, avg_cost FROM positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            held = pos["quantity"] if pos else 0

            if side == "buy" and notional + commission > cash + 1e-9:
                return self._record(order, f"insufficient cash ({cash:.2f} available)", conn)
            if side == "sell" and order.quantity > held:
                return self._record(order, f"cannot sell {order.quantity}, holding {held}", conn)

            if side == "buy":
                new_qty = held + order.quantity
                avg = ((pos["avg_cost"] * held if pos else 0.0) + notional) / new_qty
                conn.execute(
                    "INSERT INTO positions (symbol, quantity, avg_cost) VALUES (?, ?, ?)"
                    " ON CONFLICT(symbol) DO UPDATE SET quantity = excluded.quantity,"
                    " avg_cost = excluded.avg_cost",
                    (symbol, new_qty, avg),
                )
                conn.execute(
                    "UPDATE account SET cash = cash - ? WHERE id = 1", (notional + commission,)
                )
            else:
                realized = (price - pos["avg_cost"]) * order.quantity - commission
                remaining = held - order.quantity
                if remaining == 0:
                    conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                else:
                    conn.execute(
                        "UPDATE positions SET quantity = ? WHERE symbol = ?", (remaining, symbol)
                    )
                conn.execute(
                    "UPDATE account SET cash = cash + ?, realized_pnl = realized_pnl + ?"
                    " WHERE id = 1",
                    (notional - commission, realized),
                )

            order.price, order.commission, order.status = round(price, 4), commission, "filled"
            return self._record(order, None, conn)

    def rebalance(
        self, target_weights: dict[str, float], min_trade_value: float = 50.0
    ) -> list[Order]:
        """Trade toward target weights of current equity. Sells run first to free cash."""
        if any(w < 0 for w in target_weights.values()):
            raise ValueError("long-only: target weights must be >= 0")
        if sum(target_weights.values()) > 1 + 1e-6:
            raise ValueError("target weights sum to more than 1")

        equity = self.account()["equity"]
        pos = self.positions()
        held_qty = {str(s): int(q) for s, q in zip(pos["symbol"], pos["quantity"])}
        symbols = set(target_weights) | set(held_qty)

        deltas = {}
        for sym in symbols:
            price = self.price_source(sym)
            if not price:
                continue
            held = held_qty.get(sym, 0)
            # Leave room for buy-side slippage and commission so the last buy isn't rejected.
            buffer = 1 + self.slippage_rate + self.commission_rate
            target_qty = math.floor(equity * target_weights.get(sym, 0.0) / (price * buffer))
            delta = target_qty - held
            if delta != 0 and abs(delta) * price >= min_trade_value:
                deltas[sym] = delta

        sells = [self.submit_order(s, "sell", -d) for s, d in deltas.items() if d < 0]
        buys = [self.submit_order(s, "buy", d) for s, d in sorted(deltas.items()) if d > 0]
        return sells + buys

    def _record(
        self, order: Order, reason: Optional[str], conn: Optional[sqlite3.Connection] = None
    ) -> Order:
        order.reason = reason
        own = conn is None
        conn = conn or self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO orders (timestamp, symbol, side, quantity, price, commission,"
                " status, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order.timestamp,
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.price,
                    order.commission,
                    order.status,
                    order.reason,
                ),
            )
            order.id = cur.lastrowid
            conn.commit()
        finally:
            if own:
                conn.close()
        return order


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
