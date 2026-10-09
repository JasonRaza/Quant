"""Tests for the paper trading REST API and allocation service."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_pipeline.storage import ParquetStore
from paper_trading.api import create_app
from paper_trading.service import ALLOCATION_METHODS, Settings, target_weights

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def settings(tmp_path) -> Settings:
    store = ParquetStore(tmp_path / "parquet")
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=320)
    for i, sym in enumerate(SYMBOLS):
        close = 50 * (i + 1) * np.exp(np.cumsum(rng.normal(0.0005, 0.01 * (i + 1), len(dates))))
        store.save(
            sym,
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates.date,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000_000,
                }
            ),
        )
    return Settings(
        data_dir=tmp_path / "parquet", paper_db=tmp_path / "paper.sqlite", universe=SYMBOLS
    )


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.mark.parametrize("method", ALLOCATION_METHODS)
def test_target_weights_all_methods(settings, method):
    w = target_weights(settings, method, max_weight=0.4)
    assert w.sum() == pytest.approx(1.0)
    assert set(w.index) == set(SYMBOLS)
    if method != "risk_parity":
        assert (w <= 0.4 + 1e-6).all()


def test_target_weights_unknown_method(settings):
    with pytest.raises(ValueError):
        target_weights(settings, "yolo")


def test_account_and_order_flow(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/account").json()["cash"] == 100_000.0

    r = client.post("/orders", json={"symbol": "aaa", "side": "buy", "quantity": 10})
    assert r.status_code == 200 and r.json()["status"] == "filled"

    positions = client.get("/positions").json()
    assert positions[0]["symbol"] == "AAA" and positions[0]["quantity"] == 10
    assert len(client.get("/orders").json()) == 1


def test_rejected_order_returns_400(client):
    r = client.post("/orders", json={"symbol": "AAA", "side": "sell", "quantity": 5})
    assert r.status_code == 400
    assert "cannot sell" in r.json()["detail"]["reason"]


@pytest.mark.parametrize(
    "body",
    [
        {"symbol": "AAA", "side": "short", "quantity": 1},
        {"symbol": "AAA", "side": "buy", "quantity": 0},
        {"symbol": "", "side": "buy", "quantity": 1},
    ],
)
def test_request_validation(client, body):
    assert client.post("/orders", json=body).status_code == 422


def test_prices(client):
    assert client.get("/prices/aaa").json()["symbol"] == "AAA"
    assert client.get("/prices/ZZZ").status_code == 404


def test_allocation_and_rebalance(client):
    alloc = client.get("/allocations/min_variance", params={"max_weight": 0.4}).json()
    assert sum(alloc["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert client.get("/allocations/yolo").status_code == 400

    r = client.post("/rebalance", json={"method": "equal_weight", "max_weight": 0.4}).json()
    assert r["orders"] and all(o["status"] == "filled" for o in r["orders"])
    acct = r["account"]
    assert acct["market_value"] / acct["equity"] > 0.95


def test_reset(client):
    client.post("/orders", json={"symbol": "AAA", "side": "buy", "quantity": 1})
    assert client.post("/reset", json={"initial_cash": 5000}).json()["cash"] == 5000.0
    assert client.get("/positions").json() == []
