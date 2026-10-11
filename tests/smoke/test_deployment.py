"""Smoke tests against a running deployment (a local container, staging or production).

Skipped unless QUANT_SMOKE_URL points at the API, e.g.
    QUANT_SMOKE_URL=http://localhost:8000 pytest tests/smoke
Set QUANT_SMOKE_DASHBOARD_URL to also check the Streamlit app.

They only read state (or send requests that are rejected), so they are safe to run
against production after every deploy.
"""

from __future__ import annotations

import os

import httpx
import pytest

API_URL = os.environ.get("QUANT_SMOKE_URL", "").rstrip("/")
DASHBOARD_URL = os.environ.get("QUANT_SMOKE_DASHBOARD_URL", "").rstrip("/")

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not API_URL, reason="QUANT_SMOKE_URL not set"),
]


@pytest.fixture(scope="module")
def api():
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        yield client


def test_health(api):
    r = api.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_lists_every_route(api):
    paths = set(api.get("/openapi.json").json()["paths"])
    assert {"/health", "/account", "/positions", "/orders", "/rebalance"} <= paths


def test_account_shape(api):
    acct = api.get("/account").json()
    assert {"cash", "equity", "market_value"} <= set(acct)
    assert acct["equity"] >= 0


def test_bundled_market_data_is_served(api):
    r = api.get("/prices/SPY")
    assert r.status_code == 200
    assert r.json()["price"] > 0


def test_allocator_runs_on_real_data(api):
    r = api.get("/allocations/risk_parity")
    assert r.status_code == 200
    weights = r.json()["weights"]
    assert len(weights) >= 2
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_invalid_order_is_rejected_without_side_effects(api):
    before = api.get("/orders", params={"limit": 1000}).json()
    bad = {"symbol": "SPY", "side": "short", "quantity": 1}
    assert api.post("/orders", json=bad).status_code == 422
    assert api.get("/orders", params={"limit": 1000}).json() == before


@pytest.mark.skipif(not DASHBOARD_URL, reason="QUANT_SMOKE_DASHBOARD_URL not set")
def test_dashboard_health():
    r = httpx.get(f"{DASHBOARD_URL}/_stcore/health", timeout=30)
    assert r.status_code == 200
    assert r.text.strip() == "ok"
