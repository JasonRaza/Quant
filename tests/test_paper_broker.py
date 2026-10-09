"""Tests for the simulated paper broker."""

import pytest

from paper_trading.broker import PaperBroker

PRICES = {"AAA": 100.0, "BBB": 50.0}


@pytest.fixture
def prices():
    return dict(PRICES)


@pytest.fixture
def broker(tmp_path, prices):
    return PaperBroker(
        tmp_path / "paper.sqlite",
        price_source=prices.get,
        initial_cash=10_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    )


def test_buy_updates_cash_and_position(broker):
    order = broker.submit_order("aaa", "BUY", 10)
    assert order.status == "filled" and order.symbol == "AAA"
    acct = broker.account()
    assert acct["cash"] == 9_000.0
    assert acct["equity"] == 10_000.0
    pos = broker.positions().set_index("symbol")
    assert pos.loc["AAA", "quantity"] == 10


def test_costs_are_charged(tmp_path, prices):
    b = PaperBroker(tmp_path / "c.sqlite", prices.get, 10_000.0, commission_bps=10, slippage_bps=10)
    order = b.submit_order("AAA", "buy", 10)
    assert order.price == pytest.approx(100.1)
    assert order.commission == pytest.approx(1001.0 * 0.001)
    assert b.account()["cash"] == pytest.approx(10_000 - 1001.0 - 1.001, abs=0.01)


@pytest.mark.parametrize(
    "symbol,side,qty,reason",
    [
        ("AAA", "buy", 1000, "insufficient cash"),
        ("AAA", "sell", 1, "cannot sell"),
        ("ZZZ", "buy", 1, "no price"),
        ("AAA", "short", 1, "invalid side"),
        ("AAA", "buy", 0, "positive whole number"),
        ("AAA", "buy", 1.5, "positive whole number"),
    ],
)
def test_rejections_are_logged_and_change_nothing(broker, symbol, side, qty, reason):
    order = broker.submit_order(symbol, side, qty)
    assert order.status == "rejected" and reason in order.reason
    assert broker.account()["cash"] == 10_000.0
    assert broker.orders().iloc[0]["status"] == "rejected"


def test_realized_and_unrealized_pnl(broker, prices):
    broker.submit_order("AAA", "buy", 10)
    prices["AAA"] = 120.0
    assert broker.account()["unrealized_pnl"] == 200.0
    broker.submit_order("AAA", "sell", 4)
    acct = broker.account()
    assert acct["realized_pnl"] == 80.0
    assert acct["unrealized_pnl"] == 120.0
    assert acct["equity"] == 10_200.0


def test_selling_everything_closes_position(broker):
    broker.submit_order("AAA", "buy", 5)
    broker.submit_order("AAA", "sell", 5)
    assert broker.positions().empty


def test_state_persists_across_instances(broker, tmp_path, prices):
    broker.submit_order("BBB", "buy", 20)
    again = PaperBroker(tmp_path / "paper.sqlite", prices.get, initial_cash=999.0)
    assert again.account()["cash"] == 9_000.0  # existing account isn't overwritten


def test_rebalance_hits_targets_and_sells_first(broker):
    broker.submit_order("AAA", "buy", 80)  # 80% in AAA
    orders = broker.rebalance({"AAA": 0.25, "BBB": 0.75})
    assert [o.side for o in orders] == ["sell", "buy"]
    assert all(o.status == "filled" for o in orders)
    pos = broker.positions().set_index("symbol")
    assert pos.loc["AAA", "quantity"] == 25
    assert pos.loc["BBB", "quantity"] == 150


def test_rebalance_with_costs_never_overspends(tmp_path, prices):
    b = PaperBroker(tmp_path / "c.sqlite", prices.get, 10_000.0, commission_bps=10, slippage_bps=10)
    orders = b.rebalance({"AAA": 0.5, "BBB": 0.5})
    assert all(o.status == "filled" for o in orders)
    assert b.account()["cash"] >= 0


def test_rebalance_validates_weights(broker):
    with pytest.raises(ValueError):
        broker.rebalance({"AAA": -0.1})
    with pytest.raises(ValueError):
        broker.rebalance({"AAA": 0.8, "BBB": 0.8})


def test_reset(broker):
    broker.submit_order("AAA", "buy", 10)
    broker.reset(50_000.0)
    assert broker.account()["cash"] == 50_000.0
    assert broker.positions().empty and broker.orders().empty
