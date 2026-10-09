"""Chapter 11: from research to trading: orders, costs and paper trading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from paper_trading.broker import PaperBroker, ParquetPriceSource
from paper_trading.tutorial import common as c

SLUG = "trading"


def sandbox_broker() -> PaperBroker:
    """A throwaway account per browser session, separate from the Lab's paper account."""
    if "sandbox_db" not in st.session_state:
        tmp = Path(tempfile.mkdtemp(prefix="quant_tutorial_"))
        st.session_state["sandbox_db"] = str(tmp / "sandbox.sqlite")
    return PaperBroker(
        st.session_state["sandbox_db"],
        ParquetPriceSource(c.settings().data_dir),
        initial_cash=10_000.0,
    )


def render() -> None:
    c.chapter_header(SLUG)

    c.section("How orders actually get filled")
    st.markdown(
        """
A backtest says "buy 10 shares at the close". In reality you send an **order** to a
broker, which routes it to an exchange.

- **Market order**: "buy now at whatever price is available". Always fills, price not
  guaranteed.
- **Limit order**: "buy only at \\$100 or less". Price guaranteed, filling isn't.

At any moment the market shows a **bid** (the highest price someone will pay) and an
**ask** (the lowest price someone will sell at). The ask is always a bit higher. Buy
with a market order and you pay the ask; sell and you get the bid. That gap, the
**spread**, plus **commissions** and **market impact** (large orders push the price
against you), is why every backtest in this project charges costs on each trade.
"""
    )

    c.section("Paper trading: rehearsal with fake money")
    st.markdown(
        """
Between a backtest and real money sits **paper trading**: you run the strategy live,
on today's prices, but with a simulated account. It catches problems a backtest can't:
data that arrives late, code that crashes on a holiday, orders that are too small to
fill.

This project includes a **paper broker** (`paper_trading/broker.py`): a simulated
cash account stored in a database. Orders fill at the latest close in the local data,
plus 5 bps slippage and 5 bps commission. It refuses orders you couldn't afford and
won't let you sell shares you don't own. Try it below, in a **private sandbox account**
(\\$10,000, separate from the Lab's paper account).
"""
    )
    if not c.require_data():
        return

    broker = sandbox_broker()
    with c.try_it("Your sandbox account"):
        acct = broker.account()
        m = st.columns(4)
        m[0].metric(
            "Equity", c.fmt_usd(acct["equity"]), c.fmt_pct(acct["total_return"], signed=True)
        )
        m[1].metric("Cash", c.fmt_usd(acct["cash"]))
        m[2].metric("Unrealized P&L", c.fmt_usd(acct["unrealized_pnl"]))
        m[3].metric("Realized P&L", c.fmt_usd(acct["realized_pnl"]))

        flash = st.session_state.pop("sandbox_flash", None)
        if flash:
            getattr(st, flash[0])(flash[1])

        with st.form("sandbox_order"):
            col1, col2, col3 = st.columns(3)
            sym = col1.selectbox("Ticker", c.all_symbols())
            side = col2.radio("Side", ["buy", "sell"], horizontal=True)
            qty = col3.number_input("Shares", min_value=1, value=5, step=1)
            price = broker.price_source(sym)
            if price:
                st.caption(
                    c.esc(
                        f"Latest close for {sym}: ${price:,.2f} → a buy would fill near "
                        f"${price * 1.0005:,.2f}, plus a commission of about "
                        f"${price * 1.0005 * qty * 0.0005:,.2f}."
                    )
                )
            if st.form_submit_button("Send order"):
                order = broker.submit_order(sym, side, int(qty))
                if order.status == "filled":
                    st.session_state["sandbox_flash"] = (
                        "success",
                        c.esc(
                            f"Filled: {side} {order.quantity} {sym} @ ${order.price:,.2f} "
                            f"(commission ${order.commission:,.2f})"
                        ),
                    )
                else:
                    st.session_state["sandbox_flash"] = ("error", f"Rejected: {order.reason}")
                st.rerun()

        pos = broker.positions()
        if pos.empty:
            st.info("No positions yet. Buy something above, then try selling more than you own.")
        else:
            st.dataframe(pos.round(2), hide_index=True, width="stretch")
        orders = broker.orders(20)
        if not orders.empty:
            with st.expander("Order log (including rejected orders)"):
                st.dataframe(orders, hide_index=True, width="stretch")
        if st.button("Reset sandbox to $10,000"):
            broker.reset(10_000.0)
            st.rerun()

    c.section("One-click portfolio rebalancing")
    st.markdown(
        """
In the Lab's **Paper trading** tab you can pick any allocator from chapter 9 and press
*Rebalance*: the broker computes today's target weights, **sells first** to free up
cash, then buys, sizing every order so costs never overdraw the account. That's the
same loop a real systematic fund runs every day, minus the real money.

The project also exposes the account through a **REST API** (`quant-paper-api`, docs at
`http://127.0.0.1:8000/docs`) so other programs can trade it, which is how production
trading systems are usually wired together.
"""
    )

    c.section("From paper to production: a checklist")
    st.markdown(
        """
Before real money, a professional team would want all of this:

| Area | What it means |
|---|---|
| **Research** | Out-of-sample, walk-forward, realistic costs; a count of all tries |
| **Data** | Reliable vendor, point-in-time universe, gap and staleness alerts |
| **Risk limits** | Max position size, leverage and daily loss; a kill switch |
| **Execution** | Limit orders or smart routing; compare fills with the backtest |
| **Monitoring** | Alerts when live results drift; daily broker reconciliation |
| **Engineering** | Tests (80+ here), version control, reproducible builds, logs |

The gap between a good backtest and a profitable live strategy is where most quant
projects fail.
"""
    )
    c.warning_box(
        "Nothing in this project connects to a real broker or real money, and that's on purpose."
    )

    c.quiz(
        "trade_1",
        "You buy with a market order. Which price do you typically pay?",
        ["The bid", "The ask (slightly higher)", "The day's low", "Yesterday's close"],
        answer=1,
        explanation="Buyers pay the ask, sellers receive the bid; the spread is a cost.",
    )
    c.quiz(
        "trade_2",
        "What is paper trading for?",
        [
            "Trading newspaper companies",
            "Running a strategy live with simulated money to catch real-world problems "
            "before risking capital",
            "Avoiding taxes",
            "Replacing backtests entirely",
        ],
        answer=1,
        explanation="It sits between backtesting and live trading.",
    )
    c.takeaways(
        [
            "Market orders fill for sure at an uncertain price; limit orders the opposite.",
            "Spread, commission and market impact are real costs on every trade.",
            "Paper trading runs the strategy live with fake money to catch practical issues.",
            "Production adds risk limits, monitoring and engineering discipline.",
        ]
    )
    c.chapter_footer(SLUG, ["trade_1", "trade_2"])
