"""Chapter 5: trading rules, moving averages and how a backtest works."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.strategies import DualMovingAverageCrossover
from paper_trading.tutorial import common as c

SLUG = "backtesting"


@st.cache_data(show_spinner=False)
def run_dual_ma(symbol: str, fast: int, slow: int, bps: float):
    prices = c.ohlcv(symbol).reset_index()
    engine = BacktestEngine(BacktestConfig(100_000.0, commission_bps=bps, slippage_bps=bps))
    return engine.run(prices, DualMovingAverageCrossover(fast, slow))


def render() -> None:
    c.chapter_header(SLUG)

    c.section("What is a trading strategy?")
    st.markdown(
        """
A **trading strategy** is a precise rule that turns data into a decision. In this
project every strategy outputs a **signal** for each day:

- **+1** → be fully invested (*long*: you own the stock and profit if it rises)
- **0** → stay in cash
- **-1** → be *short* (you borrow the stock and sell it, profiting if it falls)

The rule must be written so precisely that a computer can follow it with no judgment.
That precision is what lets us **test** it on history.
"""
    )
    if not c.require_data():
        return

    c.section("Moving averages: smoothing out the noise")
    st.markdown(
        """
A **moving average** (MA) is the average closing price over the last N days,
recalculated every day. It smooths out daily noise and shows the underlying trend. A
short MA (e.g. 20 days) reacts quickly; a long MA (e.g. 200 days) moves slowly.
"""
    )
    bench = c.settings().benchmark
    with c.try_it("Play with the averaging window"):
        symbol = st.selectbox("Ticker", c.all_symbols(), key="ch5_ma_symbol")
        window = st.slider("Moving average window (days)", 5, 250, 50, 5)
        close = c.ohlcv(symbol)["close"]
        df = pd.DataFrame({"Price": close, f"{window}-day MA": close.rolling(window).mean()})
        st.line_chart(df.iloc[-750:], height=300)
        st.caption("Last ~3 years. The longer the window, the smoother and more delayed the line.")

    c.section("The dual moving average crossover")
    st.markdown(
        """
Our first strategy is a classic **trend-following** rule:

> **Be invested when the fast MA is above the slow MA; otherwise stay in cash.**

The idea: when recent prices (fast MA) are above the longer-term average (slow MA),
the market is trending up, so ride it. When the trend turns down, step aside and avoid
the worst of the falls.

To avoid cheating, today's decision only uses information up to **yesterday's close**,
and the trade happens at today's close. (Chapter 6 shows what goes wrong otherwise.)
"""
    )
    with c.try_it("Build the strategy step by step"):
        col1, col2, col3 = st.columns(3)
        symbol = col1.selectbox("Ticker", c.all_symbols(), key="ch5_bt_symbol")
        fast = col2.slider("Fast MA", 5, 100, 20, 5, key="ch5_fast")
        slow = col3.slider("Slow MA", 20, 250, 50, 10, key="ch5_slow")
        if fast >= slow:
            st.error("The fast window must be shorter than the slow one.")
        else:
            close = c.ohlcv(symbol)["close"]
            fast_ma, slow_ma = close.rolling(fast).mean(), close.rolling(slow).mean()
            signal = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
            view = pd.DataFrame(
                {"price": close, "fast": fast_ma, "slow": slow_ma, "in_market": signal}
            ).iloc[-500:]
            data = view.reset_index()
            lo, hi = float(view["price"].min()) * 0.97, float(view["price"].max()) * 1.03
            shade = (
                alt.Chart(data)
                .transform_filter("datum.in_market == 1")
                .transform_calculate(lo=str(lo), hi=str(hi))
                .mark_rule(opacity=0.15, color="#15803D", strokeWidth=3)
                .encode(
                    x="date:T",
                    y=alt.Y("lo:Q", scale=alt.Scale(domain=[lo, hi]), title="Price ($)"),
                    y2="hi:Q",
                )
            )
            lines = (
                alt.Chart(data)
                .transform_fold(["price", "fast", "slow"], as_=["series", "value"])
                .mark_line()
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("value:Q", scale=alt.Scale(domain=[lo, hi]), title="Price ($)"),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=["price", "fast", "slow"],
                            range=["#9CA3AF", "#0F766E", "#D97706"],
                        ),
                        title=None,
                    ),
                )
            )
            st.altair_chart((shade + lines).properties(height=320), width="stretch")
            st.caption(
                "Green shading = days the strategy is invested (last ~2 years). Gray is the "
                "price, blue the fast MA, orange the slow MA."
            )

            result = run_dual_ma(symbol, fast, slow, 5.0)
            m = result.metrics
            st.markdown(
                c.esc(
                    "**Backtest over the whole history "
                    "($100,000, 5 bps commission + 5 bps slippage)**"
                )
            )
            cols = st.columns(4)
            cols[0].metric(
                "Total return",
                c.fmt_pct(m["total_return"]),
                f"{m['total_return'] - m['benchmark_total_return']:+.1%} vs buy & hold",
            )
            cols[1].metric(
                "Sharpe", f"{m['sharpe']:.2f}", f"{m['sharpe'] - m['benchmark_sharpe']:+.2f}"
            )
            cols[2].metric(
                "Max drawdown",
                c.fmt_pct(m["max_drawdown"]),
                f"{m['max_drawdown'] - m['benchmark_max_drawdown']:+.1%}",
            )
            cols[3].metric("Trades", len(result.trades))
            curves = pd.DataFrame(
                {"Strategy": result.equity, "Buy & hold": result.benchmark_equity}
            )
            st.line_chart(curves, height=280)

    c.section("What the backtest engine actually does")
    st.markdown(
        """
A **backtest** replays history day by day as if you had followed the rule. Our
engine (`backtester/engine.py`) does this every day:

1. Compute the account value: cash + shares × today's close.
2. If the signal changed, compute the target number of shares.
3. Trade the difference at today's close, made slightly worse by **slippage**
   (you rarely get the exact quoted price), and pay a **commission**.
4. Record the account value.

At the end it computes the metrics you know from chapters 2 and 3 and compares them
to simply buying and holding. Here are the first trades of the backtest above:
"""
    )
    if fast < slow:
        trades = result.trades.copy()
        trades["action"] = trades["shares"].map(lambda s: "BUY" if s > 0 else "SELL")
        trades = trades[["date", "action", "shares", "price", "trade_value", "cost"]].head(8)
        st.dataframe(
            trades.style.format(
                {
                    "shares": "{:,.1f}",
                    "price": "${:,.2f}",
                    "trade_value": "${:,.0f}",
                    "cost": "${:,.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "trade_value is negative for sales (cash comes in). cost = commission + slippage."
        )

    c.section("Transaction costs add up")
    st.markdown(
        """
Every trade costs a little: brokerage **commission**, the **bid-ask spread** (buyers pay
slightly more than sellers receive) and **market impact** (big orders move the price).
A strategy that trades often can see its profits eaten entirely.
"""
    )
    with c.try_it("Turn up the costs"):
        bps = st.slider(
            "Cost per side (basis points; 1 bp = 0.01%)", 0.0, 50.0, 5.0, 2.5, key="ch5_bps"
        )
        rows = []
        for f, s in [(5, 20), (20, 50), (50, 200)]:
            r = run_dual_ma(bench, f, s, bps)
            rows.append(
                {
                    "strategy": f"MA {f}/{s}",
                    "trades": len(r.trades),
                    "total return": r.metrics["total_return"],
                    "costs paid": r.trades["cost"].sum() if len(r.trades) else 0.0,
                }
            )
        st.dataframe(
            pd.DataFrame(rows).style.format({"total return": "{:+.1%}", "costs paid": "${:,.0f}"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"All on {bench}. Fast-reacting rules trade more, so they suffer most as costs rise."
        )

    c.quiz(
        "bt_1",
        "In the dual MA strategy, when is the strategy invested?",
        [
            "When the price is above zero",
            "When the fast moving average is above the slow one",
            "Every Monday",
            "When volume is high",
        ],
        answer=1,
        explanation="Fast above slow means recent prices are higher than the longer-term "
        "trend, the definition of an uptrend for this rule.",
    )
    c.quiz(
        "bt_2",
        "A strategy makes 200 trades a year. What happens if costs double?",
        [
            "Nothing, costs are tiny",
            "Its returns drop noticeably, because costs are paid on every trade",
            "Its returns rise",
            "Only the Sharpe ratio changes",
        ],
        answer=1,
        explanation="Costs scale with turnover; high-frequency rules are the most sensitive.",
    )
    c.takeaways(
        [
            "A strategy is a precise rule that outputs a position signal every day.",
            "Moving averages smooth prices; the crossover rule follows trends.",
            "A backtest replays history day by day with realistic costs.",
            "Always compare against buy & hold, and watch how many trades you make.",
        ]
    )
    c.chapter_footer(SLUG, ["bt_1", "bt_2"])
