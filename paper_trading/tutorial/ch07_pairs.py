"""Chapter 7: mean reversion and pairs trading."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.strategies import MeanReversionPair, pair_spread_prices
from paper_trading.tutorial import common as c

SLUG = "pairs"


@st.cache_data(show_spinner=False)
def run_pair(a: str, b: str, lookback: int, entry: float, exit_z: float, bps: float):
    pa = c.ohlcv(a)[["close"]].rename(columns={"close": "close_a"})
    pb = c.ohlcv(b)[["close"]].rename(columns={"close": "close_b"})
    prices = pair_spread_prices(pa.join(pb, how="inner").reset_index())
    engine = BacktestEngine(BacktestConfig(100_000.0, bps * 2, bps * 2))  # two legs
    return engine.run(prices, MeanReversionPair(lookback, entry, exit_z)), prices


def render() -> None:
    c.chapter_header(SLUG)

    c.section("Trend following vs mean reversion")
    st.markdown(
        """
There are two opposite intuitions about prices:

| | Trend following | Mean reversion |
|---|---|---|
| Belief | moves continue | moves overshoot and come back |
| Buys | what has been rising | what has fallen "too far" |
| Example | the MA crossover (chapter 5) | pairs trading (this chapter) |

Both can work, in different markets and time frames. Neither works all the time.
"""
    )
    if not c.require_data():
        return

    c.section("Pairs trading")
    st.markdown(
        """
Take two companies that are economically similar, like **Coca-Cola (KO)** and
**PepsiCo (PEP)**. Their prices tend to move together because the same things affect
both (consumer spending, sugar prices). The **spread** is the gap between them.

When the spread becomes unusually wide, a pairs trader bets it will narrow again:
**short the one that's too expensive, buy the one that's too cheap**. Because you're
long one and short the other, a market-wide crash affects both legs roughly equally:
the bet is only on the *relative* move. This is called **market neutral**.

To measure "unusually wide", we use a **z-score**: how many standard deviations the
spread is from its recent average. The rule:

- z-score above **+entry** → spread too wide → short A, long B
- z-score below **-entry** → spread too narrow → long A, short B
- back within **±exit** → close the position
"""
    )
    universe = [s for s in c.all_symbols() if s != c.settings().benchmark]
    with c.try_it("Run a pairs trade"):
        col1, col2 = st.columns(2)
        a = col1.selectbox("Leg A", universe, index=universe.index("KO") if "KO" in universe else 0)
        b = col2.selectbox(
            "Leg B", universe, index=universe.index("PEP") if "PEP" in universe else 1
        )
        col3, col4, col5 = st.columns(3)
        lookback = col3.slider("Lookback (days)", 10, 120, 20, 5)
        entry = col4.slider("Entry z-score", 1.0, 3.0, 2.0, 0.25)
        exit_z = col5.slider("Exit z-score", 0.0, 1.5, 0.5, 0.25)
        if a == b:
            st.error("Pick two different tickers.")
        else:
            result, prices = run_pair(a, b, lookback, entry, exit_z, 5.0)
            spread = prices["close_a"] - prices["close_b"]
            z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
            view = (
                pd.DataFrame(
                    {"date": prices["date"], "z": z, "position": result.positions.to_numpy()}
                )
                .dropna()
                .iloc[-500:]
            )
            view["position"] = view["position"].apply(
                lambda p: "long spread" if p > 0 else ("short spread" if p < 0 else "flat")
            )
            zline = (
                alt.Chart(view)
                .mark_line(color="#455a64")
                .encode(x=alt.X("date:T", title=None), y=alt.Y("z:Q", title="Spread z-score"))
            )
            pos = (
                alt.Chart(view[view["position"] != "flat"])
                .mark_circle(size=14)
                .encode(
                    x="date:T",
                    y="z:Q",
                    color=alt.Color(
                        "position:N",
                        scale=alt.Scale(
                            domain=["long spread", "short spread"], range=["#2e7d32", "#c62828"]
                        ),
                        title=None,
                    ),
                )
            )
            bands = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "level": [entry, -entry, exit_z, -exit_z],
                            "kind": ["entry", "entry", "exit", "exit"],
                        }
                    )
                )
                .mark_rule(strokeDash=[4, 4])
                .encode(
                    y="level:Q",
                    color=alt.Color(
                        "kind:N", scale=alt.Scale(range=["#ef6c00", "#90a4ae"]), legend=None
                    ),
                )
            )
            st.altair_chart((zline + bands + pos).properties(height=300), width="stretch")
            st.caption(
                "Last ~2 years. Orange dashed lines are the entry thresholds, gray the exit. "
                "Colored dots mark days with an open position."
            )
            m = result.metrics
            cols = st.columns(4)
            cols[0].metric("Total return", c.fmt_pct(m["total_return"], signed=True))
            cols[1].metric("Sharpe", f"{m['sharpe']:.2f}")
            cols[2].metric("Max drawdown", c.fmt_pct(m["max_drawdown"]))
            cols[3].metric("Trades", len(result.trades))
            st.line_chart(result.equity.rename("Pairs strategy equity ($)"), height=220)
            st.caption("Costs are charged on both legs (5 bps commission + 5 bps slippage each).")

    c.section("Why it often fails")
    st.markdown(
        """
On KO/PEP with default settings this strategy **loses money**. That's a useful
result, not a bug:

- **The spread must actually come back.** Mean reversion only works if the gap is
  *stationary* (it keeps returning to a stable average). Two related stocks can drift
  apart for years when one company simply executes better. Statisticians test this
  with **cointegration** tests before trading a pair.
- **We use the raw price difference.** Professional versions hedge with a ratio
  estimated from data (e.g. 1 share of A vs 0.8 shares of B), or use log prices.
- **Costs bite twice.** Every entry and exit trades two stocks.
"""
    )
    with c.try_it("Do the two stocks still move together?"):
        rets = c.daily_returns(c.closes((a, b))).dropna() if a != b else None
        if rets is not None:
            rolling = rets[a].rolling(126).corr(rets[b]).dropna()
            st.line_chart(rolling.rename(f"6-month correlation {a}/{b}"), height=220)
            st.caption(
                "If the correlation of daily returns drifts lower, the pair is coming apart "
                "and the spread is less likely to snap back."
            )

    c.quiz(
        "pairs_1",
        "What does a pairs trade bet on?",
        [
            "That the whole market goes up",
            "That the gap between two related stocks returns to normal",
            "That both stocks go to zero",
            "That interest rates fall",
        ],
        answer=1,
        explanation="Being long one and short the other cancels most market moves, leaving "
        "a bet on the relative price.",
    )
    c.quiz(
        "pairs_2",
        "The spread's z-score is +2.5 and your entry threshold is 2. What does the rule do?",
        [
            "Buy A and sell B",
            "Short A and buy B, betting the spread narrows",
            "Do nothing",
            "Close all positions",
        ],
        answer=1,
        explanation="A is expensive relative to B, so you sell A and buy B.",
    )
    c.takeaways(
        [
            "Mean reversion bets that unusual moves come back; trend following bets they continue.",
            "Pairs trading is market neutral: long one stock, short a similar one.",
            "The z-score measures how unusual the spread is.",
            "It only works if the spread is truly stationary, and costs hit both legs.",
        ]
    )
    c.chapter_footer(SLUG, ["pairs_1", "pairs_2"])
