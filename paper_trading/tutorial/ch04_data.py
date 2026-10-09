"""Chapter 4: where market data comes from and how it can mislead you."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from paper_trading.tutorial import common as c

SLUG = "data"

PIPELINE_DOT = """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  yahoo [label="Yahoo Finance\\n(free daily bars)"];
  fetch [label="fetcher.py\\ndownload + retries"];
  norm  [label="normalize\\nOHLCV schema"];
  pq    [label="Parquet files\\none per ticker"];
  meta  [label="SQLite metadata\\nlast date, row counts"];
  qa    [label="quality.py\\ngaps, stale data"];
  research [label="Backtests, factors,\\nportfolios, ML", fillcolor="#e8f5e9"];
  yahoo -> fetch -> norm -> pq -> research;
  fetch -> meta; pq -> qa;
}
"""


def render() -> None:
    c.chapter_header(SLUG)

    c.section("The data pipeline")
    st.markdown(
        """
Every quant result is only as good as the data behind it. Professional funds pay
millions per year for clean data; we use the free daily bars from **Yahoo Finance**,
which is fine for learning but has quirks you must know about.

This project's first component is an **ETL pipeline** (Extract, Transform, Load):
"""
    )
    st.graphviz_chart(PIPELINE_DOT)
    st.markdown(
        """
- **Extract**: download each ticker's daily bars, retrying if the network fails.
- **Transform**: rename columns to one standard schema
  (`symbol, date, open, high, low, close, volume`), sort, drop duplicates.
- **Load**: save one **Parquet** file per ticker (a compact, fast table format) and
  record in a small **SQLite** database the last date downloaded.

Because of that metadata, re-running the pipeline only downloads the **new** days. This
property is called being **idempotent and incremental**: running it twice in a row
changes nothing, and running it tomorrow only adds tomorrow.
"""
    )

    if not c.require_data():
        return

    c.section("Splits and dividends: adjusted prices")
    st.markdown(
        """
Companies sometimes **split** their shares: in June 2024 Nvidia turned every share
into ten, so the price per share dropped about 90% overnight even though nobody lost
any money (you simply held ten times as many shares). Companies also pay
**dividends**, cash that leaves the share price but lands in your pocket.

If you fed raw prices into a backtest, a split would look like a catastrophic crash.
That's why we download **adjusted prices**: history is rescaled so that every day's
change reflects what a shareholder actually experienced.
"""
    )
    prices = c.closes()
    if "NVDA" in prices.columns:
        with c.try_it("See what a split does to raw prices"):
            nvda = prices["NVDA"].dropna()
            window = nvda["2024-03-01":"2024-09-30"]
            if window.empty:
                st.caption("Nvidia data for 2024 isn't available in your local store.")
            else:
                split_date = pd.Timestamp("2024-06-10")
                raw = window.where(window.index >= split_date, window * 10)
                df = pd.DataFrame({"Adjusted (what we store)": window, "Raw, as quoted": raw})
                long = df.reset_index().melt("date", var_name="series", value_name="price")
                chart = (
                    alt.Chart(long)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title=None),
                        y=alt.Y("price:Q", title="Price ($)"),
                        color=alt.Color("series:N", title=None),
                    )
                    .properties(height=300)
                )
                rule = (
                    alt.Chart(pd.DataFrame({"date": [split_date]}))
                    .mark_rule(strokeDash=[4, 4])
                    .encode(x="date:T")
                )
                st.altair_chart(chart + rule, width="stretch")
                st.caption(
                    "The raw series is reconstructed for illustration by multiplying pre-split "
                    "prices by 10 (ignoring small dividend adjustments). A naive backtest on it "
                    "would record a -90% day on June 10, 2024 that never happened to investors."
                )
        c.warning_box(
            "Subtle trap: when a new dividend is paid, Yahoo rescales the whole adjusted "
            "history. An incremental pipeline that only appends new days can drift slightly "
            "out of sync, so a full re-download every so often is good hygiene."
        )

    c.section("Gaps, holidays and bad ticks")
    st.markdown(
        """
Markets are closed on weekends and holidays, so a year has about **252** trading days,
not 365. Data can also have genuine problems: missing days, stale prices that stopped
updating, or one-off errors ("bad ticks"). The pipeline's quality report checks for
these. Here's a quick health check of your local data:
"""
    )
    with c.try_it("Data health check"):
        calendar = prices.index
        rows = []
        for sym in prices.columns:
            s = prices[sym]
            r = s.pct_change(fill_method=None)
            rows.append(
                {
                    "ticker": sym,
                    "first day": s.first_valid_index().date(),
                    "last day": s.last_valid_index().date(),
                    "rows": int(s.notna().sum()),
                    "missing vs calendar": int(s.isna().sum()),
                    "biggest daily move": r.abs().max(),
                    "on": r.abs().idxmax().date(),
                }
            )
        st.dataframe(
            pd.DataFrame(rows).style.format({"biggest daily move": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )
        per_year = pd.Series(calendar.year).value_counts().sort_index()
        st.markdown(
            "Trading days per year in the data: "
            + ", ".join(f"{y}: **{n}**" for y, n in per_year.items())
            + ". The current year is partial."
        )
        st.caption(
            "Big daily moves usually have a real cause (earnings, a crash, a market panic in "
            "March 2020), but a 50%+ move on a large company would be worth double-checking."
        )

    c.section("Survivorship bias: the stocks you don't see")
    st.markdown(
        """
Our universe is ten famous companies that are **still around and successful today**.
That's a problem: in 2020, nobody knew Nvidia would grow so much. Picking today's
winners and testing a strategy on their past is like judging a lottery by interviewing
only the winners.

Real research uses **point-in-time** universes: the list of stocks that existed and
were tradable *on each date*, including the ones that later went bankrupt or were
delisted.
"""
    )
    with c.try_it("Simulate survivorship bias"):
        col1, col2 = st.columns(2)
        n = col1.slider("Number of random companies", 50, 1000, 300, 50)
        fail = col2.slider("Share that go bust over 5 years", 0, 60, 20, 5, format="%d%%")
        rng = np.random.default_rng(42)
        log_rets = rng.normal(0.05, 0.35, n) * np.sqrt(5)
        total = np.exp(log_rets) - 1
        cutoff = np.quantile(total, fail / 100) if fail else -np.inf
        dead = total <= cutoff
        total[dead] = -1.0  # bankrupt companies lose everything
        m = st.columns(3)
        m[0].metric("Average return, all companies", c.fmt_pct(total.mean(), signed=True))
        m[1].metric("Average return, survivors only", c.fmt_pct(total[~dead].mean(), signed=True))
        m[2].metric("Companies that disappeared", f"{dead.sum()}")
        st.markdown(
            "If your database only contains companies that survived, every strategy looks "
            "better than it really was, because the worst outcomes were silently deleted."
        )

    c.quiz(
        "data_1",
        "Why do we use adjusted prices?",
        [
            "They're more recent",
            "So splits and dividends don't show up as fake gains or losses",
            "Because raw prices are illegal to use",
            "To make the stock look better",
        ],
        answer=1,
        explanation="Adjusted prices make every day's change match what a shareholder "
        "actually experienced.",
    )
    c.quiz(
        "data_2",
        "You test a strategy only on today's 10 biggest companies, starting in 2010. What's "
        "the main risk?",
        [
            "Too much data",
            "Survivorship bias: you picked winners with hindsight",
            "Lookahead bias in the prices",
            "None, big companies are safe",
        ],
        answer=1,
        explanation="In 2010 you couldn't have known which companies would be the biggest "
        "in 2026; the test quietly uses that knowledge.",
    )
    c.takeaways(
        [
            "A pipeline downloads, cleans and stores data; ours is incremental and idempotent.",
            "Use split- and dividend-adjusted prices for returns.",
            "Check data quality: gaps, stale series, suspicious jumps.",
            "Survivorship bias makes backtests on today's winners look too good.",
        ]
    )
    c.chapter_footer(SLUG, ["data_1", "data_2"])
