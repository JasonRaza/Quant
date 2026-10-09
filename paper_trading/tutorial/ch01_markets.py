"""Chapter 1: stocks, ETFs, prices and OHLCV bars."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from paper_trading.tutorial import common as c

SLUG = "markets"

COMPANIES = {
    "SPY": ("SPDR S&P 500 ETF", "Fund tracking the 500 largest US companies"),
    "AAPL": ("Apple", "Technology"),
    "MSFT": ("Microsoft", "Technology"),
    "GOOGL": ("Alphabet (Google)", "Technology / advertising"),
    "AMZN": ("Amazon", "E-commerce / cloud"),
    "META": ("Meta Platforms", "Social media / advertising"),
    "NVDA": ("Nvidia", "Semiconductors"),
    "JPM": ("JPMorgan Chase", "Banking"),
    "KO": ("Coca-Cola", "Beverages"),
    "PEP": ("PepsiCo", "Beverages / snacks"),
}


def candlestick(df: pd.DataFrame) -> alt.Chart:
    data = df.reset_index()
    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None),
        color=alt.condition(
            "datum.close >= datum.open", alt.value("#2e7d32"), alt.value("#c62828")
        ),
        tooltip=[
            alt.Tooltip("date:T"),
            alt.Tooltip("open:Q", format=",.2f"),
            alt.Tooltip("high:Q", format=",.2f"),
            alt.Tooltip("low:Q", format=",.2f"),
            alt.Tooltip("close:Q", format=",.2f"),
            alt.Tooltip("volume:Q", format=",.0f"),
        ],
    )
    wick = base.mark_rule().encode(
        y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="Price ($)"), y2="high:Q"
    )
    body = base.mark_bar(size=6).encode(y="open:Q", y2="close:Q")
    price = (wick + body).properties(height=300)
    volume = (
        base.mark_bar(size=6, opacity=0.6)
        .encode(y=alt.Y("volume:Q", title="Volume (shares)"))
        .properties(height=110)
    )
    return alt.vconcat(price, volume).resolve_scale(x="shared")


def render() -> None:
    c.chapter_header(SLUG)

    c.section("What is a stock?")
    st.markdown(
        """
A **stock** (or *share*) is a small piece of ownership in a company. If Apple has
about 15 billion shares and you own one, you own a tiny slice of Apple: its profits,
its factories, its brand.

Shares trade on an **exchange** (like the New York Stock Exchange or Nasdaq). All day,
buyers say what they're willing to pay and sellers say what they want to receive.
When a buyer and a seller agree, a trade happens, and that trade's price becomes "the
price" of the stock. Nobody sets prices; they emerge from millions of these trades.

Each stock is identified by a short code called a **ticker**. This project follows
these ten:
"""
    )
    st.dataframe(
        pd.DataFrame(
            [{"Ticker": t, "Name": n, "What it is": d} for t, (n, d) in COMPANIES.items()],
        ),
        hide_index=True,
        width="stretch",
    )
    c.term(
        "ETF and index",
        "An **index** is a list of stocks with a rule for weighting them, like the "
        "S&P 500 (the 500 largest US companies, weighted by size). You can't buy an index "
        "directly, but you can buy an **ETF** (exchange-traded fund) like **SPY**, which "
        "holds all those stocks for you and trades like a single stock. SPY is our "
        '**benchmark**: the "do nothing clever" option every strategy must beat.',
    )

    c.section("Reading a price chart: OHLCV bars")
    st.markdown(
        """
Prices change thousands of times a day. To keep data manageable we summarize each
day into one **bar** with five numbers, called **OHLCV**:

| Field | Meaning |
|---|---|
| **O**pen | price of the first trade of the day |
| **H**igh | highest price during the day |
| **L**ow | lowest price during the day |
| **C**lose | price of the last trade of the day (the one everybody quotes) |
| **V**olume | number of shares that changed hands that day |

A **candlestick** draws one bar per day: the thick *body* goes from open to close,
the thin *wick* from low to high. It's **green** when the day closed higher than it
opened and **red** when it closed lower.
"""
    )

    if not c.require_data():
        return

    with c.try_it("Explore real candlesticks"):
        col1, col2 = st.columns([1, 2])
        symbol = col1.selectbox(
            "Ticker",
            c.all_symbols(),
            format_func=lambda t: f"{t} · {COMPANIES.get(t, (t,))[0]}",
            key="ch1_symbol",
        )
        df = c.ohlcv(symbol)
        months = col2.select_slider(
            "How much history?", options=[1, 3, 6, 12], value=3, format_func=lambda m: f"{m} months"
        )
        window = df[df.index >= df.index[-1] - pd.DateOffset(months=months)]
        st.altair_chart(candlestick(window), width="stretch")

        last = window.iloc[-1]
        st.markdown(f"**The most recent bar ({window.index[-1].date()})**")
        cols = st.columns(5)
        for col, field in zip(cols, ["open", "high", "low", "close", "volume"]):
            fmt = "{:,.0f}" if field == "volume" else "${:,.2f}"
            col.metric(field.capitalize(), fmt.format(last[field]))
        up = last["close"] >= last["open"]
        st.markdown(
            f"That day {symbol} opened at **${last['open']:,.2f}** and closed at "
            f"**${last['close']:,.2f}**, so the candle is **{'green' if up else 'red'}**. "
            f"About **{last['volume'] / 1e6:,.1f} million** shares traded. Hover over the "
            "chart to inspect any day."
        )

    c.section("Daily data is enough to start")
    st.markdown(
        """
Professional high-frequency traders look at every single trade, down to microseconds.
This project uses **daily bars** ("end-of-day data") instead: one OHLCV row per stock
per day. It's free, small (about 1,700 rows per stock since 2020) and perfectly suited
to strategies that hold positions for days or months, which is most of quant investing.

Under the hood, each stock's history is stored as a table like this one (the most
recent rows):
"""
    )
    st.dataframe(df.tail(5).round(2), width="stretch")
    c.key_idea(
        "A price series is just a table: one row per day, columns for open, high, low, "
        "close and volume. Everything else in this tutorial is computed from tables like it."
    )

    c.quiz(
        "markets_1",
        "On a candlestick chart, a red candle means…",
        [
            "The stock lost money for the year",
            "The close was below the open that day",
            "Volume was unusually high",
            "The stock hit a new low",
        ],
        answer=1,
        explanation="Color only compares that day's close to that day's open.",
    )
    c.quiz(
        "markets_2",
        "Why is SPY used as a benchmark?",
        [
            "It's the most volatile stock",
            "It represents simply buying the whole US market, the default alternative",
            "It always goes up",
            "It has the highest price",
        ],
        answer=1,
        explanation="If a clever strategy can't beat just holding the market, there's no "
        "reason to use it.",
    )
    c.takeaways(
        [
            "A stock is a share of a company; its price comes from trades between buyers "
            "and sellers.",
            "Daily OHLCV bars summarize each day: open, high, low, close, volume.",
            "SPY (an ETF holding the S&P 500) is the benchmark to beat.",
        ]
    )
    c.chapter_footer(SLUG, ["markets_1", "markets_2"])
