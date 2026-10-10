"""Chapter 2: returns, compounding, volatility and drawdowns."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from paper_trading.tutorial import common as c

SLUG = "returns-risk"


def render() -> None:
    c.chapter_header(SLUG)

    c.section("From prices to returns")
    st.markdown(
        r"""
A price on its own says little: is \$300 expensive? It depends on the stock. What
investors care about is the **return**, the percentage change in value:

$$\text{return} = \frac{\text{price}_\text{end}}{\text{price}_\text{start}} - 1$$

A stock going from \$100 to \$110 has a return of **+10%**. Returns let you compare a
\$5 stock with a \$700 one on equal terms.
"""
    )
    if not c.require_data():
        return

    prices = c.closes()
    with c.try_it("Invest some money in the past"):
        col1, col2, col3 = st.columns(3)
        symbol = col1.selectbox("Ticker", list(prices.columns), key="ch2_symbol")
        amount = col2.number_input("Amount ($)", 100, 1_000_000, 10_000, step=1_000)
        start = col3.date_input(
            "Buy on",
            value=prices.index[0].date(),
            min_value=prices.index[0].date(),
            max_value=(prices.index[-1] - pd.Timedelta(days=30)).date(),
        )
        series = prices[symbol].dropna()
        series = series[series.index >= pd.Timestamp(start)]
        growth = amount * series / series.iloc[0]
        total = growth.iloc[-1] / amount - 1
        cagr = c.annualized_return(series)
        m = st.columns(3)
        m[0].metric("Value today", c.fmt_usd(growth.iloc[-1]), c.fmt_pct(total, signed=True))
        m[1].metric("Total return", c.fmt_pct(total, signed=True))
        m[2].metric("Per year (CAGR)", c.fmt_pct(cagr, signed=True))
        st.line_chart(growth.rename(f"Value of your {symbol} position ($)"), height=260)
        st.caption(
            "CAGR (compound annual growth rate) is the constant yearly return that would "
            "turn your starting amount into the final amount over the same period."
        )

    c.section("Compounding: why -10% then +10% isn't zero")
    st.markdown(
        """
Returns **multiply**, they don't add. Lose 10% on \\$100 and you have \\$90; gain 10%
on \\$90 and you have \\$99, not \\$100. Losses hurt more than equal-sized gains help.
"""
    )
    with c.try_it("The recovery problem"):
        loss = st.slider("Your portfolio falls by…", 5, 90, 50, 5, format="%d%%")
        needed = 1 / (1 - loss / 100) - 1
        st.metric("Gain needed just to get back to where you started", f"+{needed:.0%}")
        curve = pd.DataFrame({"loss": np.arange(5, 91, 5)})
        curve["gain_needed"] = 1 / (1 - curve["loss"] / 100) - 1
        chart = (
            alt.Chart(curve)
            .mark_line(point=True)
            .encode(
                x=alt.X("loss:Q", title="Loss (%)"),
                y=alt.Y("gain_needed:Q", title="Gain needed to recover", axis=alt.Axis(format="%")),
            )
        )
        marker = (
            alt.Chart(pd.DataFrame({"loss": [loss], "gain_needed": [needed]}))
            .mark_point(size=200, color="red", filled=True)
            .encode(x="loss:Q", y="gain_needed:Q")
        )
        st.altair_chart(chart + marker, width="stretch")
        st.markdown(
            f"After a **{loss}%** loss you need **+{needed:.0%}** to break even. This is "
            "why professionals obsess over avoiding big losses."
        )

    c.section("Volatility: how bumpy is the ride?")
    st.markdown(
        r"""
Two investments can end at the same place with very different journeys. **Volatility**
measures how much returns jump around: it's the **standard deviation** of daily
returns (a measure of spread around the average).

Daily volatility is a small number, so we **annualize** it by multiplying by
$\sqrt{252}$ (there are about 252 trading days in a year, and spread grows with the
square root of time). A stock with 30% annual volatility typically moves about ±2% on
a normal day.
"""
    )
    with c.try_it("Compare the daily moves of two tickers"):
        col1, col2 = st.columns(2)
        a = col1.selectbox("First", list(prices.columns), index=0, key="ch2_vol_a")
        b = col2.selectbox(
            "Second",
            list(prices.columns),
            index=list(prices.columns).index("NVDA") if "NVDA" in prices.columns else 1,
            key="ch2_vol_b",
        )
        rets = c.daily_returns(prices[[a, b]]).dropna()
        long = rets.melt(var_name="ticker", value_name="return")
        hist = (
            alt.Chart(long)
            .mark_bar(opacity=0.55)
            .encode(
                x=alt.X(
                    "return:Q",
                    bin=alt.Bin(maxbins=80),
                    title="Daily return",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("count():Q", stack=None, title="Number of days"),
                color="ticker:N",
            )
            .properties(height=280)
        )
        st.altair_chart(hist, width="stretch")
        m = st.columns(2)
        for col, t in zip(m, [a, b]):
            r = rets[t]
            col.metric(
                f"{t} annual volatility",
                c.fmt_pct(c.annualized_vol(r)),
                help=f"Worst day {r.min():.1%}, best day {r.max():.1%}",
            )
        st.caption(
            "A wider histogram means bigger daily swings. Both are centered near zero: on "
            "a single day, almost anything can happen."
        )

    c.section("Drawdown: the pain you actually feel")
    st.markdown(
        """
Volatility treats up and down moves the same, but investors only lose sleep over the
downs. A **drawdown** is how far the value is below its previous peak. The **maximum
drawdown** is the worst such fall over the period, the number that tells you whether
you could have stomached holding on.
"""
    )
    with c.try_it("See the underwater chart"):
        symbol = st.selectbox("Ticker", list(prices.columns), key="ch2_dd")
        p = prices[symbol].dropna()
        dd = c.drawdown(p)
        worst = dd.idxmin()
        peak = p[:worst].idxmax()
        chart = (
            alt.Chart(dd.rename("drawdown").reset_index())
            .mark_area(color="#B42318", opacity=0.5)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("drawdown:Q", axis=alt.Axis(format="%"), title="Below previous peak"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("drawdown:Q", format=".1%")],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")
        st.markdown(
            f"**{symbol}**'s maximum drawdown was **{dd.min():.1%}**: from its peak on "
            f"{peak.date()} to the bottom on {worst.date()}."
        )

    c.quiz(
        "risk_1",
        "A portfolio drops 50%. What gain does it need to get back to its starting value?",
        ["+50%", "+75%", "+100%", "+150%"],
        answer=2,
        explanation="Half of the money is left, so it must double: +100%.",
    )
    c.quiz(
        "risk_2",
        "What does volatility measure?",
        [
            "Whether a stock will go up",
            "How much returns spread out around their average",
            "The number of shares traded",
            "The biggest loss ever recorded",
        ],
        answer=1,
        explanation="Volatility is the standard deviation of returns, i.e. how bumpy the "
        "ride is, regardless of direction.",
    )
    c.takeaways(
        [
            "Returns (percent changes) make different assets comparable.",
            "Returns compound: losses need bigger gains to recover.",
            "Volatility = typical size of moves; annualize daily volatility with √252.",
            "Maximum drawdown = worst peak-to-trough fall, the pain an investor feels.",
        ]
    )
    c.chapter_footer(SLUG, ["risk_1", "risk_2"])
