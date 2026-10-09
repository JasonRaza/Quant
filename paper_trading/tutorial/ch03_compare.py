"""Chapter 3: comparing investments with risk-adjusted metrics and correlation."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from paper_trading.tutorial import common as c

SLUG = "compare"


def render() -> None:
    c.chapter_header(SLUG)

    c.section("Return alone is misleading")
    st.markdown(
        r"""
Imagine two investments that both earned 10% a year. One barely moved; the other
swung wildly and once lost half its value. They're not equally good: the second one
took far more **risk** for the same reward.

The most common way to combine reward and risk into one number is the **Sharpe
ratio**:

$$\text{Sharpe} = \frac{\text{average annual return}}{\text{annual volatility}}$$

(Strictly, the numerator is the return *above* a risk-free rate such as a savings
account; we skip that here to keep things simple.) Roughly: below 0.5 is mediocre,
around 1 is good, above 2 over many years is exceptional and should make you
suspicious.
"""
    )
    if not c.require_data():
        return

    prices = c.closes()
    with c.try_it("Risk vs return map"):
        years = sorted({d.year for d in prices.index})
        start_year = st.select_slider(
            "Start of the period", options=years[:-1], value=years[0], key="ch3_start"
        )
        window = prices[prices.index.year >= start_year]
        rets = c.daily_returns(window)
        stats = pd.DataFrame(
            {
                "annual_return": [c.annualized_return(window[s].dropna()) for s in window],
                "volatility": [c.annualized_vol(rets[s].dropna()) for s in window],
                "sharpe": [c.sharpe(rets[s].dropna()) for s in window],
            },
            index=window.columns,
        )
        stats.index.name = "ticker"
        data = stats.reset_index()
        bench = c.settings().benchmark
        points = (
            alt.Chart(data)
            .mark_circle(size=160)
            .encode(
                x=alt.X(
                    "volatility:Q", axis=alt.Axis(format="%"), title="Risk (annual volatility)"
                ),
                y=alt.Y("annual_return:Q", axis=alt.Axis(format="%"), title="Return per year"),
                color=alt.Color(
                    "sharpe:Q", scale=alt.Scale(scheme="redyellowgreen"), title="Sharpe"
                ),
                tooltip=[
                    "ticker",
                    alt.Tooltip("annual_return:Q", format=".1%"),
                    alt.Tooltip("volatility:Q", format=".1%"),
                    alt.Tooltip("sharpe:Q", format=".2f"),
                ],
            )
        )
        labels = points.mark_text(dx=12, align="left").encode(
            text="ticker:N", color=alt.value("gray")
        )
        st.altair_chart((points + labels).properties(height=360), width="stretch")
        best = stats["sharpe"].idxmax()
        st.markdown(
            f"Since **{start_year}**, the best risk-adjusted performer is **{best}** "
            f"(Sharpe {stats.loc[best, 'sharpe']:.2f}). {bench}, the whole market, has a "
            f"Sharpe of {stats.loc[bench, 'sharpe']:.2f}. Up and to the left is best: more "
            "return for less risk. Change the start year and watch the dots move."
        )
        st.dataframe(
            stats.sort_values("sharpe", ascending=False).style.format(
                {"annual_return": "{:.1%}", "volatility": "{:.1%}", "sharpe": "{:.2f}"}
            ),
            width="stretch",
        )

    c.section("Correlation: do things move together?")
    st.markdown(
        """
**Correlation** measures how much two assets move in the same direction, from **+1**
(always together) through **0** (unrelated) to **-1** (always opposite).

It matters because combining assets that don't move perfectly together **reduces
risk**: when one zigs, the other may zag. This is **diversification**, sometimes
called "the only free lunch in finance". Chapter 9 builds portfolios on this idea.
"""
    )
    with c.try_it("Correlation heatmap"):
        rets = c.daily_returns(prices).dropna()
        corr = rets.corr()
        long = corr.reset_index().melt(id_vars="index", var_name="b", value_name="corr")
        long = long.rename(columns={"index": "a"})
        heat = (
            alt.Chart(long)
            .mark_rect()
            .encode(
                x=alt.X("a:N", title=None),
                y=alt.Y("b:N", title=None),
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
                tooltip=["a", "b", alt.Tooltip("corr:Q", format=".2f")],
            )
        )
        text = heat.mark_text(fontSize=11).encode(
            text=alt.Text("corr:Q", format=".2f"), color=alt.value("black")
        )
        st.altair_chart((heat + text).properties(height=380), width="stretch")
        off = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        hi, lo = off.idxmax(), off.idxmin()
        st.markdown(
            f"The most similar pair is **{hi[0]}/{hi[1]}** ({off.max():.2f}); the least "
            f"similar is **{lo[0]}/{lo[1]}** ({off.min():.2f}). Notice that almost "
            "everything is positively correlated: large US stocks tend to rise and fall "
            "with the overall market."
        )

    c.section("Beta: how much a stock follows the market")
    st.markdown(
        """
**Beta** measures how strongly a stock reacts to market moves. A beta of 1.5 means
that when the market (SPY) moves 1%, the stock tends to move 1.5% in the same
direction. Defensive stocks (like beverage companies) often have betas below 1;
fast-growing tech stocks often above 1.
"""
    )
    with c.try_it("Estimate a beta"):
        bench = c.settings().benchmark
        others = [s for s in prices.columns if s != bench]
        symbol = st.selectbox("Ticker", others, key="ch3_beta")
        r = c.daily_returns(prices[[bench, symbol]]).dropna()
        beta = r.cov().loc[symbol, bench] / r[bench].var()
        sample = r.sample(min(len(r), 1200), random_state=0)
        scatter = (
            alt.Chart(sample)
            .mark_circle(opacity=0.35, size=18)
            .encode(
                x=alt.X(f"{bench}:Q", axis=alt.Axis(format="%"), title=f"{bench} daily return"),
                y=alt.Y(f"{symbol}:Q", axis=alt.Axis(format="%"), title=f"{symbol} daily return"),
            )
        )
        line = scatter.transform_regression(bench, symbol).mark_line(color="red")
        st.altair_chart((scatter + line).properties(height=320), width="stretch")
        st.metric(f"Beta of {symbol} vs {bench}", f"{beta:.2f}")
        st.caption("Each dot is one day. The red line's slope is the beta.")

    c.quiz(
        "compare_1",
        "Strategy A returned 12%/year with 24% volatility. Strategy B returned 8%/year "
        "with 8% volatility. Which has the better Sharpe ratio?",
        ["A (0.5)", "B (1.0)", "They're equal", "Can't tell without more data"],
        answer=1,
        explanation="12/24 = 0.5 and 8/8 = 1.0. B earns twice as much per unit of risk; "
        "you could even borrow to scale it up to A's risk level and earn more.",
    )
    c.quiz(
        "compare_2",
        "Why does diversification reduce risk?",
        [
            "Because more stocks means more return",
            "Because assets that don't move perfectly together partly cancel out each "
            "other's swings",
            "Because big companies never go down",
            "It doesn't; it only reduces fees",
        ],
        answer=1,
        explanation="With correlation below 1, some losses are offset by gains elsewhere, "
        "so the portfolio is less volatile than its average component.",
    )
    c.takeaways(
        [
            "Always compare return together with risk; the Sharpe ratio does exactly that.",
            "Rankings change a lot with the period you choose.",
            "Correlation below 1 is what makes diversification work.",
            "Beta measures sensitivity to the overall market.",
        ]
    )
    c.chapter_footer(SLUG, ["compare_1", "compare_2"])
