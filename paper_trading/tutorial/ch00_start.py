"""Chapter 0: what quant finance is and how this tutorial works."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from paper_trading.tutorial import common as c

SLUG = "start"


def market_snapshot() -> None:
    """Live cards with each ticker's last close, daily change and a 3-month sparkline."""
    prices = c.closes()
    as_of = prices.index[-1].date()
    st.markdown(f"#### :primary[:material/show_chart:] Market snapshot :gray[· as of {as_of}]")
    tickers = list(prices.columns)
    for row in range(0, len(tickers), 5):
        cols = st.columns(5)
        for col, t in zip(cols, tickers[row : row + 5]):
            s = prices[t].dropna()
            col.metric(
                t,
                f"${s.iloc[-1]:,.2f}",
                f"{s.iloc[-1] / s.iloc[-2] - 1:+.2%}",
                border=True,
                chart_data=s.iloc[-63:].round(2).tolist(),
                chart_type="area",
            )
    st.caption("Daily closes from the local data store. Run quant-fetch to update them.")


def render() -> None:
    c.chapter_header(SLUG)

    st.markdown(
        """
Welcome! This tutorial assumes you know **nothing** about trading, finance or machine
learning. By the end you'll understand how a professional-style quant research stack
works: where market data comes from, how trading strategies are tested, why most of
them fail, how portfolios are built, and how machine learning fits in. You'll also have
used every piece of this app yourself.

Everything here runs on **real market data** (daily prices from Yahoo Finance) and on
the same Python code as the rest of this project. Nothing is a canned animation: when
you move a slider, the numbers are recomputed.
"""
    )

    cards = [
        (
            ":material/school:",
            "Learn",
            "13 short chapters, from what a stock is to how ML "
            "backtests leak. No prior knowledge needed.",
            c.PAGES.get("markets"),
            "Begin chapter 1",
        ),
        (
            ":material/science:",
            "Experiment",
            "Every chapter has live experiments on real prices: "
            "move a slider and the numbers are recomputed.",
            c.PAGES.get("pitfalls"),
            "See the pitfalls demo",
        ),
        (
            ":material/monitoring:",
            "Explore",
            "The Lab puts the whole research stack in one "
            "dashboard, including a paper-trading account.",
            c.PAGES.get("lab"),
            "Open the Lab",
        ),
    ]
    cols = st.columns(3)
    for col, (icon, title, text, page, link) in zip(cols, cards):
        with col.container(border=True, height="stretch"):
            st.markdown(f"### :primary[{icon}] {title}")
            st.markdown(f":gray[{text}]")
            if page is not None:
                st.page_link(
                    page, label=link, icon=":material/arrow_forward:", icon_position="right"
                )

    if c.require_data():
        market_snapshot()

    c.section('What is "quant"?')
    st.markdown(
        """
**Quantitative finance** ("quant") means making investment decisions with **data,
statistics and code** instead of gut feeling.

| | Discretionary investor | Quant |
|---|---|---|
| Decides by | reading news, intuition, experience | rules written as code |
| Tests ideas by | "it worked for me last year" | simulating the rule on years of history |
| Scales to | a handful of stocks | thousands of stocks, every day |
| Biggest risk | emotions, inconsistency | fooling yourself with bad statistics |

A **quant developer** builds the machinery behind this: data pipelines, backtesting
engines, risk models, optimizers, trading systems. A **quant researcher** uses that
machinery to look for ideas that make money. In practice, this project lets you play
both roles.
"""
    )

    if c.require_data():
        with c.try_it("A first taste: what if you had invested $1,000 in each stock?"):
            prices = c.closes()
            start_year = st.select_slider(
                "Invest on the first trading day of",
                options=sorted({d.year for d in prices.index})[:-1],
                value=prices.index[0].year,
            )
            window = prices[prices.index.year >= start_year]
            final = 1_000 * window.iloc[-1] / window.iloc[0]
            df = final.rename("value").reset_index().rename(columns={"index": "symbol"})
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("value:Q", title=f"Value on {window.index[-1].date()} ($)"),
                    y=alt.Y("symbol:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.value >= 1000, alt.value("#15803D"), alt.value("#B42318")
                    ),
                    tooltip=["symbol", alt.Tooltip("value:Q", format="$,.0f")],
                )
                .properties(height=320)
            )
            rule = (
                alt.Chart(pd.DataFrame({"x": [1000]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
            )
            st.altair_chart(chart + rule, width="stretch")
            best, worst = final.idxmax(), final.idxmin()
            st.markdown(
                c.esc(
                    f"Starting in **{start_year}**, $1,000 in **{best}** would be worth "
                    f"**{c.fmt_usd(final[best])}**, while **{worst}** would be worth "
                    f"**{c.fmt_usd(final[worst])}**. Move the slider: the winners change "
                    "depending on when you start. Keep that in mind, it's one of the most "
                    "important lessons in this whole tutorial."
                )
            )

    c.section("How this site is organized")
    st.markdown(
        """
- **Learn** (left sidebar): this tutorial, 12 chapters that build on each other.
  Each one has explanations, interactive "Try it" experiments, a short quiz, and a
  "What to remember" box. Do them in order the first time.
- **Lab**: the full research dashboard. Once you've read a chapter, the matching Lab
  tab lets you play with the real tools without hand-holding.

Roadmap:
"""
    )
    roadmap = pd.DataFrame(
        [
            {
                "#": ch.number,
                "Chapter": ch.title,
                "You'll learn": ch.summary,
                "Minutes": ch.minutes,
            }
            for ch in c.CHAPTERS
        ]
    )
    st.dataframe(roadmap, hide_index=True, width="stretch")

    c.warning_box(
        "This is an educational project. Nothing here is investment advice, and past "
        "performance (everything you'll see) does not guarantee future results."
    )

    c.quiz(
        "start_1",
        "What best describes a quant approach to investing?",
        [
            "Buying the companies whose products you like",
            "Writing decision rules as code and testing them on historical data",
            "Following the advice of famous investors",
            "Trading as fast as possible",
        ],
        answer=1,
        explanation="Quants express ideas as precise rules and measure them with data. "
        "Speed (high-frequency trading) is just one niche of quant finance.",
    )
    c.takeaways(
        [
            "Quant = decisions made by rules, code and statistics rather than intuition.",
            "Results depend heavily on the time period you look at.",
            "The hard part isn't finding something that worked in the past; it's knowing "
            "whether it will keep working.",
        ]
    )
    c.chapter_footer(SLUG, ["start_1"])
