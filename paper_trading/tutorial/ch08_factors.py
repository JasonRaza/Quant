"""Chapter 8: cross-sectional factors and the information coefficient."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_pipeline.storage import ParquetStore
from factors.analysis import (
    forward_returns,
    ic_summary,
    information_coefficient,
    quantile_returns,
    rebalance_dates,
)
from factors.library import compute_factors
from factors.panel import build_panel
from paper_trading.tutorial import common as c

SLUG = "factors"

FACTOR_INFO = {
    "momentum_12_1": (
        "Momentum (12-1)",
        "Return over the past 12 months, skipping the most recent month. Bet: recent winners "
        "keep winning for a while.",
    ),
    "reversal_1m": (
        "Short-term reversal",
        "Minus the last month's return. Bet: last month's biggest losers bounce back.",
    ),
    "low_vol": (
        "Low volatility",
        "Minus the last 3 months' volatility. Bet: calm stocks deliver better risk-adjusted "
        "returns than their reputation suggests.",
    ),
    "liquidity": (
        "Liquidity",
        "Log of average daily dollar volume. Measures how heavily traded a stock is.",
    ),
}


@st.cache_data(show_spinner=False)
def factor_data(horizon: int):
    cfg = c.settings()
    panel = build_panel(ParquetStore(cfg.data_dir), cfg.universe)
    factors = compute_factors(panel)
    fwd = forward_returns(panel.close, horizon)
    return panel.close, factors, fwd


@st.cache_data(show_spinner=False)
def random_factor_ics(n: int) -> list[float]:
    close, _, fwd = factor_data(21)
    rng = np.random.default_rng(0)
    dates = rebalance_dates(close.index, 21)[12:]
    out = []
    for _ in range(n):
        noise = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
        out.append(float(information_coefficient(noise, fwd, dates).mean()))
    return out


def render() -> None:
    c.chapter_header(SLUG)

    c.section("Time series vs cross-section")
    st.markdown(
        """
So far each strategy looked at **one asset over time** ("should I hold SPY today?").
That's **time-series** investing.

**Cross-sectional** investing asks a different question: *on a given day, which stocks
will do better than the others?* You rank all stocks by some characteristic, buy the
top ones and avoid (or short) the bottom ones. Even if the whole market falls, you can
win if your top picks fall less than your bottom picks.

The characteristic you rank by is called a **factor**. Large investment firms run
portfolios built on factors like these:
"""
    )
    for key, (name, desc) in FACTOR_INFO.items():
        c.term(name, desc)

    if not c.require_data():
        return
    close, factors, fwd = factor_data(21)
    names = list(FACTOR_INFO)

    with c.try_it("Today's factor rankings"):
        factor = st.selectbox(
            "Factor", names, format_func=lambda k: FACTOR_INFO[k][0], key="ch8_today"
        )
        today = factors[factor].dropna(how="all").iloc[-1].dropna().sort_values()
        df = today.rename("score").reset_index().rename(columns={"index": "ticker"})
        df.columns = ["ticker", "score"]
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("score:Q", title=FACTOR_INFO[factor][0]),
                y=alt.Y("ticker:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.score >= df["score"].median(),
                    alt.value("#15803D"),
                    alt.value("#9CA3AF"),
                ),
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"As of {factors[factor].dropna(how='all').index[-1].date()}. A factor strategy "
            "would favor the green (top half) names."
        )

    c.section("Does the factor predict anything? The information coefficient")
    st.markdown(
        """
To test a factor, we check on many past dates whether **high scores were followed by
high returns**. The standard measure is the **information coefficient (IC)**: the
correlation between the factor's ranking and the next month's return ranking, computed
date by date.

- IC = **+1**: the ranking predicted the order of next month's returns perfectly.
- IC = **0**: no relationship.
- IC = **-1**: perfectly backwards.

In real life, an average IC of **0.03 to 0.05** sustained over many years and many
stocks is already valuable. That shows how little predictability there is.
"""
    )
    with c.try_it("Look at one date"):
        factor = st.selectbox(
            "Factor", names, format_func=lambda k: FACTOR_INFO[k][0], key="ch8_one"
        )
        valid = factors[factor].dropna(how="all").index.intersection(fwd.dropna(how="all").index)
        dt = st.select_slider(
            "Date",
            options=list(valid[::21]),
            value=valid[::21][len(valid[::21]) // 2],
            format_func=lambda d: d.strftime("%Y-%m-%d"),
        )
        pair = pd.DataFrame({"factor": factors[factor].loc[dt], "next_month": fwd.loc[dt]}).dropna()
        pair["factor_rank"] = pair["factor"].rank()
        pair["return_rank"] = pair["next_month"].rank()
        ic = pair["factor_rank"].corr(pair["return_rank"])
        data = pair.reset_index().rename(columns={"index": "ticker"})
        data.columns = ["ticker"] + list(data.columns[1:])
        pts = (
            alt.Chart(data)
            .mark_circle(size=160)
            .encode(
                x=alt.X("factor_rank:Q", title="Factor rank (1 = lowest)"),
                y=alt.Y("return_rank:Q", title="Next-month return rank"),
                tooltip=["ticker", alt.Tooltip("next_month:Q", format=".1%")],
            )
        )
        labels = pts.mark_text(dx=10, align="left").encode(text="ticker:N")
        st.altair_chart((pts + labels).properties(height=320), width="stretch")
        st.markdown(
            f"On **{dt.date()}**, the IC was **{ic:+.2f}**. If points line up from bottom-left "
            "to top-right, the factor ranked stocks correctly. Move the date: the IC jumps "
            "around a lot from month to month."
        )

    with c.try_it("Average over all dates"):
        horizon = st.select_slider("Forward horizon (days)", [5, 21, 63], value=21)
        close_h, factors_h, fwd_h = factor_data(horizon)
        dates = rebalance_dates(close_h.index, horizon)
        rows, cum = [], {}
        for key in names:
            ic_series = information_coefficient(factors_h[key], fwd_h, dates)
            s = ic_summary(ic_series)
            rows.append(
                {
                    "factor": FACTOR_INFO[key][0],
                    "mean IC": s["ic_mean"],
                    "t-stat": s["t_stat"],
                    "hit rate": s["hit_rate"],
                    "dates": s["n"],
                }
            )
            cum[FACTOR_INFO[key][0]] = ic_series.cumsum()
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"mean IC": "{:+.3f}", "t-stat": "{:.2f}", "hit rate": "{:.0%}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.line_chart(pd.DataFrame(cum), height=260)
        st.markdown(
            "The **t-stat** asks whether the average IC is distinguishable from zero given how "
            "much it bounces around; you'd want it above about 2. A steadily rising cumulative "
            "line means the factor works consistently; a random walk means it doesn't."
        )

    c.section("Is that just noise?")
    st.markdown(
        """
With only nine stocks, each month's IC is extremely noisy. A good sanity check is to
compare real factors with **random** ones: shuffle scores at random and see what ICs
pure chance produces.
"""
    )
    with c.try_it("Random factors vs real ones"):
        n = st.slider("Number of random factors", 20, 300, 100, 20)
        with st.spinner("Scoring random factors..."):
            rand_ics = random_factor_ics(n)
        dates = rebalance_dates(close.index, 21)[12:]
        real = {
            FACTOR_INFO[k][0]: information_coefficient(factors[k], fwd, dates).mean() for k in names
        }
        hist = (
            alt.Chart(pd.DataFrame({"mean IC": rand_ics}))
            .mark_bar(color="#CBD5E1")
            .encode(
                x=alt.X("mean IC:Q", bin=alt.Bin(maxbins=30)),
                y=alt.Y("count():Q", title="Random factors"),
            )
        )
        rules = (
            alt.Chart(pd.DataFrame({"factor": list(real), "mean IC": list(real.values())}))
            .mark_rule(strokeWidth=2)
            .encode(x="mean IC:Q", color=alt.Color("factor:N", title="Real factors"))
        )
        st.altair_chart((hist + rules).properties(height=260), width="stretch")
        inside = sum(min(rand_ics) <= v <= max(rand_ics) for v in real.values())
        st.markdown(
            f"**{inside} of {len(real)}** real factors fall inside the range produced by pure "
            "noise. On this tiny universe, we can't tell them apart from luck. Real factor "
            "research uses thousands of stocks to get enough statistical power."
        )

    c.section("From ranking to portfolio: quantiles")
    st.markdown(
        """
Another way to judge a factor: each month, split stocks into groups (**quantiles**) by
score, and look at the average next-month return of each group. A good factor shows a
**staircase**: the top group beats the middle, which beats the bottom. The difference
between top and bottom is the **long-short spread**.
"""
    )
    with c.try_it("Quantile returns"):
        factor = st.selectbox("Factor", names, format_func=lambda k: FACTOR_INFO[k][0], key="ch8_q")
        q = quantile_returns(factors[factor], fwd, rebalance_dates(close.index, 21), 3)
        means = q.mean().rename("avg next-month return").reset_index()
        means.columns = ["group", "avg next-month return"]
        means["group"] = means["group"].map(
            {"q1": "Bottom third", "q2": "Middle", "q3": "Top third"}
        )
        bar = (
            alt.Chart(means)
            .mark_bar()
            .encode(
                x=alt.X("group:N", sort=None, title=None),
                y=alt.Y("avg next-month return:Q", axis=alt.Axis(format="%")),
            )
            .properties(height=240)
        )
        st.altair_chart(bar, width="stretch")

    c.quiz(
        "fac_1",
        "What is a cross-sectional strategy?",
        [
            "Holding one stock and timing when to buy it",
            "Ranking many stocks on the same date and favoring the top-ranked ones",
            "Only trading at the end of the day",
            "Buying stocks from different countries",
        ],
        answer=1,
        explanation="It compares stocks against each other at each point in time.",
    )
    c.quiz(
        "fac_2",
        "A factor has an average IC of +0.04. What does that mean?",
        [
            "It's useless",
            "Its rankings are slightly, positively related to future returns; small but "
            "potentially valuable if consistent",
            "It predicts returns 4% of the time",
            "It returns 4% per year",
        ],
        answer=1,
        explanation="ICs are small in practice; consistency and breadth (many stocks, many "
        "dates) make them useful.",
    )
    c.takeaways(
        [
            "Factors rank stocks against each other: momentum, reversal, low volatility…",
            "The IC measures how well a ranking predicts the next period's returns.",
            "Real ICs are small; you need many stocks and dates to trust them.",
            "Compare against random factors to check you're not just seeing noise.",
        ]
    )
    c.chapter_footer(SLUG, ["fac_1", "fac_2"])
