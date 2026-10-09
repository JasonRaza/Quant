"""Chapter 6: lookahead bias, overfitting and multiple testing."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from paper_trading.tutorial import common as c

SLUG = "pitfalls"
COST = 0.001  # 10 bps per unit of position change


def ma_strategy_returns(close: pd.Series, fast: int, slow: int, lag: int = 1) -> pd.Series:
    """Vectorized dual-MA daily returns net of costs (same logic as the engine, simplified)."""
    position = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
    position = position.shift(lag).fillna(0.0)
    daily = close.pct_change().fillna(0.0)
    return position * daily - COST * position.diff().abs().fillna(0.0)


def growth(rets: pd.Series) -> pd.Series:
    return (1 + rets).cumprod()


@st.cache_data(show_spinner=False)
def grid_search(symbol: str, split: str):
    close = c.ohlcv(symbol)["close"]
    rows = []
    for fast in range(5, 65, 5):
        for slow in range(30, 260, 20):
            if fast >= slow:
                continue
            r = ma_strategy_returns(close, fast, slow)
            rows.append(
                {
                    "fast": fast,
                    "slow": slow,
                    "in_sample": c.sharpe(r[:split]),
                    "out_of_sample": c.sharpe(r[split:]),
                }
            )
    return pd.DataFrame(rows)


def render() -> None:
    c.chapter_header(SLUG)
    st.markdown(
        """
Building a backtest is easy. Building one that **doesn't lie to you** is the real job.
This chapter covers the three most common ways backtests fool people, with live demos
of each.
"""
    )
    if not c.require_data():
        return
    bench = c.settings().benchmark
    close = c.ohlcv(bench)["close"]

    c.section("1. Lookahead bias: using tomorrow's newspaper")
    st.markdown(
        """
**Lookahead bias** happens when a backtest uses information that wasn't available at
the time of the decision. It's usually an innocent bug, like computing a signal from
today's close and pretending you traded at that same close, or worse, using future
data directly.
"""
    )
    with c.try_it("Three versions of the same idea"):
        honest = ma_strategy_returns(close, 20, 50, lag=1)
        same_bar = ma_strategy_returns(close, 20, 50, lag=0)
        daily = close.pct_change().fillna(0.0)
        oracle = (daily > 0).astype(float) * daily  # knows tomorrow's direction
        curves = pd.DataFrame(
            {
                "Honest (decide yesterday, trade today)": growth(honest),
                "Subtle bug (decide and trade on the same close)": growth(same_bar),
                "Blatant bug (knows if today will be up)": growth(oracle),
            }
        )
        log_scale = st.toggle("Log scale (needed to see all three)", value=True)
        long = curves.reset_index().melt("date", var_name="version", value_name="growth")
        chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(
                    "growth:Q",
                    scale=alt.Scale(type="log" if log_scale else "linear"),
                    title="Growth of $1",
                ),
                color=alt.Color("version:N", title=None, legend=alt.Legend(orient="bottom")),
            )
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")
        finals = curves.iloc[-1]
        subtle = "better" if finals.iloc[1] > finals.iloc[0] else "worse"
        st.markdown(
            c.esc(
                f"On {bench}, $1 grows to **${finals.iloc[0]:,.2f}** honestly, "
                f"**${finals.iloc[1]:,.2f}** with the subtle bug, and an absurd "
                f"**${finals.iloc[2]:,.0f}** with perfect foresight. Here the subtle bug "
                f"happens to look {subtle}; either way the number is wrong, and you can't "
                "tell by how plausible it looks. Real quants are suspicious of any result that "
                "looks too good, and the first thing they check is timing."
            )
        )

    c.section("2. Overfitting: tuning until it fits the past")
    st.markdown(
        """
Our MA strategy has two knobs: the fast and slow windows. If you try many combinations
and keep the best one, you'll find settings that fit the **past** beautifully, partly
by luck. The honest test is **out-of-sample**: choose the settings on one period, then
judge them on a later period you never looked at.
"""
    )
    with c.try_it("Optimize on the past, test on the future"):
        years = sorted({d.year for d in close.index})
        split_year = st.select_slider(
            "Optimize up to the end of", options=years[1:-1], value=years[len(years) // 2]
        )
        split = f"{split_year}-12-31"
        grid = grid_search(bench, split)
        best = grid.loc[grid["in_sample"].idxmax()]
        heat = (
            alt.Chart(grid)
            .mark_rect()
            .encode(
                x=alt.X("fast:O", title="Fast MA"),
                y=alt.Y("slow:O", title="Slow MA", sort="descending"),
                color=alt.Color(
                    "in_sample:Q", scale=alt.Scale(scheme="redyellowgreen"), title="Sharpe"
                ),
                tooltip=[
                    "fast",
                    "slow",
                    alt.Tooltip("in_sample:Q", format=".2f"),
                    alt.Tooltip("out_of_sample:Q", format=".2f"),
                ],
            )
            .properties(height=300, title=f"In-sample Sharpe (until {split_year})")
        )
        st.altair_chart(heat, width="stretch")
        rank = grid["out_of_sample"].rank(ascending=False)[best.name]
        m = st.columns(3)
        m[0].metric(
            "Best settings in-sample",
            f"{int(best.fast)} / {int(best.slow)}",
            f"Sharpe {best.in_sample:.2f}",
        )
        m[1].metric(
            "Same settings, out-of-sample",
            f"Sharpe {best.out_of_sample:.2f}",
            f"{best.out_of_sample - best.in_sample:+.2f}",
        )
        m[2].metric(
            "Out-of-sample rank",
            f"{int(rank)} of {len(grid)}",
            help="Where the in-sample winner ranks among all settings on the later period.",
        )
        corr = grid["in_sample"].corr(grid["out_of_sample"], method="spearman")
        if corr <= -0.3:
            verdict = "settings that did best before tended to do *worse* afterwards"
        elif corr < 0.3:
            verdict = "past performance tells you very little about future performance"
        else:
            verdict = "past rankings carried some information about future rankings"
        st.markdown(
            f"The in-sample champion ranks **{int(rank)} of {len(grid)}** afterwards. Across "
            f"all settings, the rank correlation between past and future performance is "
            f"**{corr:.2f}**: {verdict}. Try different split years."
        )

    c.section("3. Multiple testing: try enough things and something works")
    st.markdown(
        """
If you test 100 strategies that are pure noise, the best of them will still look
impressive, just by chance. This is why a single good-looking backtest proves very
little unless you know **how many things were tried** to find it.
"""
    )
    with c.try_it("Generate random strategies"):
        col1, col2 = st.columns(2)
        n = col1.slider("How many random strategies?", 10, 500, 100, 10)
        seed = col2.number_input("Random seed", 0, 9999, 7)
        rng = np.random.default_rng(int(seed))
        daily = close.pct_change().fillna(0.0).to_numpy()
        signals = rng.integers(0, 2, size=(n, len(daily)))  # coin flip: in or out each day
        rets = signals * daily
        sharpes = np.sqrt(252) * rets.mean(axis=1) / rets.std(axis=1)
        hist = (
            alt.Chart(pd.DataFrame({"sharpe": sharpes}))
            .mark_bar()
            .encode(
                x=alt.X("sharpe:Q", bin=alt.Bin(maxbins=30), title="Sharpe ratio"),
                y=alt.Y("count():Q", title="Strategies"),
            )
            .properties(height=240)
        )
        st.altair_chart(hist, width="stretch")
        st.markdown(
            f"Each strategy flips a coin every day to decide whether to hold {bench} (costs "
            f"ignored). The **best** of {n} has a Sharpe of **{sharpes.max():.2f}**, "
            f"compared with {c.sharpe(close.pct_change().dropna()):.2f} for simply holding. "
            "It has zero skill; it just got lucky. Increase the count and watch the best "
            "result climb."
        )

    c.quiz(
        "pit_1",
        "Your backtest shows a Sharpe of 5 with almost no losing months. What's your first "
        "reaction?",
        [
            "Invest everything immediately",
            "Look for lookahead bias or other bugs before believing it",
            "Increase the leverage",
            "Publish it",
        ],
        answer=1,
        explanation="Results that are too good are usually bugs. Check timing, data and "
        "costs first.",
    )
    c.quiz(
        "pit_2",
        "Why is out-of-sample testing important?",
        [
            "It makes backtests faster",
            "It checks whether settings chosen on past data still work on data they never saw",
            "It removes transaction costs",
            "It's required by law",
        ],
        answer=1,
        explanation="Settings tuned on the past always look good on the past. Only unseen "
        "data tells you if there's a real effect.",
    )
    c.takeaways(
        [
            "Lookahead bias: never use information you wouldn't have had at decision time.",
            "Overfitting: the best settings on past data often disappoint afterwards.",
            "Multiple testing: with enough tries, pure luck looks like skill.",
            "Defenses: strict timing, out-of-sample tests, and counting how much you tried.",
        ]
    )
    c.chapter_footer(SLUG, ["pit_1", "pit_2"])
