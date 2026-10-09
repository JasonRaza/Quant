"""Chapter 9: diversification, optimization and walk-forward portfolio tests."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_pipeline.storage import ParquetStore
from factors.panel import build_panel
from paper_trading.tutorial import common as c
from portfolio.estimators import daily_returns, ledoit_wolf_covariance, sample_covariance
from portfolio.optimizer import (
    Constraints,
    equal_weight,
    max_sharpe,
    min_variance,
    risk_contributions,
    risk_parity,
)
from portfolio.walkforward import WalkForwardConfig, default_allocators, run_walkforward

SLUG = "portfolio"


@st.cache_data(show_spinner=False)
def universe_returns() -> pd.DataFrame:
    cfg = c.settings()
    panel = build_panel(ParquetStore(cfg.data_dir), cfg.universe)
    return daily_returns(panel.close)


@st.cache_data(show_spinner=False)
def walkforward_curves(max_weight: float) -> pd.DataFrame:
    rets = universe_returns()
    cons = Constraints(max_weight=max_weight)
    allocs = default_allocators(cons)
    labels = {
        "equal_weight": "Equal weight",
        "min_variance": "Min variance",
        "risk_parity": "Risk parity",
    }
    return pd.DataFrame(
        {labels[k]: run_walkforward(rets, a, WalkForwardConfig()).equity for k, a in allocs.items()}
    )


def trailing_max_sharpe(window: pd.DataFrame, cons: Constraints) -> pd.Series:
    """Max-Sharpe weights from one window's sample mean and covariance."""
    mu, cov = window.mean() * 252, sample_covariance(window)
    return max_sharpe(mu, cov, constraints=cons) if (mu > 0).any() else min_variance(cov, cons)


def render() -> None:
    c.chapter_header(SLUG)

    c.section("Why hold more than one stock?")
    st.markdown(
        """
Chapter 3 showed that stocks don't move perfectly together. Combine two of them and
the portfolio's volatility is **lower than the average of their volatilities**, often
much lower. The less correlated they are, the bigger the benefit.
"""
    )
    if not c.require_data():
        return
    rets = universe_returns().dropna()
    tickers = list(rets.columns)

    with c.try_it("Mix two stocks"):
        col1, col2 = st.columns(2)
        a = col1.selectbox(
            "Stock A", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0
        )
        b = col2.selectbox("Stock B", tickers, index=tickers.index("KO") if "KO" in tickers else 1)
        if a == b:
            st.error("Pick two different stocks.")
        else:
            w = st.slider(f"Weight in {a}", 0, 100, 50, 5, format="%d%%") / 100
            cov = rets[[a, b]].cov().to_numpy() * 252
            grid = np.linspace(0, 1, 51)
            vols = [np.sqrt(np.array([x, 1 - x]) @ cov @ np.array([x, 1 - x])) for x in grid]
            naive = [x * np.sqrt(cov[0, 0]) + (1 - x) * np.sqrt(cov[1, 1]) for x in grid]
            df = pd.DataFrame(
                {"weight": grid, "Actual portfolio": vols, "If perfectly correlated": naive}
            )
            long = df.melt("weight", var_name="series", value_name="volatility")
            lines = (
                alt.Chart(long)
                .mark_line()
                .encode(
                    x=alt.X("weight:Q", axis=alt.Axis(format="%"), title=f"Weight in {a}"),
                    y=alt.Y("volatility:Q", axis=alt.Axis(format="%"), title="Annual volatility"),
                    color=alt.Color("series:N", title=None),
                    strokeDash=alt.StrokeDash("series:N", legend=None),
                )
            )
            cur = np.sqrt(np.array([w, 1 - w]) @ cov @ np.array([w, 1 - w]))
            dot = (
                alt.Chart(pd.DataFrame({"weight": [w], "volatility": [cur]}))
                .mark_point(size=200, filled=True, color="red")
                .encode(x="weight:Q", y="volatility:Q")
            )
            st.altair_chart((lines + dot).properties(height=300), width="stretch")
            corr = rets[a].corr(rets[b])
            st.markdown(
                f"Correlation {a}/{b}: **{corr:.2f}**. With {w:.0%} in {a}, volatility is "
                f"**{cur:.1%}**. The gap between the two curves is the free lunch of "
                "diversification."
            )

    c.section("Optimization: the efficient frontier")
    st.markdown(
        """
With nine stocks there are infinitely many ways to split your money. **Mean-variance
optimization** (Harry Markowitz, Nobel Prize 1990) searches for the best trade-off
between expected return and risk. Each dot below is one random portfolio; the optimizers
find specific points on the cloud:

- **Min variance**: the least risky combination.
- **Max Sharpe**: the most return per unit of risk (needs return forecasts).
- **Risk parity**: each stock contributes the same amount of risk.
- **Equal weight**: 1/9 each; the simple baseline.

All are **long-only** (no short selling) and capped at a maximum weight per stock.
"""
    )
    with c.try_it("Explore the risk/return cloud (last 12 months of data)"):
        window = rets.iloc[-252:]
        cov = ledoit_wolf_covariance(window)
        mu = window.mean() * 252
        cap = st.slider("Max weight per stock", 0.12, 1.0, 0.35, 0.01, key="ch9_cap")
        cons = Constraints(max_weight=cap)
        rng = np.random.default_rng(1)
        w_rand = rng.dirichlet(np.ones(len(tickers)), 2000)
        port_ret = w_rand @ mu.to_numpy()
        port_vol = np.sqrt(np.einsum("ij,jk,ik->i", w_rand, cov.to_numpy(), w_rand))
        cloud = pd.DataFrame({"volatility": port_vol, "return": port_ret})
        special = {
            "Equal weight": equal_weight(cov),
            "Min variance": min_variance(cov, cons),
            "Risk parity": risk_parity(cov),
        }
        if (mu > 0).any():
            special["Max Sharpe"] = max_sharpe(mu, cov, constraints=cons)
        marks = pd.DataFrame(
            [
                {"portfolio": k, "volatility": float(np.sqrt(w @ cov @ w)), "return": float(w @ mu)}
                for k, w in special.items()
            ]
        )
        base = (
            alt.Chart(cloud)
            .mark_circle(size=10, opacity=0.3, color="#90a4ae")
            .encode(
                x=alt.X("volatility:Q", axis=alt.Axis(format="%"), title="Risk (volatility)"),
                y=alt.Y(
                    "return:Q",
                    axis=alt.Axis(format="%"),
                    title="Return (last 12 months, annualized)",
                ),
            )
        )
        pts = (
            alt.Chart(marks)
            .mark_point(size=220, filled=True)
            .encode(
                x="volatility:Q",
                y="return:Q",
                color=alt.Color("portfolio:N", title=None),
                tooltip=[
                    "portfolio",
                    alt.Tooltip("return:Q", format=".1%"),
                    alt.Tooltip("volatility:Q", format=".1%"),
                ],
            )
        )
        st.altair_chart((base + pts).properties(height=340), width="stretch")
        pick = st.radio("Show weights of", list(special), horizontal=True)
        st.bar_chart(special[pick].rename("weight"), height=220)
        rc = risk_contributions(special[pick], cov)
        st.caption(f"Largest risk contribution: {rc.idxmax()} ({rc.max():.0%} of portfolio risk).")

    c.section("The catch: estimates are noisy")
    st.markdown(
        """
Optimizers treat their inputs as truth, but expected returns and correlations are
**estimated from the past** and are very noisy. The optimizer then piles into whatever
happened to look best, a problem nicknamed **error maximization**. Two defenses used
in this project:

- **Shrinkage** (Ledoit-Wolf): pull the noisy covariance estimate toward a simple,
  stable target. Less noise in, less nonsense out.
- **Don't trust return forecasts**: min variance and risk parity don't use them at all.
"""
    )
    with c.try_it("Same method, two consecutive years"):
        years = sorted({d.year for d in rets.index})
        if len(years) >= 3:
            y1 = st.select_slider("First year", options=years[:-2], value=years[-3])
            r1, r2 = rets.loc[str(y1)], rets.loc[str(y1 + 1)]
            cons = Constraints(max_weight=0.5)
            w1, w2 = trailing_max_sharpe(r1, cons), trailing_max_sharpe(r2, cons)
            df = pd.DataFrame({str(y1): w1, str(y1 + 1): w2})
            long = df.rename_axis("ticker").reset_index().melt(
                "ticker", var_name="year", value_name="weight"
            )
            chart = (
                alt.Chart(long)
                .mark_bar()
                .encode(
                    x=alt.X("ticker:N", title=None),
                    xOffset="year:N",
                    y=alt.Y("weight:Q", axis=alt.Axis(format="%")),
                    color="year:N",
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")
            turnover = float((w1 - w2).abs().sum() / 2)
            st.markdown(
                f"Max-Sharpe weights estimated on {y1} vs {y1 + 1}: **{turnover:.0%}** of the "
                "portfolio would change hands. Same method, one more year of data, a completely "
                "different portfolio."
            )

    c.section("Testing allocators honestly: walk-forward")
    st.markdown(
        """
To compare allocators fairly we use a **walk-forward** test: every month, estimate on
the past year only, set the weights, then let them ride until next month while paying
trading costs. No allocator ever sees the future.
"""
    )
    with c.try_it("Walk-forward comparison"):
        cap = st.slider("Max weight per stock", 0.12, 1.0, 0.25, 0.01, key="ch9_wf_cap")
        with st.spinner("Running monthly rebalances..."):
            curves = walkforward_curves(cap)
        st.line_chart(curves, height=300)
        final = curves.iloc[-1] / curves.iloc[0] - 1
        dd = {k: float(c.drawdown(v).min()) for k, v in curves.items()}
        st.dataframe(
            pd.DataFrame({"total return": final, "max drawdown": pd.Series(dd)}).style.format(
                "{:.1%}"
            ),
            width="stretch",
        )
        st.caption(
            "More allocators (including factor tilts and max Sharpe) are in the Lab's "
            "Portfolio tab."
        )

    c.quiz(
        "port_1",
        "Two stocks each have 30% volatility and a correlation of 0.3. A 50/50 mix has volatility…",
        ["Exactly 30%", "Below 30%", "Above 30%", "Exactly 15%"],
        answer=1,
        explanation="With correlation below 1, part of the swings cancel out, so the mix is "
        "less volatile than either stock (about 24% here).",
    )
    c.quiz(
        "port_2",
        "Why do min variance and risk parity avoid using expected returns?",
        [
            "Returns don't matter",
            "Expected returns are the noisiest input, and optimizers amplify their errors",
            "It's faster to compute",
            "Regulations forbid it",
        ],
        answer=1,
        explanation="Risk (covariance) is much easier to estimate than future returns.",
    )
    c.takeaways(
        [
            "Diversification lowers risk whenever correlations are below 1.",
            "Optimizers find the best trade-off, but only relative to their (noisy) inputs.",
            "Shrinkage and return-free methods (min variance, risk parity) are more robust.",
            "Compare allocators with walk-forward tests that include costs.",
        ]
    )
    c.chapter_footer(SLUG, ["port_1", "port_2"])
