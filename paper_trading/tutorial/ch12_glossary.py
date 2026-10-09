"""Chapter 12: searchable glossary and final quiz."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from paper_trading.tutorial import common as c

SLUG = "glossary"

GLOSSARY = {
    "Adjusted price": "Historical price rescaled for splits and dividends so returns are correct.",
    "Alpha": "Return beyond what the market (or a benchmark) explains; the skill part.",
    "Annualize": "Convert a daily number to a yearly one (×252 for returns, ×√252 for volatility).",
    "Backtest": "Simulation of a strategy on historical data, day by day.",
    "Benchmark": "The default alternative a strategy must beat, here SPY.",
    "Beta": "Sensitivity of a stock to market moves.",
    "Bid / ask": "Best price buyers offer / best price sellers accept. The gap is the spread.",
    "Basis point (bp)": "0.01%. 10 bps = 0.1%.",
    "Correlation": "How much two series move together, from -1 to +1.",
    "Covariance": "Joint variability of two assets; the input optimizers use for risk.",
    "Cross-sectional": "Comparing many assets at the same date.",
    "Drawdown": "Fall from the previous peak. Maximum drawdown is the worst one.",
    "Embargo": "Extra gap left between training and test data to avoid leakage.",
    "ETF": "Exchange-traded fund; a basket of assets traded like a single stock.",
    "Factor": "A characteristic used to rank stocks (momentum, volatility…).",
    "Feature": "An input variable given to an ML model.",
    "Information coefficient (IC)": "Rank correlation between predictions and realized returns.",
    "Label": "The value an ML model is trained to predict.",
    "Leakage": "When training data contains information from the test period.",
    "Long / short": "Own an asset (profit if it rises) / borrow and sell it (profit if it falls).",
    "Lookahead bias": "Using information that wasn't available at decision time.",
    "Market neutral": "Long and short positions balanced so market moves roughly cancel.",
    "Mean reversion": "Tendency of unusual moves to come back toward an average.",
    "Momentum": "Tendency of recent winners to keep outperforming for a while.",
    "Moving average": "Average price over the last N days, updated daily.",
    "OHLCV": "Open, high, low, close, volume: one daily bar.",
    "Out-of-sample": "Data not used to build or tune the model or strategy.",
    "Overfitting": "Fitting noise in past data; looks great in-sample, fails afterwards.",
    "Paper trading": "Running a strategy live with simulated money.",
    "Purging": "Removing training samples whose label window overlaps the test period.",
    "Rebalance": "Trading back to target weights.",
    "Risk parity": "Allocation where each asset contributes equally to total risk.",
    "Sharpe ratio": "Annual return divided by annual volatility.",
    "Shrinkage": "Pulling a noisy estimate toward a stable target (e.g. Ledoit-Wolf covariance).",
    "Slippage": "Difference between the expected and the actual fill price.",
    "Survivorship bias": "Testing only on assets that survived to today.",
    "Ticker": "Short code identifying a security (AAPL, SPY…).",
    "Turnover": "How much of the portfolio is traded; drives costs.",
    "Volatility": "Standard deviation of returns; how bumpy the ride is.",
    "Walk-forward": "Repeatedly train on the past and test on the next period.",
}

FINAL_QUIZ = [
    ("A stock falls 20%. What gain gets it back to even?", ["+20%", "+25%", "+40%", "+80%"], 1),
    ("Which metric divides return by volatility?", ["Beta", "IC", "Sharpe ratio", "Drawdown"], 2),
    (
        "Using today's closing price to decide a trade executed at today's close is…",
        ["Fine", "Lookahead bias", "Survivorship bias", "Overfitting"],
        1,
    ),
    (
        "You tried 200 parameter combinations and kept the best. Its backtest Sharpe is…",
        ["Reliable", "Probably overstated", "Understated", "Unaffected"],
        1,
    ),
    (
        "A pairs trade is long KO and short PEP. The whole market drops 10%. Roughly what happens?",
        ["You lose 10%", "Both legs mostly offset", "You gain 10%", "You lose 20%"],
        1,
    ),
    (
        "An IC of 0 means the factor's ranking…",
        ["Is perfect", "Has no relationship with future returns", "Is inverted", "Is significant"],
        1,
    ),
    (
        "Which allocator needs return forecasts?",
        ["Min variance", "Risk parity", "Max Sharpe", "Equal weight"],
        2,
    ),
    (
        "Which validation is appropriate for predicting 21-day returns?",
        ["Shuffled K-fold", "Purged walk-forward", "Training on everything", "Leave-one-stock-out"],
        1,
    ),
]


def render() -> None:
    c.chapter_header(SLUG)

    c.section("Glossary")
    query = st.text_input("Search a term", placeholder="e.g. drawdown, leakage, beta")
    terms = pd.DataFrame([{"Term": k, "Meaning": v} for k, v in sorted(GLOSSARY.items())])
    if query:
        mask = terms["Term"].str.contains(query, case=False) | terms["Meaning"].str.contains(
            query, case=False
        )
        terms = terms[mask]
    st.dataframe(terms, hide_index=True, width="stretch")

    c.section("Final quiz")
    st.markdown("Eight questions covering the whole tutorial. Answer them all to see your score.")
    answers = []
    for i, (question, options, answer) in enumerate(FINAL_QUIZ, start=1):
        st.markdown(f"**{i}. {question}**")
        choice = st.radio(
            question, options, index=None, key=f"final_{i}", label_visibility="collapsed"
        )
        answers.append(None if choice is None else options.index(choice) == answer)

    if all(a is not None for a in answers):
        score = sum(answers)
        st.metric("Your score", f"{score} / {len(FINAL_QUIZ)}")
        wrong = [i for i, a in enumerate(answers, start=1) if not a]
        if not wrong:
            st.balloons()
            st.success("Perfect score. You're ready to explore the Lab on your own.")
        else:
            st.warning(
                "Review questions "
                + ", ".join(str(i) for i in wrong)
                + ". Correct answers: "
                + "; ".join(f"{i}: {FINAL_QUIZ[i - 1][1][FINAL_QUIZ[i - 1][2]]}" for i in wrong)
            )
        st.session_state.setdefault("quiz_results", {})["final"] = True

    c.section("Where to go next")
    st.markdown(
        """
- Open the **Lab → Research dashboard** and reproduce every experiment with your own
  settings.
- Read the code: each chapter maps to a module (`data_pipeline/`, `backtester/`,
  `factors/`, `portfolio/`, `ml_alpha/`, `paper_trading/`).
- Classic books: *Active Portfolio Management* (Grinold & Kahn), *Advances in Financial
  Machine Learning* (López de Prado), *Quantitative Trading* (Chan).
"""
    )
    c.chapter_footer(SLUG, ["final"])
