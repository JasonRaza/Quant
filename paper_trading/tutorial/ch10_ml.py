"""Chapter 10: machine learning for return prediction, overfitting and leakage."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

from data_pipeline.storage import ParquetStore
from factors.panel import build_panel
from ml_alpha.cli import evaluate_models
from ml_alpha.cv import PurgedWalkForward
from ml_alpha.features import build_dataset
from ml_alpha.models import cross_validate, daily_rank_ic
from paper_trading.tutorial import common as c

SLUG = "ml"

LABELS = {
    "momentum_baseline": "Momentum baseline",
    "ridge": "Ridge regression",
    "gradient_boosting": "Gradient boosting",
}


@st.cache_data(show_spinner=False)
def dataset():
    cfg = c.settings()
    panel = build_panel(ParquetStore(cfg.data_dir), cfg.universe)
    return build_dataset(panel, horizon=21, sample_every=5)


@st.cache_data(show_spinner=False)
def model_comparison():
    results, _, _ = evaluate_models(dataset())
    return pd.DataFrame(
        {
            LABELS[k]: {
                "Honest (purged walk-forward)": v["purged_walk_forward"]["ic_mean"],
                "Leaky (shuffled K-fold)": v["leaky_shuffled_kfold_ic_mean"],
            }
            for k, v in results.items()
        }
    ).T


@st.cache_data(show_spinner=False)
def depth_curve():
    ds = dataset()
    splits = list(PurgedWalkForward(5, 52, 5).split(ds.dates, ds.label_end))
    rows = []
    for depth in [1, 2, 3, 4, 6, 8, 12]:
        model = HistGradientBoostingRegressor(
            max_depth=depth, learning_rate=0.1, max_iter=150, min_samples_leaf=5, random_state=0
        )
        test = cross_validate(model, ds.X, ds.y, splits, overlap=4)
        train_ics = []
        for tr, _ in splits:
            fitted = model.fit(ds.X.iloc[tr], ds.y.iloc[tr])
            pred = pd.Series(fitted.predict(ds.X.iloc[tr]), index=ds.X.index[tr])
            train_ics.append(daily_rank_ic(pred, ds.y.iloc[tr]).mean())
        rows.append(
            {
                "tree depth": depth,
                "Training data": float(pd.Series(train_ics).mean()),
                "Unseen test data": test.summary["ic_mean"],
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    c.chapter_header(SLUG)

    c.section("What machine learning does here")
    st.markdown(
        """
In chapter 8 we tested factors **one at a time**. **Machine learning** (ML) lets a
model learn how to **combine** many signals automatically. The setup:

- **Features** (inputs): what we know about each stock on a date: past returns over
  several windows, volatility, distance from moving averages, volume, the factors.
- **Label** (target): what we want to predict: the stock's return over the **next 21
  trading days**, relative to the average of all stocks that day.
- **Training**: show the model thousands of (features → label) examples from the past
  so it learns a mapping.
- **Prediction**: give it today's features and get a score per stock.

We compare three models of increasing flexibility:
"""
    )
    c.term(
        "Momentum baseline",
        "Not really ML: it just ranks stocks by momentum. Any model must beat this to be "
        "worth its complexity.",
    )
    c.term(
        "Ridge regression",
        "A linear model: score = weighted sum of features. 'Ridge' adds a penalty that keeps "
        "weights small, which reduces overfitting.",
    )
    c.term(
        "Gradient boosting",
        "Hundreds of small decision trees, each correcting the previous ones' mistakes. Can "
        "learn non-linear patterns, and overfit them too.",
    )

    if not c.require_data():
        return
    ds = dataset()
    with c.try_it("Peek at the training data"):
        st.markdown(
            f"**{len(ds.X):,} examples** ({ds.dates.nunique()} dates × "
            f"{ds.X.index.get_level_values('symbol').nunique()} stocks), **{ds.X.shape[1]} "
            "features**. Each feature is converted to a rank between -0.5 and +0.5 on each "
            "date, so the model compares stocks rather than absolute levels."
        )
        sample = ds.X.copy()
        sample["label: next 21d relative return"] = ds.y
        st.dataframe(sample.tail(9).round(3), width="stretch")

    c.section("Overfitting: memorizing instead of learning")
    st.markdown(
        """
A flexible model can **memorize** the training data, noise included. It then looks
brilliant on data it has seen and useless on new data. Here we make gradient boosting
more and more flexible (deeper trees) and measure the IC on the training data versus on
unseen future data.
"""
    )
    with c.try_it("Make the model more complex"):
        with st.spinner("Training models of increasing depth (takes a few seconds once)..."):
            curve = depth_curve()
        long = curve.melt("tree depth", var_name="evaluated on", value_name="IC")
        chart = (
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("tree depth:O", title="Tree depth (model flexibility)"),
                y=alt.Y("IC:Q", title="Average rank IC"),
                color=alt.Color("evaluated on:N", title=None),
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")
        st.markdown(
            f"On training data the IC climbs to **{curve['Training data'].max():.2f}**: the "
            f"model 'explains' the past almost perfectly. On unseen data it stays around "
            f"**{curve['Unseen test data'].mean():+.2f}**. That gap is overfitting."
        )

    c.section("Leakage: the silent killer of ML backtests")
    st.markdown(
        """
To measure performance on unseen data we split history into training and test periods.
The classic mistake is to split **randomly** (shuffled K-fold), as you would for photos
of cats and dogs. With time series that **leaks the future**:

- Our label looks 21 days ahead. A training example from June 1 contains returns up to
  late June. If a test example sits on June 15, the model was trained on its outcome.
- Randomly shuffled folds put such neighbors on both sides of the split everywhere.

The fix is **purged walk-forward** validation: train only on the past, test on the next
block of dates, and delete ("purge") training examples whose label window overlaps the
test period, plus a small safety gap ("embargo").
"""
    )
    with c.try_it("Honest vs leaky evaluation"):
        with st.spinner("Training each model under both validation schemes..."):
            comp = model_comparison()
        long = (
            comp.rename_axis("model")
            .reset_index()
            .melt("model", var_name="validation", value_name="IC")
        )
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=200)),
                xOffset="validation:N",
                y=alt.Y("IC:Q", title="Out-of-sample mean IC"),
                color=alt.Color(
                    "validation:N",
                    scale=alt.Scale(range=["#2e7d32", "#c62828"]),
                    title=None,
                    legend=alt.Legend(orient="bottom", labelLimit=300),
                ),
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")
        gb = comp.loc[LABELS["gradient_boosting"]]
        st.markdown(
            f"Gradient boosting scores **{gb.iloc[1]:.2f}** with leaky validation, a result "
            f"that would look like a great discovery, but only **{gb.iloc[0]:+.2f}** when "
            "evaluated honestly. The baseline learns nothing, so leakage can't help it: its "
            "two bars match. Every bit of the difference is the model peeking at the future."
        )

    c.section("So does ML work?")
    st.markdown(
        """
On nine stocks, **no model reliably beats the simple momentum baseline**. That's the
expected outcome: ML needs lots of data (thousands of stocks, many years) and the
signal in stock returns is tiny. The value of this chapter isn't a money-making model;
it's knowing how to test one **without fooling yourself**, which is exactly what quant
firms look for.
"""
    )

    c.quiz(
        "ml_1",
        "A model has an IC of 0.6 on its training data and 0.0 on new data. What happened?",
        [
            "It's an excellent model",
            "It overfit: it memorized the training data, noise included",
            "The new data is wrong",
            "It needs more features",
        ],
        answer=1,
        explanation="Performance on training data says little; only unseen data counts.",
    )
    c.quiz(
        "ml_2",
        "Why is shuffled K-fold cross-validation dangerous for financial time series?",
        [
            "It's too slow",
            "Overlapping labels let the model train on information from the test period",
            "It uses too little data",
            "It only works with linear models",
        ],
        answer=1,
        explanation="With labels looking 21 days ahead, random splits leak future returns "
        "into training. Purging and walk-forward splits prevent it.",
    )
    c.takeaways(
        [
            "ML combines many features into one prediction; features in, label out.",
            "Flexible models overfit: judge them only on unseen data.",
            "Time-series validation must be walk-forward and purged to avoid leakage.",
            "Always compare against a simple baseline.",
        ]
    )
    c.chapter_footer(SLUG, ["ml_1", "ml_2"])
