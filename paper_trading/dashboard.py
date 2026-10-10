"""Streamlit dashboard: market data, backtests, factors, portfolios and the paper account.

Rendered as the Lab page of paper_trading/app.py (run: quant-dashboard).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.strategies import (
    DualMovingAverageCrossover,
    MeanReversionPair,
    pair_spread_prices,
)
from data_pipeline.storage import ParquetStore
from factors.analysis import composite_score
from factors.cli import evaluate_factors, latest_screen
from factors.library import FACTORS, compute_factors
from factors.panel import build_panel
from ml_alpha.cli import evaluate_models
from ml_alpha.features import build_dataset, latest_features
from ml_alpha.models import default_models, ridge_coefficients
from paper_trading.broker import PaperBroker, ParquetPriceSource
from paper_trading.service import ALLOCATION_METHODS, Settings, make_broker, target_weights
from portfolio.estimators import daily_returns
from portfolio.optimizer import Constraints
from portfolio.walkforward import (
    WalkForwardConfig,
    default_allocators,
    historical_max_sharpe,
    run_walkforward,
)

STRATEGY_LABELS = {"dual_ma": "Dual moving average", "mean_reversion": "Pair mean reversion"}
MODEL_LABELS = {
    "momentum_baseline": "Momentum baseline",
    "ridge": "Ridge",
    "gradient_boosting": "Gradient boosting",
}
METHOD_LABELS = {
    "equal_weight": "Equal weight",
    "min_variance": "Min variance",
    "risk_parity": "Risk parity",
    "factor_mean_variance": "Factor mean-variance",
    "max_sharpe_historical": "Max Sharpe (trailing mean)",
}


def pct(x: float) -> str:
    return f"{x:.2%}"


def usd(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


@st.cache_resource
def settings() -> Settings:
    return Settings.from_config()


def data_version() -> float:
    """Latest Parquet mtime, used to invalidate caches after quant-fetch."""
    files = list(settings().data_dir.glob("*.parquet"))
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data
def load_symbol(symbol: str, _version: float) -> pd.DataFrame:
    df = ParquetStore(settings().data_dir).load(symbol)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


@st.cache_data
def run_backtest(strategy, symbol, pair, fast, slow, capital, bps, _version):
    legs = 1 if strategy == "dual_ma" else 2  # pair trades pay costs on both legs
    engine = BacktestEngine(BacktestConfig(capital, bps * legs, bps * legs))
    if strategy == "dual_ma":
        prices = load_symbol(symbol, _version).reset_index()
        result = engine.run(prices, DualMovingAverageCrossover(fast, slow))
    else:
        a = load_symbol(pair[0], _version)[["close"]].rename(columns={"close": "close_a"})
        b = load_symbol(pair[1], _version)[["close"]].rename(columns={"close": "close_b"})
        prices = pair_spread_prices(a.join(b, how="inner").reset_index())
        result = engine.run(prices, MeanReversionPair())
    bench_label = "Buy & hold" if strategy == "dual_ma" else "Long spread & hold"
    curves = pd.DataFrame({"Strategy": result.equity, bench_label: result.benchmark_equity})
    return result.metrics, curves, len(result.trades)


@st.cache_data
def run_factor_report(universe: tuple, horizon: int, quantiles: int, _version: float):
    panel = build_panel(ParquetStore(settings().data_dir), list(universe))
    factors = compute_factors(panel)
    report, ic_series = evaluate_factors(factors, panel.close, horizon, quantiles)
    cum_ic = pd.DataFrame({k: v.cumsum() for k, v in ic_series.items()})
    return report, cum_ic, latest_screen(factors)


@st.cache_data
def run_portfolio_walkforward(universe: tuple, max_weight: float, cost_bps: float, _version):
    panel = build_panel(ParquetStore(settings().data_dir), list(universe))
    returns = daily_returns(panel.close)
    cons = Constraints(max_weight=max_weight)
    allocators = default_allocators(cons, composite_score(compute_factors(panel)))
    allocators["max_sharpe_historical"] = historical_max_sharpe(cons)
    cfg = WalkForwardConfig(cost_bps=cost_bps)
    metrics, curves = {}, {}
    for name, alloc in allocators.items():
        res = run_walkforward(returns, alloc, cfg)
        metrics[METHOD_LABELS[name]] = res.metrics
        curves[METHOD_LABELS[name]] = res.equity
    return pd.DataFrame(metrics).T, pd.DataFrame(curves)


@st.cache_data(show_spinner=False)
def run_ml(universe: tuple, horizon: int, n_splits: int, embargo: int, _version: float):
    panel = build_panel(ParquetStore(settings().data_dir), list(universe))
    ds = build_dataset(panel, horizon=horizon, sample_every=5)
    results, ic_series, period = evaluate_models(ds, horizon, 5, n_splits, 52, embargo)
    table = pd.DataFrame(
        {
            MODEL_LABELS[name]: {
                **r["purged_walk_forward"],
                "long_short_ann_return": r["long_short_ann_return"],
                "leaky_ic_mean": r["leaky_shuffled_kfold_ic_mean"],
            }
            for name, r in results.items()
        }
    ).T
    cum_ic = pd.DataFrame({MODEL_LABELS[k]: v.cumsum() for k, v in ic_series.items()})
    coefs = ridge_coefficients(default_models()["ridge"], ds.X, ds.y)

    # Today's ranking: fit each model on every labelled sample, score the latest features.
    latest = latest_features(panel)
    preds = pd.DataFrame(
        {
            MODEL_LABELS[name]: model.fit(ds.X, ds.y).predict(latest[ds.X.columns])
            for name, model in default_models().items()
        },
        index=latest.index,
    )
    meta = {
        "samples": len(ds.X),
        "features": ds.X.shape[1],
        "period": period,
        "as_of": panel.close.index[-1],
    }
    return table, cum_ic, coefs, preds, meta


def paper_broker(cfg: Settings) -> PaperBroker:
    """The shared local account, or a per-session throwaway account in public demo mode."""
    if not cfg.public_demo:
        return make_broker(cfg)
    if "demo_paper_db" not in st.session_state:
        tmp = Path(tempfile.mkdtemp(prefix="quant_demo_"))
        st.session_state["demo_paper_db"] = str(tmp / "paper.sqlite")
    return PaperBroker(st.session_state["demo_paper_db"], ParquetPriceSource(cfg.data_dir))


@st.cache_data
def current_weights(method: str, max_weight: float, _version: float) -> pd.Series:
    return target_weights(settings(), method, max_weight)


# ---- layout ----------------------------------------------------------------------

cfg = settings()
version = data_version()
with st.container(horizontal=True, vertical_alignment="center", gap="small"):
    st.badge("Research dashboard", icon=":material/science:", color="primary")
    st.badge(f"{len(cfg.universe)} stocks + {cfg.benchmark}", icon=":material/apps:", color="gray")
    st.badge(
        f"Data through {load_symbol(cfg.benchmark, version).index[-1].date()}",
        icon=":material/calendar_month:",
        color="gray",
    )
st.title("Quant Lab")
st.markdown(
    f":gray[Universe: {', '.join(cfg.universe)}. Benchmark {cfg.benchmark}. "
    "Data from the local Parquet store (refresh with `quant-fetch`).]"
)

TAB_LABELS = [
    ":material/candlestick_chart: Market",
    ":material/history: Backtest",
    ":material/leaderboard: Factors",
    ":material/donut_large: Portfolio",
    ":material/neurology: ML alpha",
    ":material/receipt_long: Paper trading",
]
tab_market, tab_bt, tab_factors, tab_port, tab_ml, tab_paper = st.tabs(TAB_LABELS)

with tab_market:
    symbols = [cfg.benchmark] + cfg.universe
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.selectbox("Symbol", symbols)
        show_ma = st.multiselect("Moving averages", [20, 50, 200], default=[50, 200])
    df = load_symbol(symbol, version)
    if df.empty:
        st.warning(f"No data for {symbol}. Run `quant-fetch` first.")
    else:
        chart = df[["close"]].rename(columns={"close": symbol})
        for n in show_ma:
            chart[f"MA{n}"] = df["close"].rolling(n).mean()
        with col1:
            last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
            st.metric(
                "Last close",
                f"{last:,.2f}",
                pct(last / prev - 1),
                border=True,
                chart_data=df["close"].iloc[-63:].round(2).tolist(),
                chart_type="area",
            )
            st.metric("1-year return", pct(last / df["close"].iloc[-252] - 1), border=True)
            st.caption(f"As of {df.index[-1].date()} · {len(df):,} bars")
        with col2:
            st.line_chart(chart, height=420)

with tab_bt:
    col1, col2 = st.columns([1, 3])
    with col1:
        strategy = st.radio("Strategy", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get)
        bt_symbol, pair, fast, slow = cfg.benchmark, ("KO", "PEP"), 20, 50
        if strategy == "dual_ma":
            bt_symbol = st.selectbox("Symbol", [cfg.benchmark] + cfg.universe, key="bt_symbol")
            fast = st.slider("Fast MA", 5, 100, 20)
            slow = st.slider("Slow MA", 20, 250, 50)
        else:
            a = st.selectbox("Leg A", cfg.universe, index=cfg.universe.index("KO"))
            b = st.selectbox("Leg B", cfg.universe, index=cfg.universe.index("PEP"))
            pair = (a, b)
        bps = st.slider("Commission & slippage (bps each)", 0.0, 25.0, 5.0, 0.5)
    with col2:
        if strategy == "dual_ma" and fast >= slow:
            st.error("Fast MA must be shorter than slow MA.")
        elif strategy == "mean_reversion" and pair[0] == pair[1]:
            st.error("Pick two different symbols.")
        else:
            m, curves, n_trades = run_backtest(
                strategy, bt_symbol, pair, fast, slow, 100_000.0, bps, version
            )
            c = st.columns(4)
            excess = m["total_return"] - m["benchmark_total_return"]
            sharpe_diff = m["sharpe"] - m["benchmark_sharpe"]
            c[0].metric(
                "Total return", pct(m["total_return"]), f"{excess:+.2%} vs benchmark", border=True
            )
            c[1].metric(
                "Sharpe", f"{m['sharpe']:.2f}", f"{sharpe_diff:+.2f} vs benchmark", border=True
            )
            c[2].metric("Max drawdown", pct(m["max_drawdown"]), border=True)
            c[3].metric("Trades", n_trades, border=True)
            st.line_chart(curves, height=380)

with tab_factors:
    col1, col2 = st.columns([1, 3])
    with col1:
        horizon = st.select_slider("Forward horizon (days)", [5, 10, 21, 63], value=21)
        quantiles = st.slider("Quantiles", 2, 5, 3)
    report, cum_ic, screen = run_factor_report(tuple(cfg.universe), horizon, quantiles, version)
    table = pd.DataFrame(report).T[["ic_mean", "t_stat", "hit_rate", "n", "long_short_ann_return"]]
    with col2:
        st.subheader("Rank IC by factor")
        st.dataframe(
            table.style.format(
                {
                    "ic_mean": "{:.3f}",
                    "t_stat": "{:.2f}",
                    "hit_rate": "{:.0%}",
                    "long_short_ann_return": "{:.1%}",
                }
            ),
            width="stretch",
        )
        st.caption("|t| > 2 would be needed to call a factor significant.")
        st.line_chart(cum_ic, height=300)
    st.subheader(f"Latest screen ({screen['as_of']})")
    st.dataframe(
        pd.DataFrame(screen["rankings"]).set_index("rank"),
        width="stretch",
        column_config={name: st.column_config.NumberColumn(format="%.3f") for name in FACTORS},
    )

with tab_port:
    col1, col2 = st.columns([1, 3])
    with col1:
        max_w = st.slider("Max weight per asset", 0.12, 1.0, 0.25, 0.01)
        cost = st.slider("Rebalance cost (bps)", 0.0, 50.0, 10.0, 1.0)
        method = st.selectbox("Show weights for", ALLOCATION_METHODS, format_func=METHOD_LABELS.get)
    with st.spinner("Running walk-forward backtests..."):
        wf_metrics, wf_curves = run_portfolio_walkforward(tuple(cfg.universe), max_w, cost, version)
    with col2:
        st.subheader("Walk-forward comparison (monthly rebalance, net of costs)")
        st.line_chart(wf_curves, height=340)
        cols = ["total_return", "sharpe", "max_drawdown", "avg_turnover", "total_costs"]
        st.dataframe(
            wf_metrics[cols].style.format(
                {
                    "total_return": "{:.1%}",
                    "sharpe": "{:.2f}",
                    "max_drawdown": "{:.1%}",
                    "avg_turnover": "{:.1%}",
                    "total_costs": "${:,.0f}",
                }
            ),
            width="stretch",
        )
    st.subheader(f"Current weights: {METHOD_LABELS[method]}")
    st.bar_chart(current_weights(method, max_w, version), height=280)

with tab_ml:
    col1, col2 = st.columns([1, 3])
    with col1:
        ml_horizon = st.select_slider(
            "Forward horizon (days)", [5, 10, 21, 63], value=21, key="ml_horizon"
        )
        ml_splits = st.slider("Walk-forward splits", 2, 8, 5)
        ml_embargo = st.slider("Embargo (days)", 0, 30, 5)
        st.caption(
            "Purged CV trains only on the past and drops samples whose label window "
            "reaches into the test period. The leaky column scores the same models with "
            "a shuffled K-fold that ignores time."
        )
    with st.spinner("Training models across folds..."):
        ml_table, ml_cum_ic, ml_coefs, ml_preds, ml_meta = run_ml(
            tuple(cfg.universe), ml_horizon, ml_splits, ml_embargo, version
        )
    with col2:
        start, end = ml_meta["period"]
        st.subheader("Out-of-sample results")
        st.caption(
            f"{ml_meta['samples']:,} samples · {ml_meta['features']} features · "
            f"tested {start.date()} to {end.date()}"
        )
        cols = ["ic_mean", "t_stat", "hit_rate", "long_short_ann_return", "leaky_ic_mean"]
        st.dataframe(
            ml_table[cols]
            .rename(
                columns={
                    "ic_mean": "IC (purged)",
                    "t_stat": "t-stat",
                    "hit_rate": "Hit rate",
                    "long_short_ann_return": "Long-short (ann.)",
                    "leaky_ic_mean": "IC (leaky CV)",
                }
            )
            .style.format(
                {
                    "IC (purged)": "{:.3f}",
                    "t-stat": "{:.2f}",
                    "Hit rate": "{:.0%}",
                    "Long-short (ann.)": "{:+.1%}",
                    "IC (leaky CV)": "{:.3f}",
                }
            ),
            width="stretch",
        )
        st.subheader("Purged vs leaky CV")
        st.bar_chart(
            ml_table[["ic_mean", "leaky_ic_mean"]].rename(
                columns={"ic_mean": "Purged walk-forward", "leaky_ic_mean": "Leaky shuffled K-fold"}
            ),
            stack=False,
            height=260,
        )
        st.caption(
            "The gap between the two bars is what leakage would add to a model's apparent skill."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative out-of-sample IC")
        st.line_chart(ml_cum_ic, height=300)
    with right:
        st.subheader("Ridge coefficients (full sample)")
        st.bar_chart(ml_coefs.rename("coefficient"), horizontal=True, height=300)

    st.subheader(f"Today's ranking ({ml_meta['as_of'].date()})")
    rank_model = st.selectbox("Rank by", list(ml_preds.columns), index=1)
    ranking = ml_preds.sort_values(rank_model, ascending=False)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    st.dataframe(
        ranking,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ml_preds.columns},
    )
    st.caption(
        f"Ridge and gradient boosting scores are predicted {ml_horizon}-day returns relative "
        "to the universe average; the baseline score is the momentum rank (-0.5 to 0.5). "
        "No model here is statistically significant, so treat the ranking as a demo."
    )

with tab_paper:
    broker = paper_broker(cfg)
    if cfg.public_demo:
        st.info(
            "Public demo: this paper account is private to your browser session and resets "
            "when you close the page.",
            icon=":material/visibility_lock:",
        )
    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    acct = broker.account()
    c = st.columns(4)
    c[0].metric("Equity", usd(acct["equity"]), pct(acct["total_return"]), border=True)
    c[1].metric("Cash", usd(acct["cash"]), border=True)
    c[2].metric("Unrealized P&L", usd(acct["unrealized_pnl"]), border=True)
    c[3].metric("Realized P&L", usd(acct["realized_pnl"]), border=True)
    st.caption(
        "Orders fill at the latest close in the local data plus 5 bps slippage and 5 bps "
        "commission. Prices only move when you refresh data with `quant-fetch`."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Place an order")
        with st.form("order"):
            o_sym = st.selectbox("Symbol", [cfg.benchmark] + cfg.universe, key="o_sym")
            o_side = st.radio("Side", ["buy", "sell"], horizontal=True)
            o_qty = st.number_input("Quantity (shares)", min_value=1, value=10, step=1)
            if st.form_submit_button("Submit order"):
                order = broker.submit_order(o_sym, o_side, int(o_qty))
                if order.status == "filled":
                    msg = (
                        f"Filled {order.side} {order.quantity} {order.symbol} @ {order.price:,.2f}"
                    )
                    st.session_state["flash"] = ("success", msg)
                else:
                    st.session_state["flash"] = ("error", f"Rejected: {order.reason}")
                st.rerun()
    with right:
        st.subheader("Rebalance to an allocator")
        with st.form("rebalance"):
            r_method = st.selectbox("Allocator", ALLOCATION_METHODS, format_func=METHOD_LABELS.get)
            r_max = st.slider("Max weight", 0.12, 1.0, 0.25, 0.01, key="r_max")
            if st.form_submit_button("Rebalance"):
                weights = current_weights(r_method, r_max, version)
                orders = broker.rebalance(weights.to_dict())
                filled = sum(o.status == "filled" for o in orders)
                label = METHOD_LABELS[r_method]
                msg = f"Rebalanced to {label}: {filled}/{len(orders)} orders filled"
                st.session_state["flash"] = ("success", msg)
                st.rerun()
        with st.expander("Reset account"):
            new_cash = st.number_input(
                "Starting cash", min_value=1_000.0, value=100_000.0, step=1_000.0
            )
            if st.button("Reset", type="secondary"):
                broker.reset(new_cash)
                st.session_state["flash"] = ("info", f"Account reset to ${new_cash:,.0f}")
                st.rerun()

    st.subheader("Positions")
    pos = broker.positions()
    if pos.empty:
        st.info("No positions yet.")
    else:
        pos["weight"] = pos["market_value"] / acct["equity"]
        st.dataframe(
            pos.set_index("symbol").style.format(
                {
                    "avg_cost": "{:,.2f}",
                    "price": "{:,.2f}",
                    "market_value": usd,
                    "unrealized_pnl": usd,
                    "weight": "{:.1%}",
                }
            ),
            width="stretch",
        )
    st.subheader("Order history")
    st.dataframe(broker.orders(200), width="stretch", hide_index=True)
