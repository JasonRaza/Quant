"""Streamlit dashboard: market data, backtests, factors, portfolios and the paper account.

Run: quant-dashboard  (or: streamlit run paper_trading/dashboard.py)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.strategies import DualMovingAverageCrossover, MeanReversionPair
from data_pipeline.storage import ParquetStore
from factors.analysis import composite_score
from factors.cli import evaluate_factors, latest_screen
from factors.library import FACTORS, compute_factors
from factors.panel import build_panel
from paper_trading.service import ALLOCATION_METHODS, Settings, make_broker, target_weights
from portfolio.estimators import daily_returns
from portfolio.optimizer import Constraints
from portfolio.walkforward import (
    WalkForwardConfig,
    default_allocators,
    historical_max_sharpe,
    run_walkforward,
)

st.set_page_config(page_title="Quant Lab", page_icon="📈", layout="wide")

STRATEGY_LABELS = {"dual_ma": "Dual moving average", "mean_reversion": "Pair mean reversion"}
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
    engine = BacktestEngine(BacktestConfig(capital, commission_bps=bps, slippage_bps=bps))
    if strategy == "dual_ma":
        prices = load_symbol(symbol, _version).reset_index()
        result = engine.run(prices, DualMovingAverageCrossover(fast, slow))
    else:
        a = load_symbol(pair[0], _version)[["close"]].rename(columns={"close": "close_a"})
        b = load_symbol(pair[1], _version)[["close"]].rename(columns={"close": "close_b"})
        prices = a.join(b, how="inner").reset_index()
        result = engine.run(prices, MeanReversionPair())
    curves = pd.DataFrame({"Strategy": result.equity, "Buy & hold": result.benchmark_equity})
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


@st.cache_data
def current_weights(method: str, max_weight: float, _version: float) -> pd.Series:
    return target_weights(settings(), method, max_weight)


# ---- layout ----------------------------------------------------------------------

cfg = settings()
version = data_version()
st.title("Quant Lab")
st.caption(
    f"Universe: {', '.join(cfg.universe)} · benchmark {cfg.benchmark} · "
    "data from the local Parquet store (refresh with `quant-fetch`)"
)

tab_market, tab_bt, tab_factors, tab_port, tab_paper = st.tabs(
    ["Market", "Backtest", "Factors", "Portfolio", "Paper trading"]
)

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
            st.metric("Last close", f"{last:,.2f}", pct(last / prev - 1))
            st.metric("1-year return", pct(last / df["close"].iloc[-252] - 1))
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
            c[0].metric("Total return", pct(m["total_return"]), f"{excess:+.2%} vs B&H")
            c[1].metric("Sharpe", f"{m['sharpe']:.2f}", f"{sharpe_diff:+.2f} vs B&H")
            c[2].metric("Max drawdown", pct(m["max_drawdown"]))
            c[3].metric("Trades", n_trades)
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
                {"ic_mean": "{:.3f}", "t_stat": "{:.2f}", "hit_rate": "{:.0%}",
                 "long_short_ann_return": "{:.1%}"}
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
                {"total_return": "{:.1%}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}",
                 "avg_turnover": "{:.1%}", "total_costs": "${:,.0f}"}
            ),
            width="stretch",
        )
    st.subheader(f"Current weights: {METHOD_LABELS[method]}")
    st.bar_chart(current_weights(method, max_w, version), height=280)

with tab_paper:
    broker = make_broker(cfg)
    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    acct = broker.account()
    c = st.columns(4)
    c[0].metric("Equity", usd(acct["equity"]), pct(acct["total_return"]))
    c[1].metric("Cash", usd(acct["cash"]))
    c[2].metric("Unrealized P&L", usd(acct["unrealized_pnl"]))
    c[3].metric("Realized P&L", usd(acct["realized_pnl"]))
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
                        f"Filled {order.side} {order.quantity} {order.symbol}"
                        f" @ {order.price:,.2f}"
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
                {"avg_cost": "{:,.2f}", "price": "{:,.2f}", "market_value": usd,
                 "unrealized_pnl": usd, "weight": "{:.1%}"}
            ),
            width="stretch",
        )
    st.subheader("Order history")
    st.dataframe(broker.orders(200), width="stretch", hide_index=True)

