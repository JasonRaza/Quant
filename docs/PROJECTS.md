# Project details

Technical notes for each part of the repo: what it does, how to run it, results on
real data, and known limitations. The [README](../README.md) has the overview; design
trade-offs are in [DESIGN.md](DESIGN.md) and the story of how it was built (and what
broke) is in [LEARNING_LOG.md](LEARNING_LOG.md).

Contents: [1. Data pipeline](#project-1-market-data-pipeline) ·
[2. Backtester](#project-2-backtest-engine) · [3. Factors](#project-3-factor-screener) ·
[4. Portfolio](#project-4-portfolio-optimizer) · [5. ML](#project-5-ml-alpha-baseline) ·
[6. Paper trading](#project-6-paper-trading-dashboard) ·
[Tutorial](#interactive-tutorial)

## Project 1: Market Data Pipeline

- CLI: `quant-fetch`
- Normalized schema: `symbol`, `date`, `open`, `high`, `low`, `close`, `volume`
- Storage: Parquet per symbol under `data/parquet/`
- Metadata: SQLite tracks `last_date`, row counts, errors
- Idempotent incremental fetches (skips up-to-date symbols)
- Retries with configurable delay
- Quality report: `quant-quality` (missing %, stale symbols, volume spikes)

### Limitations

- Yahoo Finance data may have gaps and is not adjusted for all corporate actions consistently
- Prices are split/dividend-adjusted, but incremental fetches don't re-adjust older bars when a new dividend is paid, so do a full re-fetch periodically
- No survivorship-bias-free index membership history in v1
- Approximate trading-day missing % (weekends/holidays not fully modeled)

## Project 2: Backtest Engine

- Strategies: `DualMovingAverageCrossover`, `MeanReversionPair` (KO/PEP)
- Costs: slippage moves the fill price and commission is charged once on notional (default 5 bps each)
- Trades only when the signal changes; longs are sized net of costs so cash never goes negative
- Metrics: total return, Sharpe, max drawdown, win rate, turnover
- Benchmark: buy-and-hold comparison
- No lookahead: signals shifted by one bar before execution

```bash
# Dual MA on SPY
quant-backtest --strategy dual_ma --symbol SPY --fast 20 --slow 50

# Mean reversion on KO/PEP pair
quant-backtest --strategy mean_reversion --pair KO PEP
```

Pair trades run on a dollar-neutral spread (long $1 of A, short $1 of B, rebalanced daily), and costs are charged on both legs.

Outputs: `reports/backtest_results.json`, `reports/equity_curve.png`

### Results (live data, dual-MA 20/50 on SPY)

Run on Yahoo Finance daily bars (split- and dividend-adjusted), Jan 2, 2020 – Oct 8, 2026 (`quant-fetch`):

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total return | 82.47% | 162.00% |
| Sharpe (ann.) | 0.80 | 0.81 |
| Max drawdown | -28.08% | -33.72% |
| Trades | 27 | — |
| Costs paid | $3,668 | — |

The crossover cuts the drawdown a little but gives up about half of the buy-and-hold return after costs. That's a typical result for trend filters on a strongly trending index. *Past performance does not guarantee future results.*

KO/PEP mean reversion over the same period (20-day z-score, enter at ±2, exit at ±0.5) loses money: -22.7% total return, Sharpe -0.31, 125 trades, against +49.3% for simply holding the long-KO/short-PEP spread. The raw price-difference spread isn't stable enough to fade, and the round trips get eaten by double-leg costs.

![Equity curve](../reports/equity_curve.png)

## Project 3: Factor Screener

- CLI: `quant-factors`
- Factors (`factors/library.py`): momentum 12-1, 1-month reversal, low volatility, liquidity (log dollar volume)
- Evaluation (`factors/analysis.py`): Spearman rank IC per rebalance date, IC mean / IR / t-stat / hit rate, quantile forward returns and top-minus-bottom spread
- Rebalance dates are spaced by the horizon (default 21 days) so forward-return windows don't overlap and t-stats aren't inflated
- Screen: equal-weight composite of cross-sectional z-scores, ranked on the latest date
- No lookahead: a test perturbs future prices and checks past factor values are unchanged

```bash
quant-factors --exclude SPY --horizon 21 --quantiles 3
```

Outputs: `reports/factor_report.json`, `reports/factor_ic.png`

### Results (9 large caps, Jan 2020 – Oct 8, 2026, 21-day horizon)

| Factor | IC mean | t-stat | Hit rate |
|--------|---------|--------|----------|
| momentum_12_1 | 0.045 | 0.93 | 59% |
| reversal_1m | 0.004 | 0.11 | 52% |
| low_vol | -0.055 | -0.98 | 47% |
| liquidity | 0.047 | 0.83 | 47% |

None of the factors is statistically significant (|t| < 2). With only 9 names, each cross-section is very noisy, so treat this as a working pipeline, not as evidence of alpha. Low-vol's negative IC matches the 2020–2026 period, when high-beta tech led the market.

![Cumulative IC](../reports/factor_ic.png)

### Limitations

- Tiny, survivor-biased universe (today's mega caps), so you need many more names for meaningful IC
- Price/volume factors only; value and quality need fundamentals data

## Project 4: Portfolio Optimizer

- CLI: `quant-optimize`
- Covariance (`portfolio/estimators.py`): Ledoit-Wolf shrinkage toward scaled identity, implemented from the paper and checked against scikit-learn in tests
- Expected returns: trailing mean, or factor alpha `IC × vol × z-score` from the Project 3 composite (Grinold's rule)
- Optimizers (`portfolio/optimizer.py`, SciPy SLSQP): min variance, mean-variance, max Sharpe, risk parity (equal risk contribution), equal weight. Long-only, fully invested, per-asset cap.
- Walk-forward (`portfolio/walkforward.py`): monthly rebalance, 252-day trailing estimation window only, 10 bps cost on turnover, weights drift between rebalances

```bash
quant-optimize --exclude SPY --max-weight 0.25 --cost-bps 10
```

Outputs: `reports/portfolio_report.json` (walk-forward metrics, current weights, risk contributions), `reports/portfolio_equity.png`

### Results (9 large caps, Jan 2021 – Oct 8, 2026, net of costs)

| Allocator | Total return | Sharpe | Max DD | Avg turnover / rebalance | Effective holdings |
|-----------|--------------|--------|--------|--------------------------|--------------------|
| Equal weight | 238.4% | 1.13 | -34.9% | 5% | 9.0 |
| Min variance | 106.1% | 0.95 | -22.0% | 9% | 4.9 |
| Risk parity | 167.0% | 1.13 | -26.9% | 6% | 7.4 |
| Factor mean-variance | 115.5% | 0.99 | -20.4% | 24% | 4.9 |
| Max Sharpe (trailing mean) | 246.5% | 1.19 | -25.2% | 29% | 4.6 |
| SPY buy & hold | 123.5% | 0.92 | -24.5% | — | — |

What this shows:
- Min variance and risk parity deliver what they promise: lower drawdowns than equal weight. They pay for it in return, because they tilt toward KO, PEP and JPM.
- Max Sharpe on trailing means came out on top, but it turns over ~6× more than equal weight and is notoriously unstable out of sample. It won here because big tech kept trending, so don't read it as a robust result.
- The factor tilt didn't add value. That's consistent with the insignificant ICs from Project 3.

![Portfolio equity](../reports/portfolio_equity.png)

### Limitations

- The universe is today's mega-cap winners, so the allocators that beat SPY do so mostly because of survivorship bias
- Single 9-asset universe and a single period, so differences between allocators are not statistically tested
- Costs are a flat bps on turnover, with no market impact or borrow

## Project 5: ML Alpha Baseline

- CLI: `quant-ml`
- Dataset (`ml_alpha/features.py`): one row per (date, symbol), sampled weekly. 11 cross-sectionally ranked features: short/medium-term returns, 21-day volatility, distance to the 50/200-day MA, volume z-score, and the four Project 3 factors.
- Label: 21-day forward return, demeaned across symbols, so models predict which names beat the others rather than market direction
- CV (`ml_alpha/cv.py`): expanding-window walk-forward split by date. Training samples whose 21-day label window reaches into the test block are **purged**, plus a 5-day embargo.
- Models (`ml_alpha/models.py`): momentum-rank baseline (no fitting), ridge regression, gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`)
- Evaluation: out-of-sample rank IC per date, t-stat on non-overlapping dates only, top-minus-bottom tercile spread. Each model is also scored with a shuffled K-fold to show the leakage effect.

```bash
quant-ml --exclude SPY --horizon 21 --splits 5 --embargo-days 5
```

Outputs: `reports/ml_report.json`, `reports/ml_oos_ic.png`. The same evaluation is interactive in the dashboard's **ML alpha** tab.

### Results (9 large caps, out-of-sample Jan 2022 – Sep 2026)

| Model | OOS IC (purged) | t-stat | Long-short (ann.) | IC with leaky shuffled K-fold |
|-------|-----------------|--------|-------------------|-------------------------------|
| Momentum baseline | 0.031 | 0.55 | -0.8% | 0.031 |
| Ridge | 0.038 | 0.75 | +1.5% | 0.085 |
| Gradient boosting | 0.007 | 0.14 | -2.2% | **0.243** |

What this shows:
- **Leakage is the headline.** Under a shuffled K-fold, gradient boosting looks like a strong signal (IC 0.24). Evaluated properly, it's indistinguishable from zero. The baseline, which learns nothing, scores the same under both schemes, which confirms the gap comes from training on overlapping future labels.
- Ridge only matches the one-feature momentum baseline, and nothing is statistically significant. With 9 stocks there's simply too little cross-sectional signal to learn.

![ML out-of-sample IC](../reports/ml_oos_ic.png)

### Limitations

- 9 names × ~285 weekly dates: far too small for ML to beat simple baselines; a broad universe (e.g. S&P 500) is the real test
- Hyperparameters are fixed, not tuned. Tuning would need a nested purged CV to avoid a new layer of leakage.
- Price/volume features only; no fundamentals, sentiment or alternative data

## Project 6: Paper Trading Dashboard

```bash
quant-dashboard     # Streamlit app (tutorial + dashboard) on http://localhost:8501
quant-paper-api     # REST API, interactive docs on http://127.0.0.1:8000/docs
```

**Dashboard tabs** (Lab page, `paper_trading/dashboard.py`):
- **Market**: price chart with moving averages for any watchlist symbol
- **Backtest**: dual-MA or pair mean reversion with sliders for windows and costs, metrics vs buy & hold
- **Factors**: rank IC table, cumulative IC chart and latest composite screen, with horizon/quantile controls
- **Portfolio**: walk-forward comparison of all allocators, with max-weight and cost sliders, plus today's weights
- **ML alpha**: purged vs leaky CV comparison, out-of-sample IC table and chart, ridge coefficients, today's model rankings (horizon, splits and embargo sliders)
- **Paper trading**: account equity and P&L, order ticket, one-click rebalance to any allocator, positions, order history, reset

**Paper broker** (`paper_trading/broker.py`):
- Long-only cash account persisted in SQLite (`data/paper_trading.sqlite`, git-ignored), shared by the dashboard and the API
- Whole-share market orders filled at the latest local close ± 5 bps slippage, plus 5 bps commission
- Average-cost positions, realized and unrealized P&L
- Rejected orders (insufficient cash, overselling, unknown symbol) are logged with a reason
- Rebalancing sells first to free cash, sizes buys net of costs, and liquidates holdings outside the target

**API** (`paper_trading/api.py`): `GET /account`, `/positions`, `/orders`, `/prices/{symbol}`, `/allocations/{method}`; `POST /orders`, `/rebalance`, `/reset`. Rejected orders return HTTP 400 with the reason.

### Limitations

- Prices are end-of-day closes from the local store, so fills only "move" after `quant-fetch`. There's no intraday feed or live broker connection.
- Market orders only (no limit/stop orders), single account, no authentication. The API is meant to run on localhost.


## Interactive tutorial

`quant-dashboard` opens a web app whose **Learn** section is a beginner tutorial: no prior knowledge of trading, statistics or ML assumed. Every chapter mixes plain-English explanations with interactive experiments on the real data and real project code, plus quizzes, takeaways and progress tracking.

| # | Chapter | Hands-on experiments |
|---|---------|----------------------|
| 0 | Start here | What $1,000 in each stock became, by start year |
| 1 | Markets and prices | Live candlestick + volume charts |
| 2 | Returns and risk | Invest in the past, the recovery problem, volatility histograms, underwater charts |
| 3 | Comparing investments | Risk/return map, correlation heatmap, beta regression |
| 4 | Market data | Split vs adjusted prices, data health check, survivorship-bias simulation |
| 5 | Strategies and backtesting | Moving averages, the crossover rule step by step, trade log, cost sensitivity |
| 6 | Backtesting pitfalls | Lookahead bias, in- vs out-of-sample grid search, random-strategy lottery |
| 7 | Mean reversion and pairs | Z-score bands, pairs backtest, rolling correlation |
| 8 | Factors | Today's rankings, IC on a single date, real vs random factors, quantile staircase |
| 9 | Building portfolios | Two-stock diversification, efficient frontier, weight instability, walk-forward |
| 10 | Machine learning | Training data, overfitting vs tree depth, purged vs leaky validation |
| 11 | From research to trading | Order types, a private sandbox paper account, production checklist |
| 12 | Glossary and final quiz | Searchable glossary, 8-question final quiz |

The **Lab** section is the research dashboard described above. Tutorial code lives in `paper_trading/tutorial/`; `tests/test_tutorial.py` renders every chapter on synthetic data.
