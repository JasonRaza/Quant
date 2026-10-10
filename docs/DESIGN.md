# Design decisions

Notes on the main choices I made, what I considered instead, and what each choice
costs. Most of these I'd revisit at a larger scale; for a learning project the priority
was code I could read, test and explain.

## Data

**Parquet files + a SQLite metadata table, not a database server.**
One Parquet file per ticker is fast to read with pandas, compresses well and needs no
running service. A tiny SQLite table remembers the last date fetched per ticker, which
makes the pipeline incremental (only new days are downloaded) and idempotent (running
it twice changes nothing).
*Cost:* no concurrent writers, no query language across tickers. Fine for ten tickers;
a real setup would use a columnar database or a data lake with partitioning.

**Adjusted prices from Yahoo Finance.**
Free and good enough to learn with. Split- and dividend-adjusted closes make returns
correct across corporate actions.
*Cost:* adjusted history is rewritten whenever a new dividend is paid, so an
incremental pipeline slowly drifts and needs a periodic full refresh. The universe is
also survivor-biased (see [LEARNING_LOG.md](LEARNING_LOG.md)).

## Backtesting

**An explicit day-by-day loop instead of a fully vectorized backtest.**
The engine walks through each bar, tracks cash and shares, and trades when the signal
changes. It's slower than vectorized pandas, but every number in the account can be
traced back to a line of code, and cash constraints and per-trade costs are natural to
express. The tutorial's parameter grid search, where speed matters, uses a vectorized
version of the same rule.

**Decide on yesterday's information, trade at today's close.**
Every strategy shifts its signal by one bar. A test perturbs future prices and checks
that past signals and factor values don't change, which catches lookahead bugs
automatically.

**Cost model: slippage on the fill price, commission on notional.**
Slippage makes buys fill slightly higher and sells slightly lower; commission is a
percentage of the trade's value, charged once. Long positions are sized net of both so
the account never goes into negative cash. Pair trades pay costs on both legs.
*Cost:* no market impact or borrowing fees; realistic for small, liquid trades only.

## Research

**Rank-based (cross-sectional) signals.**
Factors and ML features are converted to ranks across stocks on each date. Ranks are
robust to outliers and to regime changes in scale (volatility in 2020 vs 2024), and
they match the question being asked: which stock beats the others, not where the market
goes.

**Non-overlapping evaluation dates for IC statistics.**
With 21-day forward returns, consecutive daily ICs share most of their window and are
strongly correlated, so a naive t-stat is inflated. The factor report samples every
21st day; the ML report computes its t-stat on non-overlapping dates only.

**Purged walk-forward cross-validation for ML.**
Train on the past, test on the next block, and drop training rows whose label window
reaches into the test block (plus a small embargo). A shuffled K-fold is kept in the
report only to show how much it inflates results.

## Portfolio construction

**SciPy's SLSQP instead of a convex solver like cvxpy.**
The problems are tiny (nine weights), and two objectives (max Sharpe, risk parity)
aren't convex quadratic programs in their natural form. SLSQP handles all of them with
one dependency already in the stack.
*Cost:* it finds local optima and can fail to converge; the code raises an error rather
than silently returning bad weights, and closed-form cases are checked in tests.

**Ledoit-Wolf shrinkage implemented from the paper.**
Writing it myself was the point; a test checks it matches scikit-learn's implementation
to 1e-8.

**Walk-forward allocator comparison with drift and costs.**
Weights are re-estimated monthly on the trailing year only, then drift with prices
until the next rebalance, paying costs on turnover. This is closer to what running the
portfolio would look like than comparing optimizers on the full sample.

## Application

**Streamlit for the UI, FastAPI for the API.**
Streamlit let me build the dashboard and the tutorial in Python alone, and its testing
API (`AppTest`) lets every page run in the test suite. FastAPI exposes the same paper
account to other programs with request validation for free.
*Cost:* Streamlit reruns the whole script on every interaction, so expensive steps are
cached explicitly; a production front end would likely be a separate web app.

**One SQLite paper account shared by the dashboard and the API.**
Simple and inspectable. Whole-share, long-only market orders keep the accounting easy to
verify in tests; rejected orders are stored with a reason instead of raising.

## Testing

- **Synthetic market fixtures** let CLI, dashboard and tutorial tests run offline, so CI
  needs no network or API keys.
- **Invariant tests** encode properties that must always hold: buying at a flat price
  costs exactly one slippage plus one commission; changing future prices never changes
  past signals; purged training rows never overlap the test window.
- **Reference tests** compare against known answers: closed-form minimum variance and
  tangency portfolios, scikit-learn's Ledoit-Wolf.
- **UI smoke tests** render every tutorial chapter and the dashboard with `AppTest`.
