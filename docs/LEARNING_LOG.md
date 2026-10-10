# Learning log

What I learned building this, roughly in the order it happened. The most useful lessons
came from things that went wrong, so most entries are bugs: what the symptom was, why it
happened, and what I changed so it can't come back.

## 1. My "real" data was partly fake

**Symptom.** After fetching live data, KO, PEP and SPY had only ~500 rows each while the
other tickers had ~1,600, and their history jumped from 2023 straight to 2026.

**Cause.** The offline seed script wrote synthetic prices over existing files but didn't
touch the fetch metadata. The next incremental fetch saw an up-to-date "last date" and
appended one real bar onto two years of synthetic history.

**Fix and lesson.** The seed script now refuses to overwrite existing data unless
`--force` is passed, with a test. Idempotency has to hold for *every* writer of a
dataset, not just the main pipeline.

## 2. The backtester charged costs twice

**Symptom.** Explaining the engine line by line, I noticed the fill price already
included commission + slippage, and then the same rate was charged again as a fee.
Every trade paid roughly double the configured cost. As a side effect, cash went
slightly negative after each buy, which triggered tiny corrective trades (some worth
less than a cent) and inflated the trade count from 27 to 83.

**Fix and lesson.** Slippage now only moves the price, commission is charged once,
longs are sized net of costs, and the engine only trades when the signal changes. I
added tests that fail on the old engine: buying at a flat price must cost exactly one
slippage and one commission, and a fully invested account must never borrow cash. Tests
that check an *invariant* catch bugs that example-based tests miss.

## 3. A whole strategy had never run

**Symptom.** `quant-backtest --strategy mean_reversion` crashed with `KeyError: 'close'`,
and had since the first commit.

**Cause.** The engine trades a single `close` column; pair data only had `close_a` and
`close_b`. Nothing tested that path.

**Fix and lesson.** Pairs now trade a dollar-neutral spread series (long $1 of A, short
$1 of B) with costs on both legs, and the CLI is covered by an end-to-end test. Untested
code paths should be assumed broken.

## 4. Overlapping windows make statistics look better than they are

With 21-day forward returns, the IC computed on consecutive days reuses almost the same
returns, so the observations aren't independent and the t-stat is inflated. The factor
report therefore uses non-overlapping rebalance dates, and the ML report computes its
t-stat on non-overlapping dates only. The honest numbers are less exciting: no factor is significant on nine
stocks.

## 5. Leaky validation made a useless model look great

Under a shuffled K-fold, gradient boosting reached an out-of-sample IC of **0.24**. With
purged walk-forward validation, the same model scored **0.01**. The momentum baseline,
which learns nothing, scored the same under both schemes, which is what convinced me the
gap was entirely leakage: training rows whose 21-day label overlapped the test period.

## 6. Two features were secretly the same feature

After ranking, "1-month return" was exactly minus "1-month reversal", and "63-day
volatility" was exactly minus "low volatility". Ridge regression split its weight across
these perfectly collinear pairs. I removed the duplicates and added a test that no two
features are collinear.

## 7. Picking the universe is itself a backtest decision

Equal weight, risk parity and max Sharpe all beat SPY by a wide margin over 2021-2026
(+167% to +247% vs +124%). That's mostly because the universe is today's mega-cap
winners, chosen with hindsight. In 2020 nobody knew Nvidia would grow
like it did. The fix needs point-in-time index membership, which isn't available for
free; for now the limitation is documented everywhere the results are.

## 8. Smaller things

- Adjusted prices get rewritten when dividends are paid, so incremental fetches slowly
  drift from a fresh download. A full refresh every so often fixes it.
- An optimizer that maximizes Sharpe on trailing returns produced completely different
  portfolios from one year to the next (the tutorial shows this). Return estimates are
  far noisier than risk estimates.
- In Streamlit, a `$` in markdown starts a LaTeX formula, so "$1 grows to $1.78"
  rendered as garbled math until I escaped dollar signs.

## What I'd study next

- Point-in-time universes and a broad market (S&P 500 constituents through time).
- Fundamental data for value and quality factors.
- Transaction cost models with market impact; execution with limit orders.
- Nested purged cross-validation for hyperparameter tuning.
