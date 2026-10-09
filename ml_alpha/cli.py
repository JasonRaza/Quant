"""CLI: out-of-sample ML alpha evaluation with purged CV vs leaky CV."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.storage import ParquetStore
from factors.panel import build_panel
from ml_alpha.cv import PurgedWalkForward, leaky_kfold
from ml_alpha.features import Dataset, build_dataset
from ml_alpha.models import cross_validate, default_models, ridge_coefficients
from shared.config import load_yaml, resolve_path

logger = logging.getLogger(__name__)


def save_ic_chart(ic_series: dict[str, pd.Series], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, ic in ic_series.items():
        if not ic.empty:
            ic.cumsum().plot(ax=ax, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative out-of-sample rank IC (purged walk-forward)")
    ax.set_xlabel("Prediction date")
    ax.set_ylabel("Cumulative IC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def evaluate_models(
    ds: Dataset,
    horizon: int = 21,
    sample_every: int = 5,
    n_splits: int = 5,
    min_train_dates: int = 52,
    embargo_days: int = 5,
) -> tuple[dict, dict[str, pd.Series], tuple[pd.Timestamp, pd.Timestamp]]:
    """Score every default model under purged walk-forward CV and a leaky shuffled K-fold."""
    splitter = PurgedWalkForward(n_splits, min_train_dates, embargo_days)
    purged_splits = list(splitter.split(ds.dates, ds.label_end))
    leaky_splits = list(leaky_kfold(ds.dates, n_splits))
    overlap = max(horizon // sample_every, 1)
    periods_per_year = 252 / horizon

    results, ic_series = {}, {}
    for name, model in default_models().items():
        logger.info("Evaluating %s", name)
        purged = cross_validate(model, ds.X, ds.y, purged_splits, overlap)
        leaky = cross_validate(model, ds.X, ds.y, leaky_splits, overlap)
        ls = purged.long_short.iloc[::overlap]
        results[name] = {
            "purged_walk_forward": purged.summary,
            "long_short_ann_return": round(float(ls.mean() * periods_per_year), 4),
            "leaky_shuffled_kfold_ic_mean": leaky.summary["ic_mean"],
        }
        ic_series[name] = purged.ic

    test_dates = ds.dates[np.concatenate([t for _, t in purged_splits])]
    return results, ic_series, (test_dates.min(), test_dates.max())


def main() -> None:
    parser = argparse.ArgumentParser(description="ML alpha with purged time-series CV")
    parser.add_argument("--config", default="config/watchlist.yaml")
    parser.add_argument("--symbols", nargs="+", help="Override the watchlist universe")
    parser.add_argument("--exclude", nargs="*", default=["SPY"], help="Drop from universe")
    parser.add_argument("--horizon", type=int, default=21, help="Forward return days")
    parser.add_argument("--sample-every", type=int, default=5, help="Days between samples")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--min-train-dates", type=int, default=52)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--output-json", default="reports/ml_report.json")
    parser.add_argument("--output-chart", default="reports/ml_oos_ic.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_yaml(resolve_path(args.config))
    excluded = {s.upper() for s in args.exclude or []}
    symbols = [s for s in (args.symbols or cfg.get("symbols", [])) if s.upper() not in excluded]

    panel = build_panel(ParquetStore(resolve_path(cfg.get("data_dir", "data/parquet"))), symbols)
    ds = build_dataset(panel, horizon=args.horizon, sample_every=args.sample_every)
    dates = ds.dates.unique()
    logger.info(
        "Dataset: %d samples, %d dates (%s to %s), %d features",
        len(ds.X), len(dates), dates.min().date(), dates.max().date(), ds.X.shape[1],
    )

    results, ic_series, test_period = evaluate_models(
        ds, args.horizon, args.sample_every, args.splits, args.min_train_dates, args.embargo_days
    )

    payload = {
        "universe": panel.symbols,
        "samples": len(ds.X),
        "features": list(ds.X.columns),
        "settings": {
            "horizon_days": args.horizon,
            "sample_every_days": args.sample_every,
            "splits": args.splits,
            "min_train_dates": args.min_train_dates,
            "embargo_days": args.embargo_days,
        },
        "out_of_sample_period": [str(d.date()) for d in test_period],
        "models": results,
        "ridge_coefficients_full_sample": {
            k: round(float(v), 5) for k, v in ridge_coefficients(
                default_models()["ridge"], ds.X, ds.y
            ).items()
        },
    }
    out_json = resolve_path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_ic_chart(ic_series, resolve_path(args.output_chart))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
