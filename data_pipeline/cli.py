"""CLI for market data pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from data_pipeline.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and store OHLCV equity bars")
    parser.add_argument(
        "--config",
        default="config/watchlist.yaml",
        help="Path to watchlist YAML config",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_pipeline(args.config)
    print(json.dumps(results, indent=2))
    if results.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
