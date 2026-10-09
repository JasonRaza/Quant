.PHONY: install test fetch backtest factors seed quality lint

install:
	pip install -e ".[backtest,dev]"

test:
	pytest -v

seed:
	python scripts/seed_sample_data.py

fetch:
	quant-fetch --config config/watchlist.yaml

backtest:
	quant-backtest --strategy dual_ma --symbol SPY

factors:
	quant-factors --config config/watchlist.yaml --exclude SPY KO PEP

quality:
	quant-quality --config config/watchlist.yaml

lint:
	ruff check .
