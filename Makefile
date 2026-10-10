.PHONY: install test coverage check snapshot fetch backtest factors optimize ml dashboard api seed quality lint

install:
	pip install -e ".[all]"

test:
	pytest -v

coverage:
	pytest -q --cov --cov-report=term --cov-report=html

check: lint test

seed:
	python scripts/seed_sample_data.py

snapshot:
	cp data/parquet/*.parquet data/snapshot/

fetch:
	quant-fetch --config config/watchlist.yaml

backtest:
	quant-backtest --strategy dual_ma --symbol SPY

factors:
	quant-factors --config config/watchlist.yaml --exclude SPY

optimize:
	quant-optimize --config config/watchlist.yaml --exclude SPY

ml:
	quant-ml --config config/watchlist.yaml --exclude SPY

dashboard:
	quant-dashboard

api:
	quant-paper-api

quality:
	quant-quality --config config/watchlist.yaml

lint:
	ruff check .
