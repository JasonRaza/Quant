.PHONY: install test coverage check typecheck snapshot fetch backtest factors optimize ml dashboard api seed quality lint docker-build up down smoke

install:
	pip install -e ".[all]"

test:
	pytest -v

coverage:
	pytest -q --cov --cov-report=term --cov-report=html

check: lint typecheck test

typecheck:
	mypy

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

# ---- containers ----
docker-build:
	docker build -t quant:local .

up:
	docker compose up --build -d

down:
	docker compose down

# Smoke tests against the running stack (make up first)
smoke:
	QUANT_SMOKE_URL=http://localhost:8000 QUANT_SMOKE_DASHBOARD_URL=http://localhost:8501 \
		pytest tests/smoke --noconftest -p no:cacheprovider -v
