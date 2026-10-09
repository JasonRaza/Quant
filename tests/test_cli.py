"""End-to-end tests for the command-line tools, run on synthetic data."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from backtester import cli as backtest_cli
from data_pipeline import cli as fetch_cli
from data_pipeline import quality
from factors import cli as factors_cli
from ml_alpha import cli as ml_cli
from paper_trading import launch
from portfolio import cli as portfolio_cli


def run_main(monkeypatch, capsys, main, *args) -> dict:
    monkeypatch.setattr(sys, "argv", ["prog", *map(str, args)])
    main()
    return json.loads(capsys.readouterr().out)


def test_backtest_cli_dual_ma(synthetic_market, tmp_path, monkeypatch, capsys):
    out = run_main(
        monkeypatch, capsys, backtest_cli.main,
        "--data-dir", synthetic_market["data_dir"], "--strategy", "dual_ma", "--symbol", "SPY",
        "--output-json", tmp_path / "bt.json", "--output-chart", tmp_path / "bt.png",
    )
    assert out["strategy"] == "dual_ma"
    assert {"total_return", "sharpe", "max_drawdown"} <= set(out["metrics"])
    assert (tmp_path / "bt.json").exists() and (tmp_path / "bt.png").exists()


def test_backtest_cli_pair(synthetic_market, tmp_path, monkeypatch, capsys):
    out = run_main(
        monkeypatch, capsys, backtest_cli.main,
        "--data-dir", synthetic_market["data_dir"], "--strategy", "mean_reversion",
        "--pair", "KO", "PEP",
        "--output-json", tmp_path / "pair.json", "--output-chart", tmp_path / "pair.png",
    )
    assert out["symbol"] == ["KO", "PEP"]
    assert out["trade_count"] > 0


def test_backtest_cli_missing_symbol(synthetic_market, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--data-dir", str(synthetic_market["data_dir"]), "--symbol", "NOPE",
         "--output-json", str(tmp_path / "x.json")],
    )
    with pytest.raises(FileNotFoundError):
        backtest_cli.main()


def test_factors_cli(synthetic_market, tmp_path, monkeypatch, capsys):
    out = run_main(
        monkeypatch, capsys, factors_cli.main,
        "--config", synthetic_market["config"],
        "--output-json", tmp_path / "f.json", "--output-chart", tmp_path / "f.png",
    )
    assert "SPY" not in out["universe"]
    assert set(out["factors"]) == {"momentum_12_1", "reversal_1m", "low_vol", "liquidity"}
    assert out["screen"]["rankings"][0]["rank"] == 1


def test_portfolio_cli(synthetic_market, tmp_path, monkeypatch, capsys):
    out = run_main(
        monkeypatch, capsys, portfolio_cli.main,
        "--config", synthetic_market["config"],
        "--output-json", tmp_path / "p.json", "--output-chart", tmp_path / "p.png",
    )
    assert "benchmark_SPY" in out["walkforward"]
    for name, alloc in out["current_allocation"].items():
        assert sum(alloc["weights"].values()) == pytest.approx(1.0, abs=1e-3), name


def test_ml_cli(synthetic_market, tmp_path, monkeypatch, capsys):
    out = run_main(
        monkeypatch, capsys, ml_cli.main,
        "--config", synthetic_market["config"], "--splits", "3",
        "--output-json", tmp_path / "ml.json", "--output-chart", tmp_path / "ml.png",
    )
    assert set(out["models"]) == {"momentum_baseline", "ridge", "gradient_boosting"}
    assert out["samples"] > 0 and (tmp_path / "ml.png").exists()


def test_quality_cli(synthetic_market, monkeypatch, capsys):
    out = run_main(monkeypatch, capsys, quality.main, "--config", synthetic_market["config"])
    assert out["summary"]["symbol_count"] == 10


def test_fetch_cli_offline(synthetic_market, tmp_path, monkeypatch, capsys):
    """The fetch CLI wires config -> pipeline -> storage; the network call is stubbed."""
    config = tmp_path / "watch.yaml"
    config.write_text(
        "symbols: [AAA]\nstart_date: '2024-01-01'\nend_date: '2024-01-10'\n"
        f"data_dir: '{(tmp_path / 'pq').as_posix()}'\n"
        f"metadata_db: '{(tmp_path / 'm.db').as_posix()}'\n",
        encoding="utf-8",
    )
    bars = pd.DataFrame(
        {"symbol": "AAA", "date": pd.bdate_range("2024-01-02", periods=5).date, "open": 1.0,
         "high": 1.0, "low": 1.0, "close": [10.0, 11, 12, 13, 14], "volume": 100}
    )
    monkeypatch.setattr("data_pipeline.pipeline.fetch_symbol_bars", lambda *a, **k: bars)
    out = run_main(monkeypatch, capsys, fetch_cli.main, "--config", config)
    assert out["symbols"]["AAA"]["total_rows"] == 5
    assert not out["errors"]


def test_dashboard_launcher_runs_streamlit(monkeypatch):
    calls = []
    monkeypatch.setattr(launch.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", ["quant-dashboard", "--server.port", "9999"])
    with pytest.raises(SystemExit) as exit_info:
        launch.main()
    assert exit_info.value.code == 0
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4].endswith("app.py") and cmd[-2:] == ["--server.port", "9999"]
