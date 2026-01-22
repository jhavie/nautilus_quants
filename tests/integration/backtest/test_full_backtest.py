"""Integration tests for full backtest execution."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from nautilus_quants.backtest.config import BacktestConfig, BacktestResult
from nautilus_quants.backtest.runner import BacktestRunner


class TestFullBacktest:
    """Integration tests for complete backtest execution."""

    @pytest.fixture
    def valid_config_dict(self, backtest_catalog_path: Path, tmp_path: Path) -> dict[str, Any]:
        """Create valid configuration dictionary."""
        return {
            "strategy": {
                "type": "breakout",
                "instrument_id": "BTCUSDT",
                "params": {
                    "breakout_period": 20,
                    "sma_period": 50,
                    "position_size_pct": 0.10,
                    "max_positions": 1,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                },
            },
            "backtest": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-15",
                "bar_spec": "1h",
                "catalog_path": str(backtest_catalog_path),
                "warmup_days": 5,
            },
            "venue": {
                "name": "BINANCE",
                "oms_type": "NETTING",
                "account_type": "MARGIN",
                "starting_balance": "100000 USDT",
                "default_leverage": 5,
                "fee_model": {
                    "type": "maker_taker",
                    "maker_fee": 0.0002,
                    "taker_fee": 0.0004,
                },
            },
            "report": {
                "output_dir": str(tmp_path / "backtest_output"),
                "formats": ["csv"],
                "tearsheet": {
                    "enabled": False,  # Disable for faster tests
                },
            },
            "logging": {
                "level": "WARNING",
                "log_to_file": False,
                "bypass_logging": True,
            },
        }

    @pytest.mark.slow
    def test_full_backtest_execution(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test complete backtest execution from config to results."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        assert isinstance(result, BacktestResult)
        assert result.run_id != ""
        assert result.output_dir.exists()
        assert result.duration_seconds > 0

    @pytest.mark.slow
    def test_backtest_generates_csv_reports(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test backtest generates expected CSV reports."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        # Check CSV files are created
        assert (result.output_dir / "orders_report.csv").exists()
        assert (result.output_dir / "fills_report.csv").exists()
        assert (result.output_dir / "positions_report.csv").exists()
        assert (result.output_dir / "account_report.csv").exists()

    @pytest.mark.slow
    def test_backtest_returns_statistics(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test backtest returns performance statistics."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        # Statistics should be populated (may be empty if no trades)
        assert isinstance(result.statistics, dict)

    @pytest.mark.slow
    def test_backtest_with_yaml_file(
        self, valid_config_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test loading config from YAML file and running backtest."""
        # Write config to file
        config_file = tmp_path / "backtest.yaml"
        with open(config_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        # Load and run
        config = BacktestConfig.from_yaml(config_file)
        runner = BacktestRunner(config)

        result = runner.run()

        assert result.run_id != ""

    def test_backtest_fails_gracefully_with_missing_data(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test backtest fails gracefully when data is missing."""
        valid_config_dict["backtest"]["catalog_path"] = "/nonexistent/path"
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        assert result.success is False
        assert len(result.errors) > 0

    def test_backtest_fails_gracefully_with_unknown_strategy(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test backtest fails gracefully with unknown strategy."""
        valid_config_dict["strategy"]["type"] = "nonexistent"
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        assert result.success is False
        assert any("strategy" in e.lower() for e in result.errors)


class TestZeroTradesScenario:
    """Tests for backtest with no trades executed."""

    @pytest.fixture
    def no_trades_config(self, backtest_catalog_path: Path, tmp_path: Path) -> dict[str, Any]:
        """Config that results in no trades (very short period)."""
        return {
            "strategy": {
                "type": "breakout",
                "instrument_id": "BTCUSDT",
                "params": {
                    "breakout_period": 100,  # Very long period
                    "sma_period": 200,
                    "position_size_pct": 0.10,
                },
            },
            "backtest": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",  # Very short period
                "bar_spec": "1h",
                "catalog_path": str(backtest_catalog_path),
            },
            "venue": {
                "name": "BINANCE",
                "starting_balance": "100000 USDT",
            },
            "report": {
                "output_dir": str(tmp_path / "output"),
                "formats": ["csv"],
                "tearsheet": {"enabled": False},
            },
            "logging": {
                "level": "ERROR",
                "bypass_logging": True,
            },
        }

    @pytest.mark.slow
    def test_backtest_handles_zero_trades(
        self, no_trades_config: dict[str, Any]
    ) -> None:
        """Test backtest completes successfully with zero trades."""
        config = BacktestConfig.from_dict(no_trades_config)
        runner = BacktestRunner(config)

        result = runner.run()

        # Should still succeed even with no trades
        assert result.output_dir.exists()
        # Reports should still be created
        assert (result.output_dir / "orders_report.csv").exists()
