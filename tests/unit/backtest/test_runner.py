"""Unit tests for BacktestRunner."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.backtest.config import BacktestConfig
from nautilus_quants.backtest.runner import BacktestRunner


class TestBacktestRunner:
    """Tests for BacktestRunner class."""

    @pytest.fixture
    def valid_config_dict(self, tmp_path: Path) -> dict[str, Any]:
        """Create valid configuration dictionary with test fixture path."""
        fixture_path = Path(__file__).parent.parent.parent / "fixtures" / "backtest"
        return {
            "strategy": {
                "type": "breakout",
                "instrument_id": "BTCUSDT",
                "params": {
                    "breakout_period": 60,
                    "sma_period": 200,
                },
            },
            "backtest": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-10",
                "bar_spec": "1h",
                "catalog_path": str(fixture_path),
            },
            "venue": {
                "name": "BINANCE",
                "starting_balance": "100000 USDT",
            },
            "report": {
                "output_dir": str(tmp_path / "logs"),
            },
        }

    def test_init_stores_config(self, valid_config_dict: dict[str, Any]) -> None:
        """Test initialization stores config."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        assert runner.config == config
        assert runner.run_id == ""
        assert runner._engine is None
        assert runner._node is None

    def test_run_id_generated_on_run(self, valid_config_dict: dict[str, Any]) -> None:
        """Test run_id is generated when run is called."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        # Mock BacktestNode to avoid actual execution
        with patch("nautilus_quants.backtest.runner.BacktestNode") as mock_node_class:
            mock_node = MagicMock()
            mock_engine = MagicMock()
            mock_engine.cache.orders.return_value = []
            mock_engine.cache.positions.return_value = []
            mock_engine.cache.accounts.return_value = []
            mock_engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_general.return_value = {}
            mock_node.get_engines.return_value = [mock_engine]
            mock_node_class.return_value = mock_node

            result = runner.run()

        assert runner.run_id != ""
        assert len(runner.run_id) == 15  # YYYYMMDD_HHMMSS format

    def test_output_dir_created_on_run(
        self, valid_config_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test output directory is created when run is called."""
        valid_config_dict["report"]["output_dir"] = str(tmp_path / "backtest_output")
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        with patch("nautilus_quants.backtest.runner.BacktestNode") as mock_node_class:
            mock_node = MagicMock()
            mock_engine = MagicMock()
            mock_engine.cache.orders.return_value = []
            mock_engine.cache.positions.return_value = []
            mock_engine.cache.accounts.return_value = []
            mock_engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_general.return_value = {}
            mock_node.get_engines.return_value = [mock_engine]
            mock_node_class.return_value = mock_node

            runner.run()

        assert runner.output_dir.exists()
        assert runner.output_dir.parent == tmp_path / "backtest_output"

    def test_run_returns_result_on_success(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test run returns BacktestResult on success."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        with patch("nautilus_quants.backtest.runner.BacktestNode") as mock_node_class:
            mock_node = MagicMock()
            mock_engine = MagicMock()
            mock_engine.cache.orders.return_value = []
            mock_engine.cache.positions.return_value = []
            mock_engine.cache.accounts.return_value = []
            mock_engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_general.return_value = {}
            mock_node.get_engines.return_value = [mock_engine]
            mock_node_class.return_value = mock_node

            result = runner.run()

        assert result.success is True
        assert result.run_id == runner.run_id
        assert result.errors == []

    def test_run_returns_failure_on_strategy_error(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test run returns failure result on strategy error."""
        valid_config_dict["strategy"]["type"] = "nonexistent_strategy"
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        assert result.success is False
        assert len(result.errors) > 0
        assert "Unknown strategy type" in result.errors[0]

    def test_run_returns_failure_on_node_error(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test run returns failure result on BacktestNode error."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        with patch("nautilus_quants.backtest.runner.BacktestNode") as mock_node_class:
            mock_node_class.side_effect = RuntimeError("Node initialization failed")

            result = runner.run()

        assert result.success is False
        assert len(result.errors) > 0

    def test_engine_disposed_after_run(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test engine is disposed after run completes."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        with patch("nautilus_quants.backtest.runner.BacktestNode") as mock_node_class:
            mock_node = MagicMock()
            mock_engine = MagicMock()
            mock_engine.cache.orders.return_value = []
            mock_engine.cache.positions.return_value = []
            mock_engine.cache.accounts.return_value = []
            mock_engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
            mock_engine.portfolio.analyzer.get_performance_stats_general.return_value = {}
            mock_node.get_engines.return_value = [mock_engine]
            mock_node_class.return_value = mock_node

            runner.run()

        mock_engine.dispose.assert_called_once()

    def test_build_run_config_creates_valid_config(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test _build_run_config creates a valid BacktestRunConfig."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        # Initialize output_dir for logging config
        runner.output_dir = Path("/tmp/test")

        run_config = runner._build_run_config()

        assert run_config is not None
        assert len(run_config.venues) == 1
        assert len(run_config.data) >= 1
        assert run_config.engine is not None

    def test_build_strategy_config_with_valid_strategy(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test _build_strategy_config creates valid ImportableStrategyConfig."""
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        strategy_config = runner._build_strategy_config()

        assert strategy_config is not None
        assert "breakout" in strategy_config.strategy_path.lower()
        assert strategy_config.config is not None
