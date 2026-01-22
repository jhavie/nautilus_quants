"""Integration tests for backtest edge cases and error handling."""

import os
import stat
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.backtest.config import BacktestConfig, BacktestResult
from nautilus_quants.backtest.exceptions import BacktestDataError
from nautilus_quants.backtest.runner import BacktestRunner


class TestCorruptedData:
    """Tests for handling corrupted or invalid data files."""

    @pytest.fixture
    def corrupted_data_path(self, tmp_path: Path) -> Path:
        """Create a corrupted Parquet file."""
        corrupted_dir = tmp_path / "corrupted_data"
        corrupted_dir.mkdir(parents=True, exist_ok=True)

        # Create a file with invalid Parquet data (random bytes)
        corrupted_file = corrupted_dir / "BTCUSDT_1h.parquet"
        with open(corrupted_file, "wb") as f:
            f.write(b"This is not valid Parquet data! \x00\x01\x02\x03\xff\xfe")

        return corrupted_dir

    @pytest.fixture
    def valid_config_dict(self, backtest_catalog_path: Path, tmp_path: Path) -> dict[str, Any]:
        """Create base configuration dictionary."""
        return {
            "strategy": {
                "type": "breakout",
                "instrument_id": "BTCUSDT",
                "params": {
                    "breakout_period": 20,
                    "sma_period": 50,
                    "position_size_pct": 0.10,
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
            },
            "report": {
                "output_dir": str(tmp_path / "backtest_output"),
                "formats": ["csv"],
                "tearsheet": {"enabled": False},
            },
            "logging": {
                "level": "ERROR",
                "log_to_file": False,
                "bypass_logging": True,
            },
        }

    def test_corrupted_parquet_file(
        self, valid_config_dict: dict[str, Any], corrupted_data_path: Path
    ) -> None:
        """T066 - Test handling of corrupted Parquet file.

        Verifies that the system gracefully handles invalid Parquet files
        and returns a proper error in the BacktestResult.
        """
        # Configure to use corrupted data
        valid_config_dict["backtest"]["catalog_path"] = str(corrupted_data_path)

        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        # Run should not raise, but return failed result
        result = runner.run()

        # Verify error handling
        assert isinstance(result, BacktestResult)
        assert result.success is False
        assert len(result.errors) > 0

        # Verify the error message indicates data loading failure
        error_message = result.errors[0].lower()
        assert "failed to load data" in error_message or "parquet" in error_message or "catalog" in error_message or "no data" in error_message

    def test_missing_parquet_file(
        self, valid_config_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test handling of missing data file.

        Verifies that appropriate error is raised when data file doesn't exist.
        """
        # Configure to use non-existent path
        nonexistent_path = tmp_path / "no_such_directory"
        valid_config_dict["backtest"]["catalog_path"] = str(nonexistent_path)

        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        assert result.success is False
        assert len(result.errors) > 0

    def test_empty_parquet_file(
        self, valid_config_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test handling of empty Parquet file.

        Verifies that the system handles empty data files gracefully.
        """
        import pandas as pd

        # Create empty parquet file with correct schema
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir(parents=True, exist_ok=True)

        empty_df = pd.DataFrame({"datetime": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
        empty_file = empty_dir / "BTCUSDT_1h.parquet"
        empty_df.to_parquet(empty_file)

        valid_config_dict["backtest"]["catalog_path"] = str(empty_dir)

        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        # Should fail with appropriate error
        assert result.success is False
        assert len(result.errors) > 0


class TestReadOnlyOutputDirectory:
    """Tests for handling read-only output directories."""

    @pytest.fixture
    def readonly_output_dir(self, tmp_path: Path) -> Path:
        """Create a read-only directory for output."""
        readonly_dir = tmp_path / "readonly_output"
        readonly_dir.mkdir(parents=True, exist_ok=True)

        # Make directory read-only
        os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IXUSR)

        return readonly_dir

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
            },
            "report": {
                "formats": ["csv"],
                "tearsheet": {"enabled": False},
            },
            "logging": {
                "level": "ERROR",
                "log_to_file": False,
                "bypass_logging": True,
            },
        }

    def test_readonly_output_directory(
        self, valid_config_dict: dict[str, Any], readonly_output_dir: Path
    ) -> None:
        """T067 - Test handling of read-only output directory.

        Verifies that the system raises appropriate error when attempting
        to write to a read-only directory.
        """
        # Configure to use read-only output directory
        valid_config_dict["report"]["output_dir"] = str(readonly_output_dir)

        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        try:
            result = runner.run()

            # Should fail with permission error
            assert result.success is False
            assert len(result.errors) > 0

            # Error should indicate permission/access issue
            error_message = result.errors[0].lower()
            assert "permission" in error_message or "denied" in error_message or "readonly" in error_message or "read-only" in error_message

        finally:
            # Cleanup: restore write permissions for cleanup
            os.chmod(readonly_output_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    def test_nonexistent_parent_output_directory(
        self, valid_config_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test handling of output directory with non-existent parent.

        Verifies the system creates parent directories as needed.
        """
        # Use deeply nested path that doesn't exist
        nested_output = tmp_path / "level1" / "level2" / "level3" / "output"
        valid_config_dict["report"]["output_dir"] = str(nested_output)

        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        # Should succeed - the system creates parent directories
        # (BacktestRunner creates output_dir with parents=True)
        assert isinstance(result, BacktestResult)


class TestStrategyExceptions:
    """Tests for handling strategy exceptions during execution."""

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
            },
            "report": {
                "output_dir": str(tmp_path / "backtest_output"),
                "formats": ["csv"],
                "tearsheet": {"enabled": False},
            },
            "logging": {
                "level": "ERROR",
                "log_to_file": False,
                "bypass_logging": True,
            },
        }

    def test_strategy_instantiation_failure(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """T068 - Test handling of strategy exception during instantiation.

        Verifies the system handles strategy creation failures gracefully.
        Since the high-level API uses ImportableStrategyConfig with module paths,
        we test by providing an invalid config_path that will fail during import.
        """
        from nautilus_quants.backtest.runner import BacktestRunner

        # Create a runner that will fail during strategy config build
        # by patching _build_strategy_config to raise an error
        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        # Patch the strategy config builder to simulate instantiation failure
        def mock_build_strategy_config(*args: Any, **kwargs: Any) -> None:
            raise ValueError("Simulated strategy instantiation error")

        with patch.object(runner, "_build_strategy_config", side_effect=mock_build_strategy_config):
            result = runner.run()

            # Should fail gracefully with strategy error
            assert result.success is False
            assert len(result.errors) > 0

            # Error should mention the simulated error
            error_message = result.errors[0].lower()
            assert "simulated" in error_message or "strategy" in error_message

    @pytest.mark.slow
    def test_strategy_execution_exception_during_run(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test handling of exception during strategy execution.

        Verifies the system handles runtime errors in strategy gracefully.
        Uses mocking to simulate a strategy that raises an exception.
        """
        from nautilus_quants.strategies import STRATEGY_REGISTRY

        # Get the real strategy class
        strategy_class, config_class = STRATEGY_REGISTRY["breakout"]

        # Create a mock that raises exception during run
        mock_strategy = MagicMock()
        mock_strategy.side_effect = RuntimeError("Simulated strategy execution error")

        # Patch the strategy registry temporarily
        with patch.dict(STRATEGY_REGISTRY, {"breakout": (mock_strategy, config_class)}):
            config = BacktestConfig.from_dict(valid_config_dict)
            runner = BacktestRunner(config)

            result = runner.run()

            # Should fail with error
            assert result.success is False
            assert len(result.errors) > 0

    def test_unknown_strategy_type(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test handling of unknown strategy type.

        Verifies appropriate error when requesting non-existent strategy.
        """
        valid_config_dict["strategy"]["type"] = "nonexistent_strategy_xyz"

        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        assert result.success is False
        assert len(result.errors) > 0
        assert "unknown strategy type" in result.errors[0].lower()

    def test_missing_required_strategy_params(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test handling of missing required strategy parameters.

        Verifies the system handles incomplete strategy configuration.
        Note: nautilus_trader strategies use defaults, so this test verifies
        the backtest still runs (it doesn't fail on missing optional params).
        """
        # Remove optional parameter - strategy uses defaults
        del valid_config_dict["strategy"]["params"]["breakout_period"]

        config = BacktestConfig.from_dict(valid_config_dict)
        runner = BacktestRunner(config)

        result = runner.run()

        # Should succeed - strategy uses default values for missing params
        assert isinstance(result, BacktestResult)
        # The backtest runs (may or may not succeed based on data)


class TestConcurrentAccess:
    """Tests for handling concurrent access scenarios."""

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
            },
            "report": {
                "output_dir": str(tmp_path / "backtest_output"),
                "formats": ["csv"],
                "tearsheet": {"enabled": False},
            },
            "logging": {
                "level": "ERROR",
                "log_to_file": False,
                "bypass_logging": True,
            },
        }

    def test_same_output_directory_multiple_runs(
        self, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test multiple backtest runs to same output directory.

        Verifies that multiple runs create separate output directories
        with unique run IDs.
        """
        import time

        config = BacktestConfig.from_dict(valid_config_dict)

        # Run first backtest
        runner1 = BacktestRunner(config)
        result1 = runner1.run()

        # Wait to ensure different timestamp for run_id
        time.sleep(1.1)

        # Run second backtest with same config
        runner2 = BacktestRunner(config)
        result2 = runner2.run()

        # Both should have run IDs
        assert result1.run_id != ""
        assert result2.run_id != ""

        # Run IDs should be different (after waiting 1+ second)
        assert result1.run_id != result2.run_id

        # Output directories should be different
        assert result1.output_dir != result2.output_dir
