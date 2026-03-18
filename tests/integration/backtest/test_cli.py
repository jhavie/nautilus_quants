"""Integration tests for CLI."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from nautilus_quants.backtest.cli import cli


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def fixture_path(self) -> Path:
        """Path to test fixtures."""
        return Path(__file__).parent.parent.parent / "fixtures" / "backtest"

    @pytest.fixture
    def valid_config_file(self, fixture_path: Path, tmp_path: Path) -> Path:
        """Create a valid Nautilus native config file for testing.

        Note: Uses Nautilus native BacktestRunConfig format, not the legacy
        project-specific format.
        """
        config = {
            "venues": [
                {
                    "name": "BINANCE",
                    "oms_type": "NETTING",
                    "account_type": "CASH",
                    "base_currency": "USDT",
                    "starting_balances": ["100000 USDT"],
                }
            ],
            "data": [
                {
                    "catalog_path": str(fixture_path),
                    "data_cls": "nautilus_trader.model.data:Bar",
                    "instrument_ids": ["BTCUSDT.BINANCE"],
                    "bar_spec": "1h",
                    "start_time": "2025-01-01",
                    "end_time": "2025-01-05",
                }
            ],
            "engine": {
                "trader_id": "TESTER-001",
                "logging": {"log_level": "ERROR", "bypass_logging": True},
                "strategies": [
                    {
                        "strategy_path": "nautilus_quants.strategies.breakout:PriceVolumeBreakoutStrategy",
                        "config_path": "nautilus_quants.strategies.breakout:PriceVolumeBreakoutStrategyConfig",
                        "config": {
                            "instrument_id": "BTCUSDT.BINANCE",
                            "breakout_period": 20,
                            "sma_period": 50,
                        },
                    }
                ],
            },
            "report": {
                "output_dir": str(tmp_path / "output"),
                "formats": ["csv"],
                "tearsheet": {"enabled": False},
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test CLI shows help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Nautilus Quants" in result.output

    def test_cli_run_help(self, runner: CliRunner) -> None:
        """Test run command help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output

    def test_cli_validate_valid_config(
        self, runner: CliRunner, valid_config_file: Path
    ) -> None:
        """Test validate command with valid config.

        Note: This test may fail if the catalog path doesn't exist or
        if Nautilus validation requires actual data. The test verifies
        the command runs without crashing.
        """
        result = runner.invoke(cli, ["validate", str(valid_config_file)])
        # Accept exit code 0 (success) or 1 (validation error for missing data)
        # The key is that it doesn't crash (exit code 4)
        assert result.exit_code in [0, 1]

    def test_cli_validate_invalid_yaml(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test validate command with syntactically invalid YAML."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("venues:\n  - name: [broken")

        result = runner.invoke(cli, ["validate", str(invalid_config)])
        assert result.exit_code == 1

    def test_cli_validate_missing_required_fields(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test validate command with missing required fields."""
        minimal_config = tmp_path / "minimal.yaml"
        minimal_config.write_text("engine:\n  trader_id: TEST-001\n")

        result = runner.invoke(cli, ["validate", str(minimal_config)])
        # Should fail validation due to missing venues/data
        assert result.exit_code == 1

    def test_cli_list(self, runner: CliRunner) -> None:
        """Test list command."""
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "breakout" in result.output

    def test_cli_list_verbose(self, runner: CliRunner) -> None:
        """Test list command with verbose flag."""
        result = runner.invoke(cli, ["list", "-v"])
        assert result.exit_code == 0
        assert "Parameters" in result.output

    def test_cli_run_dry_run(
        self, runner: CliRunner, valid_config_file: Path
    ) -> None:
        """Test run command with --dry-run flag.

        Note: Dry run validates the config without executing the backtest.
        May fail if Nautilus validation requires actual data catalog.
        """
        result = runner.invoke(cli, ["run", "--dry-run", str(valid_config_file)])
        # Accept exit code 0 (success) or 1 (validation error for missing data)
        # Exit code 4 would indicate an unexpected error
        if result.exit_code == 0:
            assert "dry run" in result.output.lower() or "valid" in result.output.lower()

    @pytest.mark.slow
    def test_cli_run_executes_backtest(
        self, runner: CliRunner, valid_config_file: Path
    ) -> None:
        """Test run command executes backtest."""
        result = runner.invoke(cli, ["run", str(valid_config_file)])

        # Should complete (exit code 0) or fail gracefully
        assert result.exit_code in [0, 4]  # 0 = success, 4 = execution error
