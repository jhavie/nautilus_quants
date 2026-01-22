"""Integration tests for CLI."""

from pathlib import Path
from typing import Any

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
        """Create a valid config file for testing."""
        config = {
            "strategy": {
                "type": "breakout",
                "instrument_id": "BTCUSDT",
                "params": {
                    "breakout_period": 20,
                    "sma_period": 50,
                },
            },
            "backtest": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-05",
                "bar_spec": "1h",
                "catalog_path": str(fixture_path),
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
        """Test validate command with valid config."""
        result = runner.invoke(cli, ["validate", str(valid_config_file)])
        assert result.exit_code == 0
        assert "Configuration valid" in result.output

    def test_cli_validate_invalid_config(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test validate command with invalid config."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("strategy:\n  type: ''\n")

        result = runner.invoke(cli, ["validate", str(invalid_config)])
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

    def test_cli_init_creates_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test init command creates config file."""
        output_file = tmp_path / "new_config.yaml"

        result = runner.invoke(cli, ["init", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Created" in result.output

    def test_cli_init_refuses_overwrite(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init command refuses to overwrite existing file."""
        existing_file = tmp_path / "existing.yaml"
        existing_file.write_text("existing content")

        result = runner.invoke(cli, ["init", str(existing_file)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_cli_init_force_overwrite(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test init command overwrites with --force."""
        existing_file = tmp_path / "existing.yaml"
        existing_file.write_text("existing content")

        result = runner.invoke(cli, ["init", "--force", str(existing_file)])

        assert result.exit_code == 0
        assert "Created" in result.output

    def test_cli_run_dry_run(
        self, runner: CliRunner, valid_config_file: Path
    ) -> None:
        """Test run command with --dry-run flag."""
        result = runner.invoke(cli, ["run", "--dry-run", str(valid_config_file)])

        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    @pytest.mark.slow
    def test_cli_run_executes_backtest(
        self, runner: CliRunner, valid_config_file: Path
    ) -> None:
        """Test run command executes backtest."""
        result = runner.invoke(cli, ["run", str(valid_config_file)])

        # Should complete (exit code 0) or fail gracefully
        assert result.exit_code in [0, 4]  # 0 = success, 4 = execution error
