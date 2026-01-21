"""
Integration tests for full pipeline workflow.

Tests complete pipeline from download through transform with all stages.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from click.testing import CliRunner

import pandas as pd

from nautilus_quants.data.cli import cli, EXIT_SUCCESS, EXIT_VALIDATION_ERROR
from nautilus_quants.data.config import load_config
from nautilus_quants.data.download.binance import BinanceDownloader
from nautilus_quants.data.validate.integrity import validate_file
from nautilus_quants.data.process.processors import process_data, ProcessConfig
from nautilus_quants.data.transform.parquet import transform_to_parquet


class TestPipelineIntegration:
    """Integration tests for full pipeline workflow."""

    @pytest.fixture
    def mock_klines(self):
        """Generate mock K-line data from Binance API."""
        base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
        interval_ms = 3600000  # 1 hour

        klines = []
        for i in range(24):  # 24 hours of data
            ts = base_ts + i * interval_ms
            klines.append([
                ts,                        # Open time
                f"{42000.0 + i * 10}",     # Open
                f"{42500.0 + i * 10}",     # High
                f"{41900.0 + i * 10}",     # Low
                f"{42100.0 + i * 10}",     # Close
                "100.0",                   # Volume
                ts + interval_ms - 1,      # Close time
                "4200000.0",               # Quote volume
                1000,                      # Number of trades
                "50.0",                    # Taker buy base volume
                "2100000.0",               # Taker buy quote volume
                "0",                       # Ignore
            ])
        return klines

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration file."""
        config_path = tmp_path / "config.yaml"
        config_content = f"""
download:
  symbols:
    - BTCUSDT
  timeframes:
    - 1h
  start_date: "2024-01-01"
  end_date: "2024-01-01"
  market_type: futures
  checkpoint:
    enabled: true
    batch_size: 1000

validate:
  checks:
    - duplicates
    - gaps
    - ohlc_relationships
  max_gap_bars: 3

process:
  remove_duplicates: true
  keep_duplicate: last
  fill_small_gaps: true
  max_gap_bars: 3
  remove_invalid_ohlc: true

transform:
  merge_files: true
  overwrite: false

paths:
  raw_data: "{tmp_path}/data/raw"
  processed_data: "{tmp_path}/data/processed"
  catalog: "{tmp_path}/data/catalog"
  logs: "{tmp_path}/logs"
"""
        config_path.write_text(config_content)
        return config_path

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_full_pipeline_programmatic(self, tmp_path, mock_klines):
        """Test complete pipeline programmatically."""
        # Setup directories
        raw_dir = tmp_path / "raw" / "binance" / "BTCUSDT" / "1h"
        proc_dir = tmp_path / "processed" / "binance" / "BTCUSDT" / "1h"
        catalog_dir = tmp_path / "catalog"

        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)

        # Step 1: Simulate download by creating raw CSV
        raw_csv = raw_dir / "BTCUSDT_1h_2024-01-01.csv"
        raw_data = []
        for kline in mock_klines:
            raw_data.append({
                "timestamp": kline[0],
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "quote_volume": float(kline[7]),
                "trades_count": kline[8],
            })
        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(raw_csv, index=False)

        # Step 2: Validate
        validation_report = validate_file(raw_csv)
        assert validation_report.passed is True
        assert validation_report.total_rows == 24

        # Step 3: Process
        processed_csv = proc_dir / "BTCUSDT_1h_processed.csv"
        processing_report = process_data(raw_csv, processed_csv)
        assert processing_report.output_rows == 24

        # Step 4: Transform
        transform_result = transform_to_parquet(
            processed_csv, catalog_dir, "BTCUSDT", "1h"
        )
        assert transform_result.success is True
        assert transform_result.rows_transformed == 24

        # Verify final output
        assert catalog_dir.exists()

    def test_pipeline_handles_data_with_issues(self, tmp_path):
        """Pipeline should handle and clean data with issues."""
        # Setup directories
        raw_dir = tmp_path / "raw" / "binance" / "BTCUSDT" / "1h"
        proc_dir = tmp_path / "processed" / "binance" / "BTCUSDT" / "1h"
        catalog_dir = tmp_path / "catalog"

        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)

        # Create raw CSV with issues
        raw_csv = raw_dir / "BTCUSDT_1h_with_issues.csv"
        df = pd.DataFrame({
            "timestamp": [
                1704067200000,
                1704067200000,  # Duplicate
                1704070800000,
                # Gap: missing 1704074400000
                1704078000000,
                1704081600000,
            ],
            "open": [42000.0, 42050.0, 42100.0, 42200.0, 42300.0],
            "high": [42500.0, 42550.0, 42600.0, 42700.0, 42800.0],
            "low": [41900.0, 41950.0, 42000.0, 42100.0, 42200.0],
            "close": [42100.0, 42150.0, 42200.0, 42300.0, 42400.0],
            "volume": [100.0, 105.0, 150.0, 120.0, 130.0],
            "quote_volume": [4200000.0, 4410000.0, 6300000.0, 5040000.0, 5460000.0],
            "trades_count": [1000, 1050, 1200, 1100, 1150],
        })
        df.to_csv(raw_csv, index=False)

        # Validate (should detect issues)
        validation_report = validate_file(raw_csv)
        assert validation_report.duplicate_count >= 1
        assert validation_report.gap_count >= 1

        # Process (should fix issues)
        processed_csv = proc_dir / "BTCUSDT_1h_processed.csv"
        processing_report = process_data(raw_csv, processed_csv)

        assert processing_report.duplicates_removed >= 1
        assert processing_report.gaps_filled >= 1

        # Verify processed data is clean
        df_processed = pd.read_csv(processed_csv)
        assert df_processed["timestamp"].is_unique  # No duplicates

        # Transform should work with clean data
        transform_result = transform_to_parquet(
            processed_csv, catalog_dir, "BTCUSDT", "1h"
        )
        assert transform_result.success is True

    def test_pipeline_status_command(self, test_config, runner, tmp_path):
        """Status command should show pipeline status."""
        # Create some data for status
        raw_dir = tmp_path / "data" / "raw" / "binance" / "BTCUSDT" / "1h"
        raw_dir.mkdir(parents=True)

        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
            "quote_volume": [4200000.0],
            "trades_count": [1000],
        })
        df.to_csv(raw_dir / "BTCUSDT_1h_data.csv", index=False)

        result = runner.invoke(cli, ["--config", str(test_config), "status"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output

    def test_pipeline_clean_command_dry_run(self, test_config, runner):
        """Clean command with --all should require confirmation."""
        result = runner.invoke(
            cli,
            ["--config", str(test_config), "clean", "--all"],
            input="n\n",  # Don't confirm
        )

        assert "Aborted" in result.output

    def test_pipeline_download_dry_run(self, test_config, runner):
        """Download with --dry-run should not actually download."""
        result = runner.invoke(
            cli,
            ["--config", str(test_config), "--dry-run", "download"],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "BTCUSDT" in result.output

    def test_pipeline_validate_dry_run(self, test_config, runner):
        """Validate with --dry-run should not actually validate."""
        result = runner.invoke(
            cli,
            ["--config", str(test_config), "--dry-run", "validate"],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_pipeline_run_dry_run(self, test_config, runner):
        """Run with --dry-run should show planned steps."""
        result = runner.invoke(
            cli,
            ["--config", str(test_config), "--dry-run", "run"],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Download" in result.output
        assert "Validate" in result.output
        assert "Process" in result.output
        assert "Transform" in result.output

    def test_pipeline_skip_flags(self, test_config, runner, tmp_path):
        """Run command should respect skip flags."""
        # Create processed data to skip download/validate/process
        proc_dir = tmp_path / "data" / "processed" / "binance" / "BTCUSDT" / "1h"
        proc_dir.mkdir(parents=True)

        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
            "quote_volume": [4200000.0],
            "trades_count": [1000],
        })
        df.to_csv(proc_dir / "BTCUSDT_1h_processed.csv", index=False)

        result = runner.invoke(
            cli,
            [
                "--config", str(test_config),
                "run",
                "--skip-download",
                "--skip-validate",
                "--skip-process",
            ],
        )

        # Should only run transform step
        assert "[4/4] TRANSFORM" in result.output
        # Should NOT run skipped steps
        assert "[1/4] DOWNLOAD" not in result.output

    def test_pipeline_multiple_symbols(self, tmp_path):
        """Pipeline should handle multiple symbols."""
        catalog_dir = tmp_path / "catalog"

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            # Create raw data
            raw_dir = tmp_path / "raw" / symbol / "1h"
            raw_dir.mkdir(parents=True)
            raw_csv = raw_dir / f"{symbol}_1h_data.csv"

            df = pd.DataFrame({
                "timestamp": [1704067200000, 1704070800000],
                "open": [42000.0, 42100.0],
                "high": [42500.0, 42600.0],
                "low": [41900.0, 42000.0],
                "close": [42100.0, 42200.0],
                "volume": [100.0, 150.0],
                "quote_volume": [4200000.0, 6300000.0],
                "trades_count": [1000, 1200],
            })
            df.to_csv(raw_csv, index=False)

            # Validate
            report = validate_file(raw_csv)
            assert report.passed is True

            # Process
            proc_dir = tmp_path / "processed" / symbol / "1h"
            proc_dir.mkdir(parents=True)
            proc_csv = proc_dir / f"{symbol}_1h_processed.csv"
            process_data(raw_csv, proc_csv)

            # Transform
            result = transform_to_parquet(proc_csv, catalog_dir, symbol, "1h")
            assert result.success is True

        # Verify catalog contains both symbols
        assert catalog_dir.exists()

    def test_pipeline_report_generation(self, tmp_path, test_config, runner):
        """Pipeline run should generate reports."""
        # Create processed data
        proc_dir = tmp_path / "data" / "processed" / "binance" / "BTCUSDT" / "1h"
        proc_dir.mkdir(parents=True)

        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
            "quote_volume": [4200000.0],
            "trades_count": [1000],
        })
        df.to_csv(proc_dir / "BTCUSDT_1h_processed.csv", index=False)

        result = runner.invoke(
            cli,
            [
                "--config", str(test_config),
                "run",
                "--skip-download",
                "--skip-validate",
                "--skip-process",
            ],
        )

        # Should reference reports location
        assert "Reports:" in result.output or "Log directory:" in result.output

    def test_pipeline_exit_codes(self, tmp_path, test_config, runner):
        """Pipeline should use correct exit codes."""
        # Successful run
        proc_dir = tmp_path / "data" / "processed" / "binance" / "BTCUSDT" / "1h"
        proc_dir.mkdir(parents=True)

        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
            "quote_volume": [4200000.0],
            "trades_count": [1000],
        })
        df.to_csv(proc_dir / "BTCUSDT_1h_processed.csv", index=False)

        result = runner.invoke(
            cli,
            [
                "--config", str(test_config),
                "run",
                "--skip-download",
                "--skip-validate",
                "--skip-process",
            ],
        )

        # Transform should succeed
        assert result.exit_code == EXIT_SUCCESS
