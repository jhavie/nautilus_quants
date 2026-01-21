"""
End-to-end CLI integration tests.

Tests the actual CLI commands with real Binance API calls.
These tests require network access and may be slow.
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIEndToEnd:
    """End-to-end tests for the data pipeline CLI."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        """Create temporary data directory."""
        data = tmp_path / "data"
        data.mkdir(parents=True)
        return data

    @pytest.fixture
    def config_file(self, tmp_path: Path, data_dir: Path) -> Path:
        """Create temporary config file."""
        config_path = tmp_path / "config" / "data.yaml"
        config_path.parent.mkdir(parents=True)

        config_content = f"""
version: "1.0"
download:
  exchange: binance
  market_type: futures
  symbols:
    - BTCUSDT
  timeframes:
    - 1h
  start_date: "2024-12-01"
  end_date: "2024-12-02"

validate:
  check_duplicates: true
  check_gaps: true
  check_ohlc: true

process:
  remove_duplicates: true
  fill_small_gaps: true
  max_gap_bars: 3

transform:
  output_format: parquet
  merge_files: true

paths:
  raw_data: "{data_dir}/raw"
  processed_data: "{data_dir}/processed"
  catalog: "{data_dir}/catalog"
  logs: "{tmp_path}/logs/data_pipeline"
"""
        config_path.write_text(config_content)
        return config_path

    def test_cli_help(self) -> None:
        """Test CLI help command works."""
        result = subprocess.run(
            [sys.executable, "scripts/data/pipeline.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Binance Data Pipeline" in result.stdout
        assert "download" in result.stdout
        assert "validate" in result.stdout
        assert "run" in result.stdout

    def test_cli_dry_run_download(self, config_file: Path) -> None:
        """Test dry-run download command."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/data/pipeline.py",
                "-c", str(config_file),
                "--dry-run",
                "download",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "BTCUSDT" in result.stdout
        assert "Estimated size" in result.stdout

    def test_cli_status(self, config_file: Path) -> None:
        """Test status command."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/data/pipeline.py",
                "-c", str(config_file),
                "status",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Pipeline Status" in result.stdout

    @pytest.mark.slow
    @pytest.mark.network
    def test_cli_download_real_api(self, config_file: Path, data_dir: Path) -> None:
        """Test actual download with real Binance API.

        This test makes real API calls and downloads actual data.
        Mark with @pytest.mark.slow and @pytest.mark.network.
        """
        result = subprocess.run(
            [
                sys.executable,
                "scripts/data/pipeline.py",
                "-c", str(config_file),
                "download",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Download failed: {result.stderr}"
        assert "rows downloaded" in result.stdout
        assert "Download complete" in result.stdout

        # Verify file was created
        raw_dir = data_dir / "raw" / "binance" / "BTCUSDT" / "1h"
        csv_files = list(raw_dir.glob("*.csv"))
        assert len(csv_files) >= 1, "No CSV files created"

    @pytest.mark.slow
    @pytest.mark.network
    def test_cli_full_pipeline_real_api(self, config_file: Path, data_dir: Path) -> None:
        """Test full pipeline run with real Binance API.

        This test makes real API calls and runs the complete pipeline.
        """
        result = subprocess.run(
            [
                sys.executable,
                "scripts/data/pipeline.py",
                "-c", str(config_file),
                "run",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        assert "DOWNLOAD" in result.stdout
        assert "VALIDATE" in result.stdout
        assert "PROCESS" in result.stdout
        assert "TRANSFORM" in result.stdout
        assert "PIPELINE SUCCESS" in result.stdout

        # Verify outputs
        raw_dir = data_dir / "raw" / "binance" / "BTCUSDT" / "1h"
        proc_dir = data_dir / "processed" / "binance" / "BTCUSDT" / "1h"
        catalog_dir = data_dir / "catalog"

        assert raw_dir.exists(), "Raw data directory not created"
        assert proc_dir.exists(), "Processed data directory not created"
        assert catalog_dir.exists(), "Catalog directory not created"
