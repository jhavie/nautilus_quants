"""Unit tests for reporting utilities."""

import re
from datetime import datetime
from pathlib import Path

import pytest

from nautilus_quants.backtest.utils.reporting import create_output_directory, generate_run_id


class TestGenerateRunId:
    """Tests for generate_run_id function."""

    def test_returns_string(self) -> None:
        """Test run_id is a string."""
        run_id = generate_run_id()
        assert isinstance(run_id, str)

    def test_format_matches_expected(self) -> None:
        """Test run_id matches YYYYMMDD_HHMMSS format."""
        run_id = generate_run_id()
        # Format: YYYYMMDD_HHMMSS
        pattern = r"^\d{8}_\d{6}$"
        assert re.match(pattern, run_id), f"Run ID {run_id} doesn't match expected format"

    def test_is_parseable_datetime(self) -> None:
        """Test run_id can be parsed back to datetime."""
        run_id = generate_run_id()
        parsed = datetime.strptime(run_id, "%Y%m%d_%H%M%S")
        assert isinstance(parsed, datetime)

    def test_unique_on_successive_calls(self) -> None:
        """Test successive calls produce different IDs (if time passes)."""
        import time

        run_id1 = generate_run_id()
        time.sleep(1.1)  # Wait for second to change
        run_id2 = generate_run_id()
        assert run_id1 != run_id2


class TestCreateOutputDirectory:
    """Tests for create_output_directory function."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Test directory is created."""
        base_dir = tmp_path / "backtest_runs"
        run_id = "20250121_120000"

        result = create_output_directory(base_dir, run_id)

        assert result.exists()
        assert result.is_dir()
        assert result == base_dir / run_id

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        """Test nested directories are created."""
        base_dir = tmp_path / "logs" / "backtest_runs"
        run_id = "20250121_120000"

        result = create_output_directory(base_dir, run_id)

        assert result.exists()
        assert result.parent.exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        """Test calling twice doesn't fail."""
        base_dir = tmp_path / "backtest_runs"
        run_id = "20250121_120000"

        result1 = create_output_directory(base_dir, run_id)
        result2 = create_output_directory(base_dir, run_id)

        assert result1 == result2
        assert result1.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Test accepts string path."""
        base_dir = str(tmp_path / "backtest_runs")
        run_id = "20250121_120000"

        result = create_output_directory(base_dir, run_id)

        assert result.exists()
        assert isinstance(result, Path)
