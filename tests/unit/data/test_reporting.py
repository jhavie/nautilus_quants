"""
Unit tests for the reporting infrastructure.

Tests run ID generation, report writing, and log directory creation.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from nautilus_quants.data.reporting import (
    ReportWriter,
    create_log_dir,
    generate_run_id,
)


class TestGenerateRunId:
    """Tests for generate_run_id function."""

    def test_run_id_format(self) -> None:
        """Test run ID follows YYYYMMDD_HHMMSS format."""
        run_id = generate_run_id()

        # Should be 15 characters: YYYYMMDD_HHMMSS
        assert len(run_id) == 15
        assert run_id[8] == "_"

        # Should be parseable as datetime
        parsed = datetime.strptime(run_id, "%Y%m%d_%H%M%S")
        assert parsed is not None

    def test_run_id_unique(self) -> None:
        """Test that consecutive run IDs are likely different."""
        import time

        run_id1 = generate_run_id()
        time.sleep(1.1)  # Wait just over a second
        run_id2 = generate_run_id()

        assert run_id1 != run_id2


class TestCreateLogDir:
    """Tests for create_log_dir function."""

    def test_create_log_dir(self, tmp_path: Path) -> None:
        """Test log directory creation."""
        base_path = tmp_path / "logs"
        run_id = "20240101_120000"

        log_dir = create_log_dir(base_path, run_id)

        assert log_dir.exists()
        assert log_dir == base_path / run_id

    def test_create_log_dir_nested(self, tmp_path: Path) -> None:
        """Test log directory creation with nested path."""
        base_path = tmp_path / "logs" / "data_pipeline"
        run_id = "20240101_120000"

        log_dir = create_log_dir(base_path, run_id)

        assert log_dir.exists()
        assert log_dir.parent.exists()

    def test_create_log_dir_idempotent(self, tmp_path: Path) -> None:
        """Test that creating the same log dir twice is idempotent."""
        base_path = tmp_path / "logs"
        run_id = "20240101_120000"

        log_dir1 = create_log_dir(base_path, run_id)
        log_dir2 = create_log_dir(base_path, run_id)

        assert log_dir1 == log_dir2
        assert log_dir1.exists()


class TestReportWriter:
    """Tests for ReportWriter class."""

    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        """Create a temporary log directory."""
        log_path = tmp_path / "logs" / "20240101_120000"
        log_path.mkdir(parents=True)
        return log_path

    @pytest.fixture
    def writer(self, log_dir: Path) -> ReportWriter:
        """Create a ReportWriter instance."""
        return ReportWriter(log_dir)

    def test_write_step_report(self, writer: ReportWriter, log_dir: Path) -> None:
        """Test writing a step report."""
        summary = {
            "symbols_total": 2,
            "symbols_successful": 2,
            "total_rows": 17520,
        }
        details = [
            {"symbol": "BTCUSDT", "files": 1, "size_mb": 10.5, "time_seconds": 5.0, "success": True},
            {"symbol": "ETHUSDT", "files": 1, "size_mb": 8.2, "time_seconds": 4.5, "success": True},
        ]

        report_path = writer.write_step_report(
            step_name="download",
            run_id="20240101_120000",
            summary=summary,
            details_by_symbol=details,
        )

        assert report_path.exists()
        assert report_path.name == "download_report.txt"

        content = report_path.read_text()
        assert "DOWNLOAD REPORT" in content
        assert "Run ID: 20240101_120000" in content
        assert "symbols_total: 2" in content
        assert "BTCUSDT" in content
        assert "ETHUSDT" in content

    def test_write_step_report_with_errors(
        self, writer: ReportWriter, log_dir: Path
    ) -> None:
        """Test writing a step report with errors."""
        summary = {"status": "failed"}
        errors = ["API rate limit exceeded", "Network connection lost"]

        report_path = writer.write_step_report(
            step_name="download",
            run_id="20240101_120000",
            summary=summary,
            errors=errors,
        )

        content = report_path.read_text()
        assert "ERRORS" in content
        assert "API rate limit exceeded" in content
        assert "Network connection lost" in content

    def test_write_summary_report(self, writer: ReportWriter, log_dir: Path) -> None:
        """Test writing the summary report."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 5, 30)
        config = {
            "download": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "timeframes": ["1h", "4h"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "exchange": "binance",
                "market_type": "futures",
            }
        }
        step_results = [
            {"step_name": "download", "success": True, "duration_seconds": 50.0, "summary": {}},
            {"step_name": "validate", "success": True, "duration_seconds": 5.0, "summary": {}},
            {"step_name": "process", "success": True, "duration_seconds": 10.0, "summary": {}},
            {"step_name": "transform", "success": True, "duration_seconds": 15.0, "summary": {}},
        ]

        report_path = writer.write_summary_report(
            run_id="20240101_120000",
            start_time=start_time,
            end_time=end_time,
            config=config,
            step_results=step_results,
        )

        assert report_path.exists()
        assert report_path.name == "summary_20240101_120000.txt"

        content = report_path.read_text()
        assert "DATA PIPELINE SUMMARY REPORT" in content
        assert "Status: SUCCESS" in content
        assert "DOWNLOAD" in content
        assert "VALIDATE" in content
        assert "PROCESS" in content
        assert "TRANSFORM" in content

    def test_write_summary_report_with_failure(
        self, writer: ReportWriter, log_dir: Path
    ) -> None:
        """Test summary report shows FAILED when any step fails."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 1, 0)
        config = {"download": {"symbols": [], "timeframes": []}}
        step_results = [
            {"step_name": "download", "success": True, "duration_seconds": 50.0, "summary": {}},
            {"step_name": "validate", "success": False, "duration_seconds": 5.0, "summary": {}},
        ]

        report_path = writer.write_summary_report(
            run_id="20240101_120000",
            start_time=start_time,
            end_time=end_time,
            config=config,
            step_results=step_results,
        )

        content = report_path.read_text()
        assert "Status: FAILED" in content

    def test_write_details_json(self, writer: ReportWriter, log_dir: Path) -> None:
        """Test writing the details JSON file."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 5, 30)
        config = {
            "download": {
                "symbols": ["BTCUSDT"],
                "timeframes": ["1h"],
            }
        }
        steps = [
            {
                "step_name": "download",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "elapsed_seconds": 330.0,
                "success": True,
                "summary": {"rows": 8760},
            }
        ]

        report_path = writer.write_details_json(
            run_id="20240101_120000",
            start_time=start_time,
            end_time=end_time,
            config=config,
            steps=steps,
        )

        assert report_path.exists()
        assert report_path.name == "details_20240101_120000.json"

        with open(report_path) as f:
            data = json.load(f)

        assert data["run_id"] == "20240101_120000"
        assert data["success"] is True
        assert data["elapsed_seconds"] == 330.0
        assert len(data["steps"]) == 1
        assert data["steps"][0]["step_name"] == "download"

    def test_write_details_json_with_errors(
        self, writer: ReportWriter, log_dir: Path
    ) -> None:
        """Test details JSON includes errors."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 1, 0)
        errors = ["Error 1", "Error 2"]

        report_path = writer.write_details_json(
            run_id="20240101_120000",
            start_time=start_time,
            end_time=end_time,
            config={},
            steps=[],
            errors=errors,
        )

        with open(report_path) as f:
            data = json.load(f)

        assert data["errors"] == ["Error 1", "Error 2"]

    def test_writer_creates_directory_if_missing(self, tmp_path: Path) -> None:
        """Test ReportWriter creates log directory if it doesn't exist."""
        new_dir = tmp_path / "new_logs"
        assert not new_dir.exists()

        writer = ReportWriter(new_dir)

        assert new_dir.exists()
