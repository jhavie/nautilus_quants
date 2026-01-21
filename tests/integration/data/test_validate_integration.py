"""
Integration tests for validation workflow.

Tests complete validation workflow including file loading and report generation.
"""

import pandas as pd
import pytest
from pathlib import Path

from nautilus_quants.data.validate.integrity import validate_file
from nautilus_quants.data.validate.consistency import (
    check_duplicates,
    check_gaps,
    check_monotonic,
    check_ohlc_relationships,
)
from nautilus_quants.data.types import (
    ValidationCheckType,
    ValidationSeverity,
)


class TestValidationIntegration:
    """Integration tests for validation workflow."""

    @pytest.fixture
    def valid_csv(self, tmp_path):
        """Create a valid CSV file for testing."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_2024-01-01.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000, 1704078000000],
            "open": [42000.0, 42100.0, 42200.0, 42300.0],
            "high": [42500.0, 42600.0, 42700.0, 42800.0],
            "low": [41900.0, 42000.0, 42100.0, 42200.0],
            "close": [42100.0, 42200.0, 42300.0, 42400.0],
            "volume": [100.0, 150.0, 120.0, 130.0],
            "quote_volume": [4200000.0, 6300000.0, 5040000.0, 5460000.0],
            "trades_count": [1000, 1200, 1100, 1150],
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def csv_with_issues(self, tmp_path):
        """Create a CSV file with validation issues."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_issues.csv"

        df = pd.DataFrame({
            "timestamp": [
                1704067200000,
                1704067200000,  # Duplicate
                1704078000000,  # Gap (missing 1704070800000, 1704074400000)
                1704081600000,
            ],
            "open": [42000.0, 42100.0, 42200.0, -42300.0],  # Last has negative price
            "high": [42500.0, 42600.0, 42700.0, 42800.0],
            "low": [41900.0, 42000.0, 42100.0, 42200.0],
            "close": [42100.0, 42200.0, 42300.0, 42400.0],
            "volume": [100.0, 150.0, 120.0, 130.0],
            "quote_volume": [4200000.0, 6300000.0, 5040000.0, 5460000.0],
            "trades_count": [1000, 1200, 1100, 1150],
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_validate_file_valid_csv_passes(self, valid_csv):
        """Valid CSV file should pass validation."""
        report = validate_file(valid_csv)

        assert report.passed is True
        assert report.total_rows == 4
        assert report.error_count == 0
        assert report.symbol == "BTCUSDT"
        assert report.timeframe == "1h"

    def test_validate_file_detects_duplicates(self, csv_with_issues):
        """Validation should detect duplicate timestamps."""
        report = validate_file(csv_with_issues)

        assert report.duplicate_count >= 1
        assert any(
            issue.check_type == ValidationCheckType.DUPLICATES
            for issue in report.issues
        )

    def test_validate_file_detects_gaps(self, csv_with_issues):
        """Validation should detect gaps in timestamp sequence."""
        report = validate_file(csv_with_issues)

        assert report.gap_count >= 1
        assert any(
            issue.check_type == ValidationCheckType.GAPS
            for issue in report.issues
        )

    def test_validate_file_detects_invalid_ohlc(self, csv_with_issues):
        """Validation should detect invalid OHLC relationships."""
        report = validate_file(csv_with_issues)

        assert report.invalid_ohlc_count >= 1
        assert any(
            issue.check_type == ValidationCheckType.OHLC_RELATIONSHIP
            for issue in report.issues
        )

    def test_validate_file_extracts_symbol_timeframe(self, valid_csv):
        """Validation should extract symbol and timeframe from path."""
        report = validate_file(valid_csv)

        assert report.symbol == "BTCUSDT"
        assert report.timeframe == "1h"

    def test_validate_file_nonexistent_file(self, tmp_path):
        """Validation should handle non-existent file."""
        nonexistent = tmp_path / "nonexistent.csv"
        report = validate_file(nonexistent)

        assert report.passed is False
        assert report.error_count > 0

    def test_validate_file_empty_csv(self, tmp_path):
        """Validation should handle empty CSV file."""
        data_dir = tmp_path / "ETHUSDT" / "4h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "ETHUSDT_4h_empty.csv"

        df = pd.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "quote_volume": [],
            "trades_count": [],
        })
        df.to_csv(csv_path, index=False)

        report = validate_file(csv_path)

        assert report.total_rows == 0

    def test_full_validation_pipeline(self, tmp_path):
        """Test complete validation pipeline with multiple checks."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_full_test.csv"

        # Create CSV with multiple issue types
        df = pd.DataFrame({
            "timestamp": [
                1704067200000,
                1704070800000,
                1704070800000,  # Duplicate
                1704078000000,  # Gap
                1704081600000,
            ],
            "open": [42000.0, 42100.0, 42150.0, 42200.0, 0.0],  # Zero price
            "high": [42500.0, 42600.0, 42650.0, 42700.0, 42800.0],
            "low": [41900.0, 42000.0, 42050.0, 42100.0, 42200.0],
            "close": [42100.0, 42200.0, 42250.0, 42300.0, 42400.0],
            "volume": [100.0, 150.0, 125.0, 120.0, -130.0],  # Negative volume
            "quote_volume": [4200000.0, 6300000.0, 5250000.0, 5040000.0, 5460000.0],
            "trades_count": [1000, 1200, 1100, 1100, 1150],
        })
        df.to_csv(csv_path, index=False)

        report = validate_file(csv_path)

        # Should detect multiple issues
        assert report.passed is False
        assert report.duplicate_count >= 1
        assert report.gap_count >= 1
        assert report.invalid_ohlc_count >= 1

        # Verify issue types present
        issue_types = {issue.check_type for issue in report.issues}
        assert ValidationCheckType.DUPLICATES in issue_types
        assert ValidationCheckType.GAPS in issue_types

    def test_validation_report_contains_all_fields(self, valid_csv):
        """Validation report should contain all expected fields."""
        report = validate_file(valid_csv)

        # Verify report fields
        assert hasattr(report, "symbol")
        assert hasattr(report, "timeframe")
        assert hasattr(report, "file_path")
        assert hasattr(report, "total_rows")
        assert hasattr(report, "passed")
        assert hasattr(report, "error_count")
        assert hasattr(report, "warning_count")
        assert hasattr(report, "duplicate_count")
        assert hasattr(report, "gap_count")
        assert hasattr(report, "invalid_ohlc_count")
        assert hasattr(report, "issues")

    def test_validate_multiple_files(self, tmp_path):
        """Validation should handle multiple files independently."""
        reports = []

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            data_dir = tmp_path / symbol / "1h"
            data_dir.mkdir(parents=True)
            csv_path = data_dir / f"{symbol}_1h_data.csv"

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
            df.to_csv(csv_path, index=False)

            report = validate_file(csv_path)
            reports.append(report)

        assert len(reports) == 2
        assert reports[0].symbol == "BTCUSDT"
        assert reports[1].symbol == "ETHUSDT"
        assert all(r.passed for r in reports)
