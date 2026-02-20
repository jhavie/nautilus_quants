"""
Unit tests for integrity validation module.

Tests schema validation, file existence checks, and data type validation.
"""

import pandas as pd
import pytest
from pathlib import Path
import tempfile

from nautilus_quants.data.validate.integrity import (
    validate_schema,
    validate_file,
    REQUIRED_COLUMNS,
    COLUMN_TYPES,
)
from nautilus_quants.data.types import (
    ValidationCheckType,
    ValidationSeverity,
)


class TestValidateSchema:
    """Tests for schema validation function."""

    def test_valid_schema_passes(self):
        """Valid DataFrame with all required columns should pass."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000],
            "open": [42000.0, 42100.0],
            "high": [42500.0, 42600.0],
            "low": [41900.0, 42000.0],
            "close": [42100.0, 42500.0],
            "volume": [100.0, 150.0],
            "quote_volume": [4200000.0, 6300000.0],
            "trades_count": [1000, 1200],
            "taker_buy_base_volume": [50.0, 75.0],
            "taker_buy_quote_volume": [2100000.0, 3150000.0],
        })

        issues = validate_schema(df)

        assert len(issues) == 0

    def test_missing_columns_detected(self):
        """Missing required columns should be reported as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            # Missing: low, close, volume, quote_volume, trades_count
        })

        issues = validate_schema(df)

        assert len(issues) > 0
        schema_issues = [i for i in issues if i.check_type == ValidationCheckType.SCHEMA]
        assert len(schema_issues) == 1
        assert schema_issues[0].severity == ValidationSeverity.ERROR
        assert "Missing required columns" in schema_issues[0].message

    def test_all_required_columns_defined(self):
        """Verify REQUIRED_COLUMNS contains expected columns."""
        expected = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades_count", "taker_buy_base_volume", "taker_buy_quote_volume"]
        assert set(REQUIRED_COLUMNS) == set(expected)

    def test_column_types_defined(self):
        """Verify COLUMN_TYPES contains expected type mappings."""
        assert COLUMN_TYPES["timestamp"] == "int64"
        assert COLUMN_TYPES["open"] == "float64"
        assert COLUMN_TYPES["high"] == "float64"
        assert COLUMN_TYPES["low"] == "float64"
        assert COLUMN_TYPES["close"] == "float64"
        assert COLUMN_TYPES["volume"] == "float64"
        assert COLUMN_TYPES["quote_volume"] == "float64"
        assert COLUMN_TYPES["trades_count"] == "int64"

    def test_extra_columns_allowed(self):
        """Extra columns beyond required should not cause errors."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
            "quote_volume": [4200000.0],
            "trades_count": [1000],
            "taker_buy_base_volume": [50.0],
            "taker_buy_quote_volume": [2100000.0],
            "extra_column": ["some_value"],  # Extra column
        })

        issues = validate_schema(df)

        assert len(issues) == 0


class TestValidateFile:
    """Tests for file-level validation function."""

    def test_file_not_found_returns_error(self, tmp_path):
        """Non-existent file should return ERROR."""
        non_existent = tmp_path / "BTCUSDT" / "1h" / "nonexistent.csv"

        report = validate_file(non_existent)

        assert report.passed is False
        assert report.error_count == 1
        assert "File not found" in report.issues[0].message

    def test_valid_csv_passes_validation(self, tmp_path):
        """Valid CSV file should pass validation."""
        # Create directory structure
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_20240101_20240107.csv"

        # Write valid data
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
            "open": [42000.0, 42100.0, 42200.0],
            "high": [42500.0, 42600.0, 42700.0],
            "low": [41900.0, 42000.0, 42100.0],
            "close": [42100.0, 42200.0, 42300.0],
            "volume": [100.0, 150.0, 120.0],
            "quote_volume": [4200000.0, 6300000.0, 5040000.0],
            "trades_count": [1000, 1200, 1100],
            "taker_buy_base_volume": [50.0, 75.0, 60.0],
            "taker_buy_quote_volume": [2100000.0, 3150000.0, 2520000.0],
        })
        df.to_csv(csv_path, index=False)

        report = validate_file(csv_path)

        assert report.passed is True
        assert report.symbol == "BTCUSDT"
        assert report.timeframe == "1h"
        assert report.total_rows == 3
        assert report.error_count == 0

    def test_invalid_csv_format_returns_error(self, tmp_path):
        """Malformed CSV should return ERROR."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_20240101_20240107.csv"

        # Write invalid content
        with open(csv_path, "w") as f:
            f.write("not,valid,csv\n")
            f.write("missing columns\n")

        report = validate_file(csv_path)

        # Should fail schema validation due to missing columns
        assert report.passed is False
        assert report.error_count > 0

    def test_extracts_symbol_and_timeframe_from_path(self, tmp_path):
        """Symbol and timeframe should be extracted from directory structure."""
        data_dir = tmp_path / "ETHUSDT" / "4h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "ETHUSDT_4h_20240101_20240107.csv"

        # Write valid data
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [2200.0],
            "high": [2250.0],
            "low": [2180.0],
            "close": [2230.0],
            "volume": [500.0],
            "quote_volume": [1100000.0],
            "trades_count": [2000],
            "taker_buy_base_volume": [50.0],
            "taker_buy_quote_volume": [2100000.0],
        })
        df.to_csv(csv_path, index=False)

        report = validate_file(csv_path)

        assert report.symbol == "ETHUSDT"
        assert report.timeframe == "4h"

    def test_specific_checks_can_be_selected(self, tmp_path):
        """Only specified checks should be run when checks parameter provided."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "test.csv"

        # Write data with duplicates (would fail DUPLICATES check)
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000],  # Duplicate
            "open": [42000.0, 42100.0],
            "high": [42500.0, 42600.0],
            "low": [41900.0, 42000.0],
            "close": [42100.0, 42200.0],
            "volume": [100.0, 150.0],
            "quote_volume": [4200000.0, 6300000.0],
            "trades_count": [1000, 1200],
            "taker_buy_base_volume": [50.0, 75.0],
            "taker_buy_quote_volume": [2100000.0, 3150000.0],
        })
        df.to_csv(csv_path, index=False)

        # Only run schema check (should pass, duplicates not checked)
        report = validate_file(csv_path, checks=[ValidationCheckType.SCHEMA])

        # Schema passes, duplicates not checked
        assert report.error_count == 0
        # But duplicate_count should be 0 since we didn't check for it
        assert report.duplicate_count == 0

    def test_empty_dataframe_handled(self, tmp_path):
        """Empty CSV file should be handled gracefully."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "empty.csv"

        # Write header only
        with open(csv_path, "w") as f:
            f.write("timestamp,open,high,low,close,volume,quote_volume,trades_count,taker_buy_base_volume,taker_buy_quote_volume\n")

        report = validate_file(csv_path)

        assert report.total_rows == 0
        # Empty file with valid schema should pass
        assert report.passed is True
