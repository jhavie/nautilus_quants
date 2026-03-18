"""
Unit tests for consistency validation module.

Tests duplicate detection, gap detection, monotonic checks, and OHLC validation.
"""

import pandas as pd
import pytest

from nautilus_quants.data.validate.consistency import (
    check_duplicates,
    check_gaps,
    check_monotonic,
    check_ohlc_relationships,
    INTERVAL_MS,
)
from nautilus_quants.data.types import (
    ValidationCheckType,
    ValidationSeverity,
)


class TestCheckDuplicates:
    """Tests for duplicate timestamp detection."""

    def test_no_duplicates_returns_empty(self):
        """DataFrame without duplicates should return no issues."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
        })

        issues = check_duplicates(df)

        assert len(issues) == 0

    def test_single_duplicate_detected(self):
        """Single duplicate timestamp should be detected."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000, 1704074400000],
        })

        issues = check_duplicates(df)

        assert len(issues) == 1
        assert issues[0].check_type == ValidationCheckType.DUPLICATES
        assert issues[0].severity == ValidationSeverity.WARNING
        assert issues[0].timestamp == 1704067200000
        assert "Duplicate" in issues[0].message

    def test_multiple_duplicates_detected(self):
        """Multiple duplicate timestamps should each be detected."""
        df = pd.DataFrame({
            "timestamp": [
                1704067200000, 1704067200000,  # Duplicate 1
                1704074400000, 1704074400000,  # Duplicate 2
            ],
        })

        issues = check_duplicates(df)

        assert len(issues) == 2
        timestamps = {i.timestamp for i in issues}
        assert 1704067200000 in timestamps
        assert 1704074400000 in timestamps

    def test_triple_duplicate_counted_correctly(self):
        """Triple duplicate should report correct count."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000, 1704067200000],
        })

        issues = check_duplicates(df)

        assert len(issues) == 1
        assert issues[0].details["count"] == 3

    def test_missing_timestamp_column_handled(self):
        """DataFrame without timestamp column should return empty."""
        df = pd.DataFrame({
            "other_column": [1, 2, 3],
        })

        issues = check_duplicates(df)

        assert len(issues) == 0


class TestCheckGaps:
    """Tests for gap detection in timestamp sequence."""

    def test_no_gaps_returns_empty(self):
        """Consecutive timestamps should return no issues."""
        # 1h interval = 3600000 ms
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
        })

        issues = check_gaps(df, "1h")

        assert len(issues) == 0

    def test_small_gap_detected_as_warning(self):
        """Gap of 2 bars (within max_gap_bars) should be WARNING."""
        # 1h interval = 3600000 ms
        # Gap of 2 hours = 2 bars missing
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704078000000],  # 3 hour gap = 2 missing bars
        })

        issues = check_gaps(df, "1h", max_gap_bars=3)

        assert len(issues) == 1
        assert issues[0].check_type == ValidationCheckType.GAPS
        assert issues[0].severity == ValidationSeverity.WARNING
        assert issues[0].details["gap_bars"] == 2

    def test_large_gap_detected_as_error(self):
        """Gap larger than max_gap_bars should be ERROR."""
        # Gap of 5 hours = 4 missing bars
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704085200000],  # 5 hour gap
        })

        issues = check_gaps(df, "1h", max_gap_bars=3)

        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
        assert issues[0].details["gap_bars"] == 4

    def test_different_timeframes_supported(self):
        """Gap detection should work for different timeframes."""
        # 4h interval = 14400000 ms
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704096000000],  # 8 hour gap = 1 missing 4h bar
        })

        issues = check_gaps(df, "4h", max_gap_bars=3)

        assert len(issues) == 1
        assert issues[0].details["gap_bars"] == 1
        assert issues[0].severity == ValidationSeverity.WARNING

    def test_interval_ms_mapping_correct(self):
        """Verify interval milliseconds are correctly defined."""
        assert INTERVAL_MS["1m"] == 60 * 1000
        assert INTERVAL_MS["1h"] == 60 * 60 * 1000
        assert INTERVAL_MS["4h"] == 4 * 60 * 60 * 1000
        assert INTERVAL_MS["1d"] == 24 * 60 * 60 * 1000

    def test_single_row_returns_empty(self):
        """Single row DataFrame should return no gap issues."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
        })

        issues = check_gaps(df, "1h")

        assert len(issues) == 0


class TestCheckMonotonic:
    """Tests for monotonic timestamp validation."""

    def test_strictly_increasing_returns_empty(self):
        """Strictly increasing timestamps should return no issues."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
        })

        issues = check_monotonic(df)

        assert len(issues) == 0

    def test_non_increasing_detected(self):
        """Non-increasing timestamp should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000, 1704074400000],  # Duplicate = not strictly increasing
        })

        issues = check_monotonic(df)

        assert len(issues) == 1
        assert issues[0].check_type == ValidationCheckType.MONOTONIC
        assert issues[0].severity == ValidationSeverity.ERROR

    def test_decreasing_timestamp_detected(self):
        """Decreasing timestamp should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704074400000, 1704070800000, 1704067200000],  # Decreasing
        })

        issues = check_monotonic(df)

        assert len(issues) == 2  # Two non-increasing transitions

    def test_single_row_returns_empty(self):
        """Single row DataFrame should return no issues."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
        })

        issues = check_monotonic(df)

        assert len(issues) == 0


class TestCheckOHLCRelationships:
    """Tests for OHLC price relationship validation."""

    def test_valid_ohlc_returns_empty(self):
        """Valid OHLC relationships should return no issues."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        issues = check_ohlc_relationships(df)

        assert len(issues) == 0

    def test_high_less_than_open_close_detected(self):
        """High < max(open, close) should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [41500.0],  # Invalid: high < open
            "low": [41000.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        issues = check_ohlc_relationships(df)

        ohlc_issues = [i for i in issues if i.check_type == ValidationCheckType.OHLC_RELATIONSHIP]
        assert len(ohlc_issues) >= 1
        assert any("High" in i.message for i in ohlc_issues)

    def test_low_greater_than_open_close_detected(self):
        """Low > min(open, close) should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [43000.0],
            "low": [42500.0],  # Invalid: low > open
            "close": [42100.0],
            "volume": [100.0],
        })

        issues = check_ohlc_relationships(df)

        ohlc_issues = [i for i in issues if i.check_type == ValidationCheckType.OHLC_RELATIONSHIP]
        assert len(ohlc_issues) >= 1
        assert any("Low" in i.message for i in ohlc_issues)

    def test_low_greater_than_high_detected(self):
        """Low > High should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [41000.0],  # Invalid: high < low
            "low": [42000.0],
            "close": [41500.0],
            "volume": [100.0],
        })

        issues = check_ohlc_relationships(df)

        ohlc_issues = [i for i in issues if i.check_type == ValidationCheckType.OHLC_RELATIONSHIP]
        assert len(ohlc_issues) >= 1

    def test_zero_price_detected(self):
        """Zero price should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [0.0],  # Invalid: zero price
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        issues = check_ohlc_relationships(df)

        assert len(issues) >= 1
        assert any("Price <= 0" in i.message for i in issues)

    def test_negative_price_detected(self):
        """Negative price should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [-42000.0],  # Invalid: negative price
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        issues = check_ohlc_relationships(df)

        assert len(issues) >= 1

    def test_negative_volume_detected(self):
        """Negative volume should be detected as ERROR."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [-100.0],  # Invalid: negative volume
        })

        issues = check_ohlc_relationships(df)

        volume_issues = [i for i in issues if i.check_type == ValidationCheckType.VOLUME]
        assert len(volume_issues) == 1
        assert "Negative volume" in volume_issues[0].message

    def test_zero_volume_allowed(self):
        """Zero volume should be allowed (no trades in period)."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [0.0],  # Valid: zero volume
        })

        issues = check_ohlc_relationships(df)

        volume_issues = [i for i in issues if i.check_type == ValidationCheckType.VOLUME]
        assert len(volume_issues) == 0

    def test_missing_columns_handled(self):
        """DataFrame missing OHLC columns should return empty."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "other": [42000.0],
        })

        issues = check_ohlc_relationships(df)

        assert len(issues) == 0

    def test_multiple_rows_validated(self):
        """All rows should be validated, not just first."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
            "open": [42000.0, 42100.0, -42200.0],  # Third row invalid
            "high": [42500.0, 42600.0, 42700.0],
            "low": [41900.0, 42000.0, 42100.0],
            "close": [42100.0, 42200.0, 42300.0],
            "volume": [100.0, 150.0, 120.0],
        })

        issues = check_ohlc_relationships(df)

        assert len(issues) >= 1
        # Issue should be on the third row (index 2)
        assert any(i.row_index == 2 for i in issues)
