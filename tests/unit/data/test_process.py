"""
Unit tests for data processing module.

Tests duplicate removal, gap filling, and invalid OHLC removal.
"""

import pandas as pd
import pytest
from pathlib import Path

from nautilus_quants.data.process.processors import (
    remove_duplicates,
    fill_gaps,
    remove_invalid_ohlc,
    process_data,
    ProcessConfig,
    INTERVAL_MS,
)
from nautilus_quants.data.types import ProcessingAction


class TestRemoveDuplicates:
    """Tests for duplicate removal function."""

    def test_no_duplicates_unchanged(self):
        """DataFrame without duplicates should remain unchanged."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
            "close": [42000.0, 42100.0, 42200.0],
        })

        result_df, actions = remove_duplicates(df)

        assert len(result_df) == 3
        assert len(actions) == 0

    def test_keep_last_duplicate(self):
        """With keep='last', last occurrence should be retained."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000, 1704074400000],
            "close": [42000.0, 42100.0, 42200.0],  # Second value should be kept
        })

        result_df, actions = remove_duplicates(df, keep="last")

        assert len(result_df) == 2
        assert result_df.iloc[0]["close"] == 42100.0  # Last duplicate kept
        assert len(actions) == 1
        assert actions[0].action_type == "remove_duplicate"

    def test_keep_first_duplicate(self):
        """With keep='first', first occurrence should be retained."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000, 1704074400000],
            "close": [42000.0, 42100.0, 42200.0],  # First value should be kept
        })

        result_df, actions = remove_duplicates(df, keep="first")

        assert len(result_df) == 2
        assert result_df.iloc[0]["close"] == 42000.0  # First duplicate kept
        assert len(actions) == 1

    def test_multiple_duplicate_groups(self):
        """Multiple groups of duplicates should all be handled."""
        df = pd.DataFrame({
            "timestamp": [
                1704067200000, 1704067200000,  # Duplicate group 1
                1704074400000, 1704074400000,  # Duplicate group 2
            ],
            "close": [42000.0, 42100.0, 42200.0, 42300.0],
        })

        result_df, actions = remove_duplicates(df, keep="last")

        assert len(result_df) == 2
        assert len(actions) == 2

    def test_triple_duplicate_removes_two(self):
        """Triple duplicate should remove 2 rows, keep 1."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000, 1704067200000],
            "close": [42000.0, 42100.0, 42200.0],
        })

        result_df, actions = remove_duplicates(df, keep="last")

        assert len(result_df) == 1
        assert result_df.iloc[0]["close"] == 42200.0
        assert len(actions) == 2

    def test_action_contains_before_value(self):
        """Action should contain the removed row's data."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000],
            "close": [42000.0, 42100.0],
        })

        _, actions = remove_duplicates(df, keep="last")

        assert actions[0].before_value is not None
        assert actions[0].before_value["close"] == 42000.0

    def test_missing_timestamp_column_unchanged(self):
        """DataFrame without timestamp column should remain unchanged."""
        df = pd.DataFrame({
            "other": [1, 2, 3],
        })

        result_df, actions = remove_duplicates(df)

        assert len(result_df) == 3
        assert len(actions) == 0


class TestFillGaps:
    """Tests for gap filling function."""

    def test_no_gaps_unchanged(self):
        """Consecutive timestamps should remain unchanged."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],  # 1h intervals
            "open": [42000.0, 42100.0, 42200.0],
            "high": [42500.0, 42600.0, 42700.0],
            "low": [41900.0, 42000.0, 42100.0],
            "close": [42100.0, 42200.0, 42300.0],
            "volume": [100.0, 150.0, 120.0],
            "quote_volume": [4200000.0, 6300000.0, 5040000.0],
            "trades_count": [1000, 1200, 1100],
        })

        result_df, actions = fill_gaps(df, "1h")

        assert len(result_df) == 3
        assert len(actions) == 0

    def test_small_gap_filled(self):
        """Gap within max_gap_bars should be filled."""
        # Gap of 2 hours = 1 missing bar
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704074400000],  # 2h gap
            "open": [42000.0, 42200.0],
            "high": [42500.0, 42700.0],
            "low": [41900.0, 42100.0],
            "close": [42100.0, 42300.0],
            "volume": [100.0, 120.0],
            "quote_volume": [4200000.0, 5040000.0],
            "trades_count": [1000, 1100],
        })

        result_df, actions = fill_gaps(df, "1h", max_gap_bars=3)

        assert len(result_df) == 3  # Original 2 + 1 filled
        assert len(actions) == 1
        assert actions[0].action_type == "fill_gap"

    def test_large_gap_not_filled(self):
        """Gap larger than max_gap_bars should not be filled."""
        # Gap of 5 hours = 4 missing bars
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704085200000],  # 5h gap
            "open": [42000.0, 42200.0],
            "high": [42500.0, 42700.0],
            "low": [41900.0, 42100.0],
            "close": [42100.0, 42300.0],
            "volume": [100.0, 120.0],
            "quote_volume": [4200000.0, 5040000.0],
            "trades_count": [1000, 1100],
        })

        result_df, actions = fill_gaps(df, "1h", max_gap_bars=3)

        assert len(result_df) == 2  # No fill
        assert len(actions) == 0

    def test_filled_row_uses_forward_fill(self):
        """Filled rows should use previous close as all OHLC."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704074400000],  # 2h gap
            "open": [42000.0, 42200.0],
            "high": [42500.0, 42700.0],
            "low": [41900.0, 42100.0],
            "close": [42100.0, 42300.0],
            "volume": [100.0, 120.0],
            "quote_volume": [4200000.0, 5040000.0],
            "trades_count": [1000, 1100],
        })

        result_df, actions = fill_gaps(df, "1h", max_gap_bars=3)

        # The filled row should use close of first row as OHLC
        filled_row = result_df[result_df["timestamp"] == 1704070800000].iloc[0]
        assert filled_row["open"] == 42100.0  # Previous close
        assert filled_row["high"] == 42100.0
        assert filled_row["low"] == 42100.0
        assert filled_row["close"] == 42100.0
        assert filled_row["volume"] == 0.0

    def test_was_filled_column_added(self):
        """Filled rows should have was_filled=True."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704074400000],
            "open": [42000.0, 42200.0],
            "high": [42500.0, 42700.0],
            "low": [41900.0, 42100.0],
            "close": [42100.0, 42300.0],
            "volume": [100.0, 120.0],
            "quote_volume": [4200000.0, 5040000.0],
            "trades_count": [1000, 1100],
        })

        result_df, actions = fill_gaps(df, "1h", max_gap_bars=3)

        assert "was_filled" in result_df.columns
        # Check that at least one row was filled (the gap row)
        assert len(actions) == 1  # One gap filled
        # The filled row should have was_filled=True
        filled_row = result_df[result_df["timestamp"] == 1704070800000]
        assert len(filled_row) == 1
        # Check was_filled is True (handle both bool and float representation)
        was_filled_val = filled_row.iloc[0]["was_filled"]
        assert was_filled_val == True or was_filled_val == 1.0

    def test_multiple_gaps_filled(self):
        """Multiple gaps should all be filled."""
        # Two gaps of 1 bar each
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704074400000, 1704081600000],  # 2h gaps
            "open": [42000.0, 42200.0, 42400.0],
            "high": [42500.0, 42700.0, 42900.0],
            "low": [41900.0, 42100.0, 42300.0],
            "close": [42100.0, 42300.0, 42500.0],
            "volume": [100.0, 120.0, 140.0],
            "quote_volume": [4200000.0, 5040000.0, 5880000.0],
            "trades_count": [1000, 1100, 1200],
        })

        result_df, actions = fill_gaps(df, "1h", max_gap_bars=3)

        assert len(result_df) == 5  # 3 original + 2 filled
        assert len(actions) == 2

    def test_different_timeframe(self):
        """Gap filling should work for different timeframes."""
        # 4h timeframe, gap of 8 hours = 1 missing bar
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704096000000],  # 8h gap
            "open": [42000.0, 42200.0],
            "high": [42500.0, 42700.0],
            "low": [41900.0, 42100.0],
            "close": [42100.0, 42300.0],
            "volume": [100.0, 120.0],
            "quote_volume": [4200000.0, 5040000.0],
            "trades_count": [1000, 1100],
        })

        result_df, actions = fill_gaps(df, "4h", max_gap_bars=3)

        assert len(result_df) == 3  # 2 original + 1 filled
        assert len(actions) == 1


class TestRemoveInvalidOHLC:
    """Tests for invalid OHLC removal function."""

    def test_valid_ohlc_unchanged(self):
        """Valid OHLC data should remain unchanged."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        result_df, actions = remove_invalid_ohlc(df)

        assert len(result_df) == 1
        assert len(actions) == 0

    def test_high_less_than_close_removed(self):
        """Row with high < close should be removed."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000],
            "open": [42000.0, 42000.0],
            "high": [41500.0, 42500.0],  # First row invalid
            "low": [41000.0, 41900.0],
            "close": [42100.0, 42100.0],
            "volume": [100.0, 100.0],
        })

        result_df, actions = remove_invalid_ohlc(df)

        assert len(result_df) == 1
        assert result_df.iloc[0]["timestamp"] == 1704070800000
        assert len(actions) == 1
        assert actions[0].action_type == "remove_invalid_ohlc"

    def test_low_greater_than_open_removed(self):
        """Row with low > open should be removed."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [43000.0],
            "low": [42500.0],  # Invalid: low > open
            "close": [42100.0],
            "volume": [100.0],
        })

        result_df, actions = remove_invalid_ohlc(df)

        assert len(result_df) == 0
        assert len(actions) == 1

    def test_negative_price_removed(self):
        """Row with negative price should be removed."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [-42000.0],  # Invalid
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        result_df, actions = remove_invalid_ohlc(df)

        assert len(result_df) == 0

    def test_zero_price_removed(self):
        """Row with zero price should be removed."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [0.0],  # Invalid
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        result_df, actions = remove_invalid_ohlc(df)

        assert len(result_df) == 0

    def test_negative_volume_removed(self):
        """Row with negative volume should be removed."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [-100.0],  # Invalid
        })

        result_df, actions = remove_invalid_ohlc(df)

        assert len(result_df) == 0

    def test_zero_volume_allowed(self):
        """Row with zero volume should be allowed."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [0.0],  # Valid
        })

        result_df, actions = remove_invalid_ohlc(df)

        assert len(result_df) == 1
        assert len(actions) == 0

    def test_action_contains_before_value(self):
        """Action should contain the removed row's data."""
        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [-42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
        })

        _, actions = remove_invalid_ohlc(df)

        assert actions[0].before_value is not None
        assert actions[0].before_value["open"] == -42000.0


class TestProcessData:
    """Tests for the process_data orchestration function."""

    def test_process_data_creates_output_file(self, tmp_path):
        """process_data should create the output CSV file."""
        # Create input file
        input_dir = tmp_path / "BTCUSDT" / "1h"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "BTCUSDT_1h_test.csv"

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
        df.to_csv(input_path, index=False)

        # Output path
        output_dir = tmp_path / "output" / "BTCUSDT" / "1h"
        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        report = process_data(input_path, output_path)

        assert output_path.exists()
        assert report.input_rows == 2
        assert report.output_rows == 2

    def test_process_data_removes_duplicates(self, tmp_path):
        """process_data should remove duplicates when configured."""
        input_dir = tmp_path / "BTCUSDT" / "1h"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "test.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067200000],  # Duplicate
            "open": [42000.0, 42100.0],
            "high": [42500.0, 42600.0],
            "low": [41900.0, 42000.0],
            "close": [42100.0, 42200.0],
            "volume": [100.0, 150.0],
            "quote_volume": [4200000.0, 6300000.0],
            "trades_count": [1000, 1200],
        })
        df.to_csv(input_path, index=False)

        output_path = tmp_path / "output.csv"
        config = ProcessConfig(remove_duplicates=True)

        report = process_data(input_path, output_path, config=config)

        assert report.input_rows == 2
        assert report.output_rows == 1
        assert report.duplicates_removed == 1

    def test_process_data_fills_gaps(self, tmp_path):
        """process_data should fill gaps when configured."""
        input_dir = tmp_path / "BTCUSDT" / "1h"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "test.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704074400000],  # 2h gap
            "open": [42000.0, 42100.0],
            "high": [42500.0, 42600.0],
            "low": [41900.0, 42000.0],
            "close": [42100.0, 42200.0],
            "volume": [100.0, 150.0],
            "quote_volume": [4200000.0, 6300000.0],
            "trades_count": [1000, 1200],
        })
        df.to_csv(input_path, index=False)

        output_path = tmp_path / "output.csv"
        config = ProcessConfig(fill_small_gaps=True, max_gap_bars=3)

        report = process_data(input_path, output_path, config=config)

        assert report.input_rows == 2
        assert report.output_rows == 3  # 2 + 1 filled
        assert report.gaps_filled == 1

    def test_process_data_removes_invalid_ohlc(self, tmp_path):
        """process_data should remove invalid OHLC rows when configured."""
        input_dir = tmp_path / "BTCUSDT" / "1h"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "test.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000],
            "open": [-42000.0, 42100.0],  # First row invalid
            "high": [42500.0, 42600.0],
            "low": [41900.0, 42000.0],
            "close": [42100.0, 42200.0],
            "volume": [100.0, 150.0],
            "quote_volume": [4200000.0, 6300000.0],
            "trades_count": [1000, 1200],
        })
        df.to_csv(input_path, index=False)

        output_path = tmp_path / "output.csv"
        config = ProcessConfig(remove_invalid_ohlc=True)

        report = process_data(input_path, output_path, config=config)

        assert report.input_rows == 2
        assert report.output_rows == 1
        assert report.invalid_rows_removed == 1

    def test_process_data_extracts_symbol_timeframe(self, tmp_path):
        """process_data should extract symbol and timeframe from path."""
        input_dir = tmp_path / "ETHUSDT" / "4h"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "test.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [2200.0],
            "high": [2250.0],
            "low": [2180.0],
            "close": [2230.0],
            "volume": [500.0],
            "quote_volume": [1100000.0],
            "trades_count": [2000],
        })
        df.to_csv(input_path, index=False)

        output_path = tmp_path / "output.csv"
        report = process_data(input_path, output_path)

        assert report.symbol == "ETHUSDT"
        assert report.timeframe == "4h"

    def test_process_config_defaults(self):
        """ProcessConfig should have correct defaults."""
        config = ProcessConfig()

        assert config.remove_duplicates is True
        assert config.keep_duplicate == "last"
        assert config.fill_small_gaps is True
        assert config.max_gap_bars == 3
        assert config.remove_invalid_ohlc is True
