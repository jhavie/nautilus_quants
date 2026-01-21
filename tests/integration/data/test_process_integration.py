"""
Integration tests for processing workflow.

Tests complete processing pipeline including duplicate removal,
gap filling, and invalid OHLC handling.
"""

import pandas as pd
import pytest
from pathlib import Path

from nautilus_quants.data.process.processors import (
    process_data,
    ProcessConfig,
    remove_duplicates,
    fill_gaps,
    remove_invalid_ohlc,
)


class TestProcessIntegration:
    """Integration tests for processing workflow."""

    @pytest.fixture
    def raw_csv_with_issues(self, tmp_path):
        """Create a raw CSV file with various issues to process."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_raw.csv"

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
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory."""
        output = tmp_path / "processed" / "BTCUSDT" / "1h"
        output.mkdir(parents=True)
        return output

    def test_process_data_creates_output_file(self, raw_csv_with_issues, output_dir):
        """Processing should create output file."""
        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        report = process_data(raw_csv_with_issues, output_path)

        assert output_path.exists()
        assert report.input_rows == 5
        # Note: output_rows = input - duplicates + gaps_filled
        # With 1 duplicate removed and 1 gap filled, output equals input
        assert report.duplicates_removed == 1

    def test_process_removes_duplicates(self, raw_csv_with_issues, output_dir):
        """Processing should remove duplicate timestamps."""
        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        report = process_data(raw_csv_with_issues, output_path)

        # Verify duplicates removed
        assert report.duplicates_removed == 1

        # Verify output has no duplicates
        df = pd.read_csv(output_path)
        assert df["timestamp"].is_unique

    def test_process_fills_gaps(self, raw_csv_with_issues, output_dir):
        """Processing should fill small gaps in data."""
        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        report = process_data(raw_csv_with_issues, output_path)

        # Verify gaps were filled
        assert report.gaps_filled >= 1

        # Verify output has was_filled column
        df = pd.read_csv(output_path)
        assert "was_filled" in df.columns
        assert df["was_filled"].any()  # At least one row was filled

    def test_process_removes_invalid_ohlc(self, tmp_path, output_dir):
        """Processing should remove rows with invalid OHLC relationships."""
        # Create CSV with invalid OHLC
        data_dir = tmp_path / "ETHUSDT" / "4h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "ETHUSDT_4h_raw.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704081600000],
            "open": [2200.0, -2300.0],  # Second row has negative open
            "high": [2250.0, 2350.0],
            "low": [2180.0, 2280.0],
            "close": [2230.0, 2330.0],
            "volume": [500.0, 600.0],
            "quote_volume": [1100000.0, 1400000.0],
            "trades_count": [2000, 2200],
        })
        df.to_csv(csv_path, index=False)

        output_path = tmp_path / "processed" / "ETHUSDT_4h_processed.csv"

        report = process_data(csv_path, output_path)

        assert report.invalid_rows_removed == 1
        assert report.output_rows == 1

        # Verify output has valid data only
        df_out = pd.read_csv(output_path)
        assert len(df_out) == 1
        assert df_out.iloc[0]["open"] > 0

    def test_process_preserves_valid_data(self, tmp_path, output_dir):
        """Processing should preserve valid data unchanged."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_clean.csv"

        # Create clean data with no issues
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000, 1704074400000],
            "open": [42000.0, 42100.0, 42200.0],
            "high": [42500.0, 42600.0, 42700.0],
            "low": [41900.0, 42000.0, 42100.0],
            "close": [42100.0, 42200.0, 42300.0],
            "volume": [100.0, 150.0, 120.0],
            "quote_volume": [4200000.0, 6300000.0, 5040000.0],
            "trades_count": [1000, 1200, 1100],
        })
        df.to_csv(csv_path, index=False)

        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        report = process_data(csv_path, output_path)

        assert report.input_rows == 3
        assert report.output_rows == 3
        assert report.duplicates_removed == 0
        assert report.gaps_filled == 0
        assert report.invalid_rows_removed == 0

        # Verify output matches input
        df_out = pd.read_csv(output_path)
        assert len(df_out) == 3

    def test_process_config_options(self, raw_csv_with_issues, output_dir):
        """Processing should respect configuration options."""
        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        # Configure to NOT fill gaps
        config = ProcessConfig(
            remove_duplicates=True,
            keep_duplicate="first",
            fill_small_gaps=False,
            max_gap_bars=3,
            remove_invalid_ohlc=True,
        )

        report = process_data(raw_csv_with_issues, output_path, config=config)

        # Duplicates should be removed but gaps not filled
        assert report.duplicates_removed >= 1
        assert report.gaps_filled == 0

    def test_process_max_gap_bars_limit(self, tmp_path, output_dir):
        """Processing should not fill gaps larger than max_gap_bars."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_large_gap.csv"

        # Create data with 5-hour gap (4 missing bars)
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704085200000],  # 5 hour gap
            "open": [42000.0, 42100.0],
            "high": [42500.0, 42600.0],
            "low": [41900.0, 42000.0],
            "close": [42100.0, 42200.0],
            "volume": [100.0, 150.0],
            "quote_volume": [4200000.0, 6300000.0],
            "trades_count": [1000, 1200],
        })
        df.to_csv(csv_path, index=False)

        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        config = ProcessConfig(max_gap_bars=3)  # Only fill gaps of 3 bars or less

        report = process_data(csv_path, output_path, config=config)

        # Gap should NOT be filled (4 bars > max 3)
        assert report.gaps_filled == 0
        df_out = pd.read_csv(output_path)
        assert len(df_out) == 2

    def test_process_report_contains_actions(self, raw_csv_with_issues, output_dir):
        """Processing report should contain detailed actions."""
        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        report = process_data(raw_csv_with_issues, output_path)

        # Verify report has actions
        assert len(report.actions) > 0

        # Verify action types
        action_types = {action.action_type for action in report.actions}
        assert "remove_duplicate" in action_types or "fill_gap" in action_types

    def test_process_extracts_symbol_timeframe(self, raw_csv_with_issues, output_dir):
        """Processing should extract symbol and timeframe from path."""
        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        report = process_data(raw_csv_with_issues, output_path)

        assert report.symbol == "BTCUSDT"
        assert report.timeframe == "1h"

    def test_process_filled_rows_have_correct_values(self, tmp_path, output_dir):
        """Filled rows should use forward-fill values correctly."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_gap.csv"

        # Create data with 2-hour gap (1 missing bar)
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704074400000],  # 2 hour gap
            "open": [42000.0, 42200.0],
            "high": [42500.0, 42700.0],
            "low": [41900.0, 42100.0],
            "close": [42100.0, 42300.0],
            "volume": [100.0, 120.0],
            "quote_volume": [4200000.0, 5040000.0],
            "trades_count": [1000, 1100],
        })
        df.to_csv(csv_path, index=False)

        output_path = output_dir / "BTCUSDT_1h_processed.csv"

        config = ProcessConfig(fill_small_gaps=True, max_gap_bars=3)
        report = process_data(csv_path, output_path, config=config)

        # Verify gap was filled
        assert report.gaps_filled == 1

        # Verify filled row values
        df_out = pd.read_csv(output_path)
        assert len(df_out) == 3

        # Find the filled row
        filled_row = df_out[df_out["was_filled"] == True]
        assert len(filled_row) == 1

        # Filled row should have previous close as all OHLC values
        assert filled_row.iloc[0]["open"] == 42100.0  # Previous close
        assert filled_row.iloc[0]["high"] == 42100.0
        assert filled_row.iloc[0]["low"] == 42100.0
        assert filled_row.iloc[0]["close"] == 42100.0
        assert filled_row.iloc[0]["volume"] == 0.0

    def test_process_multiple_files(self, tmp_path):
        """Processing should handle multiple files independently."""
        reports = []

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            # Create input
            data_dir = tmp_path / "raw" / symbol / "1h"
            data_dir.mkdir(parents=True)
            csv_path = data_dir / f"{symbol}_1h_raw.csv"

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

            # Process
            output_dir = tmp_path / "processed" / symbol / "1h"
            output_dir.mkdir(parents=True)
            output_path = output_dir / f"{symbol}_1h_processed.csv"

            report = process_data(csv_path, output_path)
            reports.append(report)

        assert len(reports) == 2
        assert reports[0].symbol == "BTCUSDT"
        assert reports[1].symbol == "ETHUSDT"
