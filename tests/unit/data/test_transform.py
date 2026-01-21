"""
Unit tests for transform module.

Tests CSV to Bar conversion and Parquet transformation.
"""

import pandas as pd
import pytest
from pathlib import Path
from decimal import Decimal

from nautilus_quants.data.transform.parquet import (
    csv_to_bars,
    transform_to_parquet,
    TransformResult,
    TIMEFRAME_TO_STEP,
    _get_bar_type,
)


class TestTimeframeMapping:
    """Tests for timeframe to step mapping."""

    def test_timeframe_mapping_defined(self):
        """Verify timeframe mappings are correct."""
        assert TIMEFRAME_TO_STEP["1m"] == 1
        assert TIMEFRAME_TO_STEP["5m"] == 5
        assert TIMEFRAME_TO_STEP["15m"] == 15
        assert TIMEFRAME_TO_STEP["30m"] == 30
        assert TIMEFRAME_TO_STEP["1h"] == 60
        assert TIMEFRAME_TO_STEP["4h"] == 240
        assert TIMEFRAME_TO_STEP["1d"] == 1440


class TestGetBarType:
    """Tests for bar type creation."""

    def test_creates_bar_type_for_symbol(self):
        """Should create valid BarType for symbol."""
        bar_type = _get_bar_type("BTCUSDT", "1h")

        assert bar_type is not None
        assert "BTCUSDT" in str(bar_type.instrument_id)
        assert "BINANCE" in str(bar_type.instrument_id)

    def test_different_timeframes(self):
        """Should handle different timeframes."""
        bar_type_1h = _get_bar_type("BTCUSDT", "1h")
        bar_type_4h = _get_bar_type("BTCUSDT", "4h")

        # Both should be valid but different
        assert bar_type_1h is not None
        assert bar_type_4h is not None

    def test_different_symbols(self):
        """Should handle different symbols."""
        bar_type_btc = _get_bar_type("BTCUSDT", "1h")
        bar_type_eth = _get_bar_type("ETHUSDT", "1h")

        assert "BTCUSDT" in str(bar_type_btc.instrument_id)
        assert "ETHUSDT" in str(bar_type_eth.instrument_id)


class TestCsvToBars:
    """Tests for CSV to Bar conversion."""

    def test_converts_csv_to_bars(self, tmp_path):
        """Should convert CSV data to Bar objects."""
        csv_path = tmp_path / "test.csv"

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

        bars = csv_to_bars(csv_path, "BTCUSDT", "1h")

        assert len(bars) == 2
        # Check first bar
        assert bars[0].open.as_double() == 42000.0
        assert bars[0].high.as_double() == 42500.0
        assert bars[0].low.as_double() == 41900.0
        assert bars[0].close.as_double() == 42100.0

    def test_preserves_all_ohlcv_values(self, tmp_path):
        """Should preserve all OHLCV values accurately."""
        csv_path = tmp_path / "test.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42123.45],
            "high": [42567.89],
            "low": [41987.65],
            "close": [42345.67],
            "volume": [123.456],
            "quote_volume": [5200000.0],
            "trades_count": [1500],
        })
        df.to_csv(csv_path, index=False)

        bars = csv_to_bars(csv_path, "BTCUSDT", "1h")

        assert len(bars) == 1
        bar = bars[0]
        assert abs(bar.open.as_double() - 42123.45) < 0.01
        assert abs(bar.high.as_double() - 42567.89) < 0.01
        assert abs(bar.low.as_double() - 41987.65) < 0.01
        assert abs(bar.close.as_double() - 42345.67) < 0.01

    def test_converts_timestamp_to_nanoseconds(self, tmp_path):
        """Timestamp should be converted from ms to ns."""
        csv_path = tmp_path / "test.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000],  # Milliseconds
            "open": [42000.0],
            "high": [42500.0],
            "low": [41900.0],
            "close": [42100.0],
            "volume": [100.0],
            "quote_volume": [4200000.0],
            "trades_count": [1000],
        })
        df.to_csv(csv_path, index=False)

        bars = csv_to_bars(csv_path, "BTCUSDT", "1h")

        # ts_event should be in nanoseconds (ms * 1_000_000)
        expected_ns = 1704067200000 * 1_000_000
        assert bars[0].ts_event == expected_ns

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """Empty CSV should return empty list."""
        csv_path = tmp_path / "empty.csv"

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

        bars = csv_to_bars(csv_path, "BTCUSDT", "1h")

        assert len(bars) == 0

    def test_handles_different_symbols(self, tmp_path):
        """Should handle different symbol names."""
        csv_path = tmp_path / "test.csv"

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
        df.to_csv(csv_path, index=False)

        bars = csv_to_bars(csv_path, "ETHUSDT", "4h")

        assert len(bars) == 1
        assert "ETHUSDT" in str(bars[0].bar_type.instrument_id)


class TestTransformToParquet:
    """Tests for Parquet transformation function."""

    def test_transform_returns_success_result(self, tmp_path):
        """Successful transform should return success result."""
        csv_path = tmp_path / "test.csv"
        catalog_path = tmp_path / "catalog"

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

        result = transform_to_parquet(csv_path, catalog_path, "BTCUSDT", "1h")

        assert result.success is True
        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert result.rows_transformed == 2
        assert len(result.errors) == 0

    def test_transform_creates_catalog_directory(self, tmp_path):
        """Transform should create catalog directory if not exists."""
        csv_path = tmp_path / "test.csv"
        catalog_path = tmp_path / "new_catalog"

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
        df.to_csv(csv_path, index=False)

        assert not catalog_path.exists()

        result = transform_to_parquet(csv_path, catalog_path, "BTCUSDT", "1h")

        assert catalog_path.exists()
        assert result.success is True

    def test_transform_empty_csv_returns_failure(self, tmp_path):
        """Empty CSV should return failure result."""
        csv_path = tmp_path / "empty.csv"
        catalog_path = tmp_path / "catalog"

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

        result = transform_to_parquet(csv_path, catalog_path, "BTCUSDT", "1h")

        assert result.success is False
        assert "No data" in result.errors[0]

    def test_transform_invalid_file_returns_failure(self, tmp_path):
        """Invalid input file should return failure result."""
        csv_path = tmp_path / "nonexistent.csv"
        catalog_path = tmp_path / "catalog"

        result = transform_to_parquet(csv_path, catalog_path, "BTCUSDT", "1h")

        assert result.success is False
        assert len(result.errors) > 0

    def test_transform_result_dataclass(self):
        """TransformResult should have correct fields."""
        result = TransformResult(
            success=True,
            symbol="BTCUSDT",
            timeframe="1h",
            input_file="/path/to/input.csv",
            output_path="/path/to/catalog",
            rows_transformed=100,
            errors=[],
        )

        assert result.success is True
        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert result.rows_transformed == 100

    def test_transform_result_with_errors(self):
        """TransformResult should store error messages."""
        result = TransformResult(
            success=False,
            symbol="BTCUSDT",
            timeframe="1h",
            input_file="/path/to/input.csv",
            output_path="/path/to/catalog",
            rows_transformed=0,
            errors=["Error 1", "Error 2"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert "Error 1" in result.errors


class TestTransformIntegration:
    """Integration-like tests for transform workflow."""

    def test_full_transform_workflow(self, tmp_path):
        """Test complete transform workflow from CSV to catalog."""
        # Create processed CSV
        csv_path = tmp_path / "BTCUSDT_1h_processed.csv"
        catalog_path = tmp_path / "catalog"

        df = pd.DataFrame({
            "timestamp": [
                1704067200000,
                1704070800000,
                1704074400000,
                1704078000000,
                1704081600000,
            ],
            "open": [42000.0, 42100.0, 42200.0, 42300.0, 42400.0],
            "high": [42500.0, 42600.0, 42700.0, 42800.0, 42900.0],
            "low": [41900.0, 42000.0, 42100.0, 42200.0, 42300.0],
            "close": [42100.0, 42200.0, 42300.0, 42400.0, 42500.0],
            "volume": [100.0, 150.0, 120.0, 130.0, 140.0],
            "quote_volume": [4200000.0, 6300000.0, 5040000.0, 5460000.0, 5880000.0],
            "trades_count": [1000, 1200, 1100, 1150, 1250],
        })
        df.to_csv(csv_path, index=False)

        # Transform
        result = transform_to_parquet(csv_path, catalog_path, "BTCUSDT", "1h")

        # Verify result
        assert result.success is True
        assert result.rows_transformed == 5
        assert catalog_path.exists()

    def test_transform_different_timeframes(self, tmp_path):
        """Should handle different timeframes correctly."""
        for timeframe in ["1m", "5m", "1h", "4h", "1d"]:
            csv_path = tmp_path / f"test_{timeframe}.csv"
            catalog_path = tmp_path / f"catalog_{timeframe}"

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
            df.to_csv(csv_path, index=False)

            result = transform_to_parquet(csv_path, catalog_path, "BTCUSDT", timeframe)

            assert result.success is True, f"Failed for timeframe {timeframe}"
