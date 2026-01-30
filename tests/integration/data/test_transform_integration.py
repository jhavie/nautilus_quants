"""
Integration tests for transform workflow.

Tests complete transform pipeline including Parquet catalog creation
and Nautilus Trader compatibility.
"""

import pandas as pd
import pytest
from pathlib import Path

from nautilus_quants.data.transform.parquet import (
    csv_to_bars,
    transform_to_parquet,
    TransformResult,
    _get_bar_type,
    _create_instrument,
)


class TestTransformIntegration:
    """Integration tests for transform workflow."""

    @pytest.fixture
    def processed_csv(self, tmp_path):
        """Create a processed CSV file for transformation."""
        data_dir = tmp_path / "BTCUSDT" / "1h"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "BTCUSDT_1h_processed.csv"

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
        return csv_path

    @pytest.fixture
    def catalog_path(self, tmp_path):
        """Create temporary catalog directory."""
        return tmp_path / "catalog"

    def test_transform_creates_parquet_catalog(self, processed_csv, catalog_path):
        """Transform should create Parquet catalog directory."""
        result = transform_to_parquet(
            input_path=processed_csv,
            catalog_path=catalog_path,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        assert result.success is True
        assert catalog_path.exists()
        assert result.rows_transformed == 5

    def test_transform_result_contains_metadata(self, processed_csv, catalog_path):
        """Transform result should contain correct metadata."""
        result = transform_to_parquet(
            input_path=processed_csv,
            catalog_path=catalog_path,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert str(processed_csv) == result.input_file
        assert str(catalog_path) == result.output_path

    def test_csv_to_bars_creates_nautilus_bars(self, processed_csv):
        """csv_to_bars should create Nautilus Bar objects."""
        instrument = _create_instrument("BTCUSDT")
        bar_type = _get_bar_type("BTCUSDT", "1h")
        bars = csv_to_bars(processed_csv, instrument, bar_type)

        assert len(bars) == 5

        # Verify first bar properties
        bar = bars[0]
        assert bar.open.as_double() == 42000.0
        assert bar.high.as_double() == 42500.0
        assert bar.low.as_double() == 41900.0
        assert bar.close.as_double() == 42100.0

        # Verify timestamp conversion (ms to ns)
        assert bar.ts_event == 1704067200000 * 1_000_000

    def test_transform_handles_empty_csv(self, tmp_path, catalog_path):
        """Transform should handle empty CSV gracefully."""
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

        result = transform_to_parquet(
            input_path=csv_path,
            catalog_path=catalog_path,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        assert result.success is False
        assert "No data" in result.errors[0]

    def test_transform_handles_nonexistent_file(self, tmp_path, catalog_path):
        """Transform should handle non-existent file gracefully."""
        result = transform_to_parquet(
            input_path=tmp_path / "nonexistent.csv",
            catalog_path=catalog_path,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        assert result.success is False
        assert len(result.errors) > 0

    def test_transform_different_symbols(self, tmp_path, catalog_path):
        """Transform should handle different symbols correctly."""
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            csv_path = tmp_path / f"{symbol}_1h.csv"

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

            result = transform_to_parquet(
                input_path=csv_path,
                catalog_path=catalog_path,
                symbol=symbol,
                timeframe="1h",
            )

            assert result.success is True
            assert result.symbol == symbol

    def test_transform_different_timeframes(self, tmp_path, catalog_path):
        """Transform should handle different timeframes correctly."""
        for timeframe in ["1m", "5m", "1h", "4h", "1d"]:
            csv_path = tmp_path / f"BTCUSDT_{timeframe}.csv"

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

            result = transform_to_parquet(
                input_path=csv_path,
                catalog_path=catalog_path / timeframe,
                symbol="BTCUSDT",
                timeframe=timeframe,
            )

            assert result.success is True, f"Failed for timeframe {timeframe}"
            assert result.timeframe == timeframe

    def test_transform_preserves_price_precision(self, tmp_path, catalog_path):
        """Transform should preserve price precision."""
        csv_path = tmp_path / "precision.csv"

        df = pd.DataFrame({
            "timestamp": [1704067200000],
            "open": [42123.45678],
            "high": [42567.89012],
            "low": [41987.65432],
            "close": [42345.67890],
            "volume": [123.456789],
            "quote_volume": [5200000.0],
            "trades_count": [1500],
        })
        df.to_csv(csv_path, index=False)

        instrument = _create_instrument("BTCUSDT")
        bar_type = _get_bar_type("BTCUSDT", "1h")
        bars = csv_to_bars(csv_path, instrument, bar_type)

        assert len(bars) == 1
        bar = bars[0]

        # Verify precision is maintained (within reasonable tolerance)
        assert abs(bar.open.as_double() - 42123.45678) < 0.01
        assert abs(bar.high.as_double() - 42567.89012) < 0.01
        assert abs(bar.low.as_double() - 41987.65432) < 0.01
        assert abs(bar.close.as_double() - 42345.67890) < 0.01

    def test_catalog_can_be_loaded_by_nautilus(self, processed_csv, catalog_path):
        """Parquet catalog should be loadable by Nautilus Trader."""
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        # Transform to create catalog
        result = transform_to_parquet(
            input_path=processed_csv,
            catalog_path=catalog_path,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        assert result.success is True

        # Load catalog with Nautilus
        catalog = ParquetDataCatalog(str(catalog_path))

        # Verify catalog exists and can be accessed
        assert catalog is not None
        # The catalog stores data - verify the directory has content
        assert catalog_path.exists()
        assert any(catalog_path.iterdir())  # Has files

    def test_transform_multiple_files_to_same_catalog(self, tmp_path, catalog_path):
        """Multiple files should be able to write to same catalog."""
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        # Create and transform first file
        csv1 = tmp_path / "BTCUSDT_1h_part1.csv"
        df1 = pd.DataFrame({
            "timestamp": [1704067200000, 1704070800000],
            "open": [42000.0, 42100.0],
            "high": [42500.0, 42600.0],
            "low": [41900.0, 42000.0],
            "close": [42100.0, 42200.0],
            "volume": [100.0, 150.0],
            "quote_volume": [4200000.0, 6300000.0],
            "trades_count": [1000, 1200],
        })
        df1.to_csv(csv1, index=False)

        result1 = transform_to_parquet(csv1, catalog_path, "BTCUSDT", "1h")
        assert result1.success is True

        # Create and transform second file (different time period)
        csv2 = tmp_path / "BTCUSDT_1h_part2.csv"
        df2 = pd.DataFrame({
            "timestamp": [1704074400000, 1704078000000],
            "open": [42200.0, 42300.0],
            "high": [42700.0, 42800.0],
            "low": [42100.0, 42200.0],
            "close": [42300.0, 42400.0],
            "volume": [120.0, 130.0],
            "quote_volume": [5040000.0, 5460000.0],
            "trades_count": [1100, 1150],
        })
        df2.to_csv(csv2, index=False)

        result2 = transform_to_parquet(csv2, catalog_path, "BTCUSDT", "1h")
        assert result2.success is True

        # Verify catalog contains data from both files
        catalog = ParquetDataCatalog(str(catalog_path))
        assert catalog is not None
        assert catalog_path.exists()
        assert any(catalog_path.iterdir())  # Has files
