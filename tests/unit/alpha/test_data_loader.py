"""Tests for CatalogDataLoader."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nautilus_quants.alpha.data_loader import CatalogDataLoader


class TestCatalogDataLoader:
    """Test CatalogDataLoader."""

    def test_init(self):
        with patch("nautilus_quants.alpha.data_loader.ParquetDataCatalog"):
            loader = CatalogDataLoader("/data/catalog", bar_spec="1h")
            assert loader._bar_spec == "1h"

    def test_init_custom_bar_spec(self):
        with patch("nautilus_quants.alpha.data_loader.ParquetDataCatalog"):
            loader = CatalogDataLoader("/data/catalog", bar_spec="4h")
            assert loader._bar_spec == "4h"

    @patch("nautilus_quants.alpha.data_loader.ParquetDataCatalog")
    def test_load_bars_returns_dict(self, mock_catalog_cls):
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog

        # Simulate catalog returning bars
        mock_bar = MagicMock()
        mock_bar.ts_event = 1000000000
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 95.0
        mock_bar.close = 102.0
        mock_bar.volume = 1000.0
        mock_catalog.bars.return_value = [mock_bar]

        loader = CatalogDataLoader("/data/catalog")
        result = loader.load_bars(["BTCUSDT.BINANCE"])

        assert isinstance(result, dict)
        assert "BTCUSDT.BINANCE" in result
        assert isinstance(result["BTCUSDT.BINANCE"], list)

    @patch("nautilus_quants.alpha.data_loader.ParquetDataCatalog")
    def test_load_bars_empty_result(self, mock_catalog_cls):
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_catalog.bars.return_value = []

        loader = CatalogDataLoader("/data/catalog")
        result = loader.load_bars(["BTCUSDT.BINANCE"])

        assert "BTCUSDT.BINANCE" in result
        assert result["BTCUSDT.BINANCE"] == []

    @patch("nautilus_trader.model.data.Bar")
    def test_bars_to_dataframe(self, mock_bar_model):
        """bars_to_dataframe should use Bar.to_dict() for conversion."""
        mock_bar1 = MagicMock()
        mock_bar2 = MagicMock()

        # Bar.to_dict() returns dicts with string OHLCV values
        mock_bar_model.to_dict.side_effect = [
            {
                "ts_event": 1_000_000_000_000,
                "open": "100.00", "high": "105.00", "low": "95.00",
                "close": "102.00", "volume": "1000.00",
                "type": "Bar", "bar_type": "TEST",
            },
            {
                "ts_event": 2_000_000_000_000,
                "open": "102.00", "high": "110.00", "low": "100.00",
                "close": "108.00", "volume": "1500.00",
                "type": "Bar", "bar_type": "TEST",
            },
        ]

        df = CatalogDataLoader.bars_to_dataframe([mock_bar1, mock_bar2])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        # Verify float conversion from string values
        assert df["close"].iloc[0] == 102.0
        assert df["volume"].iloc[1] == 1500.0

    def test_bars_to_dataframe_empty(self):
        df = CatalogDataLoader.bars_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
