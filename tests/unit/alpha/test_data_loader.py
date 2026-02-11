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

    def test_bars_to_dataframe(self):
        mock_bar1 = MagicMock()
        mock_bar1.ts_event = 1_000_000_000_000  # nanoseconds
        mock_bar1.open = MagicMock(__float__=lambda s: 100.0)
        mock_bar1.high = MagicMock(__float__=lambda s: 105.0)
        mock_bar1.low = MagicMock(__float__=lambda s: 95.0)
        mock_bar1.close = MagicMock(__float__=lambda s: 102.0)
        mock_bar1.volume = MagicMock(__float__=lambda s: 1000.0)

        mock_bar2 = MagicMock()
        mock_bar2.ts_event = 2_000_000_000_000
        mock_bar2.open = MagicMock(__float__=lambda s: 102.0)
        mock_bar2.high = MagicMock(__float__=lambda s: 110.0)
        mock_bar2.low = MagicMock(__float__=lambda s: 100.0)
        mock_bar2.close = MagicMock(__float__=lambda s: 108.0)
        mock_bar2.volume = MagicMock(__float__=lambda s: 1500.0)

        df = CatalogDataLoader.bars_to_dataframe([mock_bar1, mock_bar2])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"

    def test_bars_to_dataframe_empty(self):
        df = CatalogDataLoader.bars_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
