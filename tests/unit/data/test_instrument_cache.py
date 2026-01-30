"""Tests for instrument_cache module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nautilus_quants.data.transform.instrument_cache import (
    DEFAULT_PRECISION,
    SYMBOL_PATTERNS,
    InstrumentCache,
    InstrumentPrecision,
)


class TestInstrumentPrecision:
    """Tests for InstrumentPrecision dataclass."""

    def test_basic_creation(self):
        """Test creating InstrumentPrecision."""
        info = InstrumentPrecision(
            symbol="1000PEPEUSDT",
            price_precision=7,
            quantity_precision=0,
            tick_size="0.0000001",
            step_size="0.001",
            min_price="0.0000001",
            max_price="200",
            min_qty="100",
            max_qty="10000000",
        )
        assert info.symbol == "1000PEPEUSDT"
        assert info.price_precision == 7
        assert info.tick_size == "0.0000001"


class TestInstrumentCacheInit:
    """Tests for InstrumentCache initialization."""

    def test_default_init(self):
        """Test default initialization uses current directory."""
        cache = InstrumentCache()
        assert cache.cache_path == Path.cwd() / ".instrument_cache.json"

    def test_custom_cache_dir(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)
            assert cache.cache_path == Path(tmpdir) / ".instrument_cache.json"

    def test_path_string(self):
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(str(tmpdir))
            assert cache.cache_path == Path(tmpdir) / ".instrument_cache.json"


class TestInferPrecisionFromSymbol:
    """Tests for heuristic precision inference."""

    def test_1000pepe_pattern(self):
        """Test 1000PEPE pattern matching."""
        info = InstrumentCache.infer_precision_from_symbol("1000PEPEUSDT")
        assert info.price_precision == 7
        assert info.tick_size == "0.0000001"

    def test_1000shib_pattern(self):
        """Test 1000SHIB pattern matching."""
        info = InstrumentCache.infer_precision_from_symbol("1000SHIBUSDT")
        assert info.price_precision == 8
        assert info.tick_size == "0.00000001"

    def test_1000bonk_pattern(self):
        """Test 1000BONK pattern matching."""
        info = InstrumentCache.infer_precision_from_symbol("1000BONKUSDT")
        assert info.price_precision == 7

    def test_btc_pattern(self):
        """Test BTC pattern matching."""
        info = InstrumentCache.infer_precision_from_symbol("BTCUSDT")
        assert info.price_precision == 2
        assert info.tick_size == "0.01"

    def test_eth_pattern(self):
        """Test ETH pattern matching."""
        info = InstrumentCache.infer_precision_from_symbol("ETHUSDT")
        assert info.price_precision == 2

    def test_default_fallback(self):
        """Test default precision for unknown symbols."""
        info = InstrumentCache.infer_precision_from_symbol("UNKNOWN123")
        assert info.price_precision == DEFAULT_PRECISION["price_precision"]
        assert info.tick_size == DEFAULT_PRECISION["tick_size"]


class TestCacheReadWrite:
    """Tests for cache persistence."""

    def test_save_and_load_cache(self):
        """Test saving and loading cache from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            # Create test data
            test_data = {
                "1000PEPEUSDT": {
                    "symbol": "1000PEPEUSDT",
                    "price_precision": 7,
                    "quantity_precision": 0,
                    "tick_size": "0.0000001",
                    "step_size": "0.001",
                    "min_price": "0.0000001",
                    "max_price": "200",
                    "min_qty": "100",
                    "max_qty": "10000000",
                },
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "price_precision": 2,
                    "quantity_precision": 3,
                    "tick_size": "0.01",
                    "step_size": "0.001",
                    "min_price": "0.01",
                    "max_price": "1000000",
                    "min_qty": "0.001",
                    "max_qty": "10000",
                },
            }

            # Save cache
            cache._save_cache(test_data)

            # Load cache
            loaded = cache._load_cache()
            assert loaded is not None
            assert "1000PEPEUSDT" in loaded
            assert loaded["1000PEPEUSDT"]["price_precision"] == 7
            assert loaded["BTCUSDT"]["price_precision"] == 2

    def test_cache_expiration(self):
        """Test that expired cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)
            cache.CACHE_TTL_HOURS = 0  # Expire immediately

            # Save old cache
            test_data = {"1000PEPEUSDT": {"price_precision": 7}}
            cache._save_cache(test_data)

            # Should return None due to expiration
            loaded = cache._load_cache()
            assert loaded is None

    def test_load_nonexistent_cache(self):
        """Test loading cache when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)
            loaded = cache._load_cache()
            assert loaded is None


class TestFetchExchangeInfo:
    """Tests for API fetching."""

    @pytest.mark.asyncio
    async def test_fetch_exchange_info_success(self):
        """Test successful API fetch."""
        mock_response = {
            "symbols": [
                {
                    "symbol": "1000PEPEUSDT",
                    "status": "TRADING",
                    "pricePrecision": 7,
                    "quantityPrecision": 0,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.0000001", "minPrice": "0.0000001", "maxPrice": "200"},
                        {"filterType": "LOT_SIZE", "minQty": "100", "maxQty": "10000000", "stepSize": "1"},
                    ],
                },
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "pricePrecision": 2,
                    "quantityPrecision": 3,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01", "maxPrice": "1000000"},
                        {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "10000", "stepSize": "0.001"},
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            # Mock aiohttp session
            mock_session = AsyncMock()
            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = Mock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response_obj)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await cache.fetch_exchange_info("futures")

            assert "1000PEPEUSDT" in result
            assert result["1000PEPEUSDT"].price_precision == 7
            assert result["BTCUSDT"].price_precision == 2

    @pytest.mark.asyncio
    async def test_skip_non_trading_symbols(self):
        """Test that non-TRADING symbols are skipped."""
        mock_response = {
            "symbols": [
                {
                    "symbol": "DELISTEDUSDT",
                    "status": "BREAK",
                    "pricePrecision": 2,
                    "quantityPrecision": 3,
                    "filters": [],
                },
                {
                    "symbol": "ACTIVEUSDT",
                    "status": "TRADING",
                    "pricePrecision": 2,
                    "quantityPrecision": 3,
                    "filters": [],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            mock_session = AsyncMock()
            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = Mock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response_obj)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await cache.fetch_exchange_info()

            assert "DELISTEDUSDT" not in result
            assert "ACTIVEUSDT" in result


class TestGetOrFetchExchangeInfo:
    """Tests for get_or_fetch_exchange_info method."""

    def test_returns_cached_data(self):
        """Test that cached data is returned without API call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            # Save valid cache
            test_data = {
                "1000PEPEUSDT": {
                    "symbol": "1000PEPEUSDT",
                    "price_precision": 7,
                    "quantity_precision": 0,
                    "tick_size": "0.0000001",
                    "step_size": "0.001",
                    "min_price": "0.0000001",
                    "max_price": "200",
                    "min_qty": "100",
                    "max_qty": "10000000",
                }
            }
            cache._save_cache(test_data)

            # Should return cached data without calling API
            result = cache.get_or_fetch_exchange_info()

            assert "1000PEPEUSDT" in result
            assert result["1000PEPEUSDT"].price_precision == 7

    def test_fetches_on_cache_miss(self):
        """Test that API is called when cache is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            # Mock fetch_exchange_info
            mock_info = InstrumentPrecision(
                symbol="1000PEPEUSDT",
                price_precision=7,
                quantity_precision=0,
                tick_size="0.0000001",
                step_size="0.001",
                min_price="0.0000001",
                max_price="200",
                min_qty="100",
                max_qty="10000000",
            )

            with patch.object(cache, "fetch_exchange_info", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = {"1000PEPEUSDT": mock_info}
                result = cache.get_or_fetch_exchange_info()

            assert "1000PEPEUSDT" in result
            assert result["1000PEPEUSDT"].price_precision == 7

    def test_returns_empty_dict_on_api_failure(self):
        """Test that empty dict is returned when API fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            with patch.object(cache, "fetch_exchange_info", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.side_effect = Exception("Network error")
                result = cache.get_or_fetch_exchange_info()

            assert result == {}


class TestGetSymbolInfo:
    """Tests for get_symbol_info method."""

    def test_get_existing_symbol(self):
        """Test getting info for an existing symbol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            # Save cache with symbol
            test_data = {
                "1000PEPEUSDT": {
                    "symbol": "1000PEPEUSDT",
                    "price_precision": 7,
                    "quantity_precision": 0,
                    "tick_size": "0.0000001",
                    "step_size": "0.001",
                    "min_price": "0.0000001",
                    "max_price": "200",
                    "min_qty": "100",
                    "max_qty": "10000000",
                }
            }
            cache._save_cache(test_data)

            info = cache.get_symbol_info("1000PEPEUSDT")
            assert info is not None
            assert info.price_precision == 7

    def test_get_nonexistent_symbol(self):
        """Test getting info for a non-existent symbol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)
            cache._exchange_info = {}  # Empty cache

            info = cache.get_symbol_info("NONEXISTENT")
            assert info is None


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_end_to_end_with_cache(self):
        """Test full workflow: fetch, cache, and retrieve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InstrumentCache(tmpdir)

            # Mock API response
            mock_info = InstrumentPrecision(
                symbol="1000PEPEUSDT",
                price_precision=7,
                quantity_precision=0,
                tick_size="0.0000001",
                step_size="0.001",
                min_price="0.0000001",
                max_price="200",
                min_qty="100",
                max_qty="10000000",
            )

            # First call: fetch from API
            with patch.object(cache, "fetch_exchange_info", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = {"1000PEPEUSDT": mock_info}
                result1 = cache.get_or_fetch_exchange_info()

            assert "1000PEPEUSDT" in result1
            mock_fetch.assert_called_once()

            # Second call: should use cache
            cache2 = InstrumentCache(tmpdir)  # New instance, same cache file
            result2 = cache2.get_or_fetch_exchange_info()

            assert "1000PEPEUSDT" in result2
            # fetch_exchange_info should not be called again
