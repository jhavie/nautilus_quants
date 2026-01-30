"""Binance instrument metadata cache with API fetch and local persistence."""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class InstrumentPrecision:
    """Precision information for a trading pair."""

    symbol: str
    price_precision: int
    quantity_precision: int
    tick_size: str
    step_size: str
    min_price: str
    max_price: str
    min_qty: str
    max_qty: str


# Fallback patterns for inferring precision from symbol name
SYMBOL_PATTERNS: list[tuple[str, dict[str, Any]]] = [
    (r"^1000PEPE", {"price_precision": 7, "tick_size": "0.0000001", "quantity_precision": 0, "step_size": "1"}),
    (r"^1000SHIB", {"price_precision": 8, "tick_size": "0.00000001", "quantity_precision": 0, "step_size": "1"}),
    (r"^1000BONK", {"price_precision": 7, "tick_size": "0.0000001", "quantity_precision": 0, "step_size": "1"}),
    (r"^1000FLOKI", {"price_precision": 7, "tick_size": "0.0000001", "quantity_precision": 0, "step_size": "1"}),
    (r"^BTC", {"price_precision": 2, "tick_size": "0.01", "quantity_precision": 3, "step_size": "0.001"}),
    (r"^ETH", {"price_precision": 2, "tick_size": "0.01", "quantity_precision": 3, "step_size": "0.001"}),
]

DEFAULT_PRECISION = {
    "price_precision": 4,
    "tick_size": "0.0001",
    "quantity_precision": 3,
}


class InstrumentCache:
    """Cache for Binance instrument metadata.

    Fetches from Binance API with 24h local cache.
    Falls back to heuristic inference if API fails.
    """

    CACHE_FILENAME = ".instrument_cache.json"
    CACHE_TTL_HOURS = 24
    BINANCE_FUTURES_API = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    BINANCE_SPOT_API = "https://api.binance.com/api/v3/exchangeInfo"

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache file. Defaults to current directory.
        """
        if cache_dir is None:
            cache_dir = Path.cwd()
        self.cache_path = Path(cache_dir) / self.CACHE_FILENAME
        self._exchange_info: dict[str, InstrumentPrecision] | None = None

    def _load_cache(self) -> dict[str, Any] | None:
        """Load cache from disk if it exists and is not expired."""
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Check if cache is expired
            import time

            cached_time = cache_data.get("_cached_at", 0)
            age_hours = (time.time() - cached_time) / 3600

            if age_hours > self.CACHE_TTL_HOURS:
                return None

            return cache_data.get("data")
        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def _save_cache(self, data: dict[str, Any]) -> None:
        """Save cache to disk."""
        import time

        cache_data = {
            "_cached_at": time.time(),
            "data": data,
        }

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, default=str)
        except IOError as e:
            # Log warning but don't fail
            import warnings

            warnings.warn(f"Failed to save instrument cache: {e}")

    async def fetch_exchange_info(
        self, market_type: str = "futures"
    ) -> dict[str, InstrumentPrecision]:
        """Fetch instrument info from Binance API.

        Args:
            market_type: "futures" or "spot"

        Returns:
            Dictionary mapping symbol to InstrumentPrecision

        Raises:
            aiohttp.ClientError: If API request fails
        """
        url = (
            self.BINANCE_FUTURES_API
            if market_type == "futures"
            else self.BINANCE_SPOT_API
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                data = await response.json()

        # Parse symbols
        symbols_data = data.get("symbols", [])
        exchange_info: dict[str, InstrumentPrecision] = {}

        for symbol_data in symbols_data:
            symbol = symbol_data.get("symbol", "")
            if not symbol or symbol_data.get("status") != "TRADING":
                continue

            # Extract filters
            tick_size = "0.01"
            step_size = "0.001"
            min_price = "0.01"
            max_price = "1000000"
            min_qty = "0.001"
            max_qty = "10000"

            for f in symbol_data.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick_size = f.get("tickSize", tick_size)
                    min_price = f.get("minPrice", min_price)
                    max_price = f.get("maxPrice", max_price)
                elif f.get("filterType") == "LOT_SIZE":
                    step_size = f.get("stepSize", step_size)
                    min_qty = f.get("minQty", min_qty)
                    max_qty = f.get("maxQty", max_qty)

            exchange_info[symbol] = InstrumentPrecision(
                symbol=symbol,
                price_precision=symbol_data.get("pricePrecision", 2),
                quantity_precision=symbol_data.get("quantityPrecision", 3),
                tick_size=tick_size,
                step_size=step_size,
                min_price=min_price,
                max_price=max_price,
                min_qty=min_qty,
                max_qty=max_qty,
            )

        return exchange_info

    def get_or_fetch_exchange_info(
        self,
        market_type: str = "futures",
        max_age_hours: int | None = None,
    ) -> dict[str, InstrumentPrecision]:
        """Get exchange info from cache or fetch from API.

        Args:
            market_type: "futures" or "spot"
            max_age_hours: Override default cache TTL

        Returns:
            Dictionary mapping symbol to InstrumentPrecision.
            Returns empty dict if both cache and API fail.
        """
        if max_age_hours is None:
            max_age_hours = self.CACHE_TTL_HOURS

        # Try to load from cache
        cached_data = self._load_cache()
        if cached_data:
            # Convert cached dict back to InstrumentPrecision objects
            result: dict[str, InstrumentPrecision] = {}
            for symbol, data in cached_data.items():
                if symbol.startswith("_"):  # Skip metadata keys
                    continue
                result[symbol] = InstrumentPrecision(**data)
            return result

        # Fetch from API
        try:
            exchange_info = asyncio.run(self.fetch_exchange_info(market_type))

            # Save to cache
            cache_data = {
                symbol: {
                    "symbol": info.symbol,
                    "price_precision": info.price_precision,
                    "quantity_precision": info.quantity_precision,
                    "tick_size": info.tick_size,
                    "step_size": info.step_size,
                    "min_price": info.min_price,
                    "max_price": info.max_price,
                    "min_qty": info.min_qty,
                    "max_qty": info.max_qty,
                }
                for symbol, info in exchange_info.items()
            }
            self._save_cache(cache_data)

            return exchange_info
        except Exception as e:
            # Log warning and return empty dict
            import warnings

            warnings.warn(f"Failed to fetch exchange info: {e}")
            return {}

    def get_symbol_info(self, symbol: str) -> InstrumentPrecision | None:
        """Get info for a specific symbol.

        Args:
            symbol: Trading pair symbol (e.g., "1000PEPEUSDT")

        Returns:
            InstrumentPrecision or None if not found
        """
        if self._exchange_info is None:
            self._exchange_info = self.get_or_fetch_exchange_info()

        return self._exchange_info.get(symbol)

    @staticmethod
    def infer_precision_from_symbol(symbol: str) -> InstrumentPrecision:
        """Infer precision from symbol name patterns.

        Used as fallback when API is unavailable.

        Args:
            symbol: Trading pair symbol

        Returns:
            InstrumentPrecision with inferred values
        """
        # Try to match patterns
        for pattern, defaults in SYMBOL_PATTERNS:
            if re.match(pattern, symbol):
                step_size = defaults.get("step_size", "0.001")
                qty_precision = defaults.get("quantity_precision", 3)
                return InstrumentPrecision(
                    symbol=symbol,
                    price_precision=defaults["price_precision"],
                    quantity_precision=qty_precision,
                    tick_size=defaults["tick_size"],
                    step_size=step_size,
                    min_price=defaults["tick_size"],
                    max_price="1000000",
                    min_qty=step_size,
                    max_qty="10000",
                )

        # Return default
        return InstrumentPrecision(
            symbol=symbol,
            price_precision=DEFAULT_PRECISION["price_precision"],
            quantity_precision=DEFAULT_PRECISION["quantity_precision"],
            tick_size=DEFAULT_PRECISION["tick_size"],
            step_size="0.001",
            min_price=DEFAULT_PRECISION["tick_size"],
            max_price="1000000",
            min_qty="0.001",
            max_qty="10000",
        )
