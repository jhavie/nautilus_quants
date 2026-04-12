# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for Santiment slug mapping module."""
import pytest

from nautilus_quants.data.santiment.slug_map import (
    AVAILABLE,
    FR_AVAILABLE,
    OI_AVAILABLE,
    SLUG_MAP,
    instrument_to_slug,
    instrument_to_ticker,
    ticker_to_slug,
)


class TestSlugMap:
    def test_slug_map_not_empty(self) -> None:
        assert len(SLUG_MAP) >= 90

    def test_available_is_intersection(self) -> None:
        assert AVAILABLE == FR_AVAILABLE & OI_AVAILABLE

    def test_available_size(self) -> None:
        assert len(AVAILABLE) >= 80

    def test_major_coins_in_available(self) -> None:
        for ticker in ("BTC", "ETH", "SOL", "DOGE", "LINK"):
            assert ticker in AVAILABLE


class TestTickerToSlug:
    def test_btc(self) -> None:
        assert ticker_to_slug("BTC") == "bitcoin"

    def test_eth(self) -> None:
        assert ticker_to_slug("ETH") == "ethereum"

    def test_case_insensitive(self) -> None:
        assert ticker_to_slug("btc") == "bitcoin"

    def test_unknown_returns_none(self) -> None:
        assert ticker_to_slug("NONEXISTENT") is None


class TestInstrumentToSlug:
    def test_standard(self) -> None:
        assert instrument_to_slug("BTCUSDT.BINANCE") == "bitcoin"

    def test_1000_prefix(self) -> None:
        assert instrument_to_slug("1000SHIBUSDT.BINANCE") == "shiba-inu"

    def test_unknown_instrument(self) -> None:
        assert instrument_to_slug("FOOBARUSDT.BINANCE") is None


class TestInstrumentToTicker:
    def test_standard(self) -> None:
        assert instrument_to_ticker("BTCUSDT.BINANCE") == "BTC"

    def test_1000_prefix(self) -> None:
        assert instrument_to_ticker("1000SHIBUSDT.BINANCE") == "SHIB"
