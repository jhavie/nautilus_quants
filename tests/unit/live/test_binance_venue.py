"""Unit tests for Binance venue config builder."""

import os
from unittest.mock import patch

import pytest

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType

from nautilus_quants.live.config import VenueConfig
from nautilus_quants.live.exceptions import LiveConfigError
from nautilus_quants.live.venues.binance import (
    build_binance_data_config,
    build_binance_exec_config,
)


@pytest.fixture()
def binance_env():
    """Provide Binance environment variables for testing."""
    env = {
        "BINANCE_API_KEY": "test-binance-key",
        "BINANCE_API_SECRET": "test-binance-secret",
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture()
def binance_venue() -> VenueConfig:
    return VenueConfig(
        name="BINANCE",
        instrument_type="SWAP",
        contract_type="LINEAR",
        margin_mode="CROSS",
        is_demo=True,
    )


class TestBuildBinanceDataConfig:
    """Tests for build_binance_data_config."""

    def test_builds_config(self, binance_env, binance_venue) -> None:
        config = build_binance_data_config(binance_venue)
        assert config.api_key == "test-binance-key"
        assert config.api_secret == "test-binance-secret"
        assert config.testnet is True

    def test_account_type_usdt_futures(self, binance_env, binance_venue) -> None:
        config = build_binance_data_config(binance_venue)
        assert config.account_type == BinanceAccountType.USDT_FUTURES

    def test_account_type_coin_futures(self, binance_env) -> None:
        venue = VenueConfig(
            name="BINANCE",
            instrument_type="SWAP",
            contract_type="INVERSE",
        )
        config = build_binance_data_config(venue)
        assert config.account_type == BinanceAccountType.COIN_FUTURES

    def test_account_type_spot(self, binance_env) -> None:
        venue = VenueConfig(
            name="BINANCE",
            instrument_type="SPOT",
        )
        config = build_binance_data_config(venue)
        assert config.account_type == BinanceAccountType.SPOT

    def test_instrument_ids_loaded(self, binance_env, binance_venue) -> None:
        config = build_binance_data_config(
            binance_venue,
            instrument_ids=["BTCUSDT", "ETHUSDT"],
        )
        assert config.instrument_provider.load_ids is not None
        assert "BTCUSDT.BINANCE" in config.instrument_provider.load_ids

    def test_missing_api_key_raises(self, binance_venue) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LiveConfigError, match="BINANCE_API_KEY"):
                build_binance_data_config(binance_venue)


class TestBuildBinanceExecConfig:
    """Tests for build_binance_exec_config."""

    def test_builds_config(self, binance_env, binance_venue) -> None:
        config = build_binance_exec_config(binance_venue)
        assert config.api_key == "test-binance-key"
        assert config.testnet is True
        assert config.account_type == BinanceAccountType.USDT_FUTURES

    def test_load_all_when_no_ids(self, binance_env, binance_venue) -> None:
        config = build_binance_exec_config(binance_venue)
        assert config.instrument_provider.load_all is True
