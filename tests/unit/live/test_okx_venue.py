"""Unit tests for OKX venue config builder."""

import os
from unittest.mock import patch

import pytest

from nautilus_quants.live.config import VenueConfig
from nautilus_quants.live.exceptions import LiveConfigError
from nautilus_quants.live.venues.okx import (
    build_okx_data_config,
    build_okx_exec_config,
)


@pytest.fixture()
def okx_env():
    """Provide OKX environment variables for testing."""
    env = {
        "OKX_API_KEY": "test-key",
        "OKX_API_SECRET": "test-secret",
        "OKX_API_PASSPHRASE": "test-pass",
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture()
def okx_venue() -> VenueConfig:
    return VenueConfig(
        name="OKX",
        instrument_type="SWAP",
        contract_type="LINEAR",
        margin_mode="CROSS",
        is_demo=True,
    )


class TestBuildOkxDataConfig:
    """Tests for build_okx_data_config."""

    def test_builds_config(self, okx_env, okx_venue) -> None:
        config = build_okx_data_config(okx_venue)
        assert config.api_key == "test-key"
        assert config.api_secret == "test-secret"
        assert config.api_passphrase == "test-pass"
        assert config.is_demo is True

    def test_instrument_ids_loaded(self, okx_env, okx_venue) -> None:
        config = build_okx_data_config(
            okx_venue,
            instrument_ids=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        )
        assert config.instrument_provider.load_ids is not None
        assert "BTC-USDT-SWAP.OKX" in config.instrument_provider.load_ids

    def test_suffixed_instrument_ids_loaded_without_double_suffix(
        self,
        okx_env,
        okx_venue,
    ) -> None:
        config = build_okx_data_config(
            okx_venue,
            instrument_ids=["BTC-USDT-SWAP.OKX"],
        )
        assert config.instrument_provider.load_ids == frozenset({"BTC-USDT-SWAP.OKX"})

    def test_load_all_when_no_ids(self, okx_env, okx_venue) -> None:
        config = build_okx_data_config(okx_venue)
        assert config.instrument_provider.load_all is True

    def test_missing_api_key_raises(self, okx_venue) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LiveConfigError, match="OKX_API_KEY"):
                build_okx_data_config(okx_venue)

    def test_invalid_instrument_type_raises(self, okx_env) -> None:
        venue = VenueConfig(name="OKX", instrument_type="INVALID")
        with pytest.raises(LiveConfigError, match="Unsupported OKX instrument_type"):
            build_okx_data_config(venue)


class TestBuildOkxExecConfig:
    """Tests for build_okx_exec_config."""

    def test_builds_config(self, okx_env, okx_venue) -> None:
        config = build_okx_exec_config(okx_venue)
        assert config.api_key == "test-key"
        assert config.is_demo is True

    def test_margin_mode_set(self, okx_env, okx_venue) -> None:
        config = build_okx_exec_config(okx_venue)
        from nautilus_trader.core.nautilus_pyo3.okx import OKXMarginMode
        assert config.margin_mode == OKXMarginMode.CROSS

    def test_isolated_margin(self, okx_env) -> None:
        venue = VenueConfig(name="OKX", margin_mode="ISOLATED")
        config = build_okx_exec_config(venue)
        from nautilus_trader.core.nautilus_pyo3.okx import OKXMarginMode
        assert config.margin_mode == OKXMarginMode.ISOLATED

    def test_exec_suffixed_instrument_ids_loaded_without_double_suffix(
        self,
        okx_env,
        okx_venue,
    ) -> None:
        config = build_okx_exec_config(
            okx_venue,
            instrument_ids=["ETH-USDT-SWAP.OKX"],
        )
        assert config.instrument_provider.load_ids == frozenset({"ETH-USDT-SWAP.OKX"})

    def test_use_fills_channel(self, okx_env) -> None:
        venue = VenueConfig(name="OKX", use_fills_channel=True)
        config = build_okx_exec_config(venue)
        assert config.use_fills_channel is True

    def test_invalid_margin_mode_raises(self, okx_env) -> None:
        venue = VenueConfig(name="OKX", margin_mode="INVALID")
        with pytest.raises(LiveConfigError, match="Unsupported OKX margin_mode"):
            build_okx_exec_config(venue)
