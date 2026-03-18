"""Unit tests for live config dataclasses."""

import pytest

from nautilus_quants.live.config import InstrumentsConfig, LiveConfig, VenueConfig


class TestVenueConfig:
    """Tests for VenueConfig dataclass."""

    def test_create_okx_venue(self) -> None:
        venue = VenueConfig(
            name="OKX",
            instrument_type="SWAP",
            contract_type="LINEAR",
            margin_mode="CROSS",
            is_demo=True,
        )
        assert venue.name == "OKX"
        assert venue.instrument_type == "SWAP"
        assert venue.contract_type == "LINEAR"
        assert venue.margin_mode == "CROSS"
        assert venue.is_demo is True

    def test_create_binance_venue(self) -> None:
        venue = VenueConfig(
            name="BINANCE",
            instrument_type="SWAP",
            contract_type="LINEAR",
            margin_mode="CROSS",
            is_demo=False,
        )
        assert venue.name == "BINANCE"
        assert venue.is_demo is False

    def test_defaults(self) -> None:
        venue = VenueConfig(name="OKX")
        assert venue.instrument_type == "SWAP"
        assert venue.contract_type == "LINEAR"
        assert venue.margin_mode == "CROSS"
        assert venue.is_demo is True
        assert venue.http_timeout_secs == 10
        assert venue.use_fills_channel is False

    def test_frozen(self) -> None:
        venue = VenueConfig(name="OKX")
        with pytest.raises(AttributeError):
            venue.name = "BINANCE"  # type: ignore[misc]


class TestInstrumentsConfig:
    """Tests for InstrumentsConfig dataclass."""

    def test_create(self) -> None:
        instruments = InstrumentsConfig(
            bar_spec="1h",
            instrument_ids=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        )
        assert instruments.bar_spec == "1h"
        assert len(instruments.instrument_ids) == 2

    def test_defaults(self) -> None:
        instruments = InstrumentsConfig(bar_spec="4h")
        assert instruments.instrument_ids == []

    def test_frozen(self) -> None:
        instruments = InstrumentsConfig(bar_spec="1h")
        with pytest.raises(AttributeError):
            instruments.bar_spec = "4h"  # type: ignore[misc]


class TestLiveConfig:
    """Tests for LiveConfig dataclass."""

    def test_create(self) -> None:
        venue = VenueConfig(name="OKX")
        instruments = InstrumentsConfig(
            bar_spec="1h",
            instrument_ids=["BTC-USDT-SWAP"],
        )
        config = LiveConfig(venue=venue, instruments=instruments)
        assert config.venue.name == "OKX"
        assert len(config.instruments.instrument_ids) == 1

    def test_frozen(self) -> None:
        venue = VenueConfig(name="OKX")
        instruments = InstrumentsConfig(bar_spec="1h")
        config = LiveConfig(venue=venue, instruments=instruments)
        with pytest.raises(AttributeError):
            config.venue = VenueConfig(name="BINANCE")  # type: ignore[misc]
