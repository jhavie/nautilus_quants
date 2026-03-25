"""Unit tests for live config parser."""

import pytest

from nautilus_quants.live.config import InstrumentsConfig, VenueConfig
from nautilus_quants.live.exceptions import LiveConfigError
from nautilus_quants.live.utils.config_parser import (
    extract_data_configs,
    get_nautilus_config_dict,
    inject_data_configs,
    parse_live_config,
)


def _make_config_dict(
    venue_name: str = "OKX",
    instrument_ids: list[str] | None = None,
    bar_spec: str = "1h",
) -> dict:
    """Helper to build a minimal config dict."""
    if instrument_ids is None:
        instrument_ids = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    return {
        "venue": {
            "name": venue_name,
            "instrument_type": "SWAP",
            "contract_type": "LINEAR",
            "margin_mode": "CROSS",
            "is_demo": True,
        },
        "instruments": {
            "bar_spec": bar_spec,
            "instrument_ids": instrument_ids,
        },
        "engine": {
            "trader_id": "TEST-001",
            "logging": {"log_level": "INFO"},
            "actors": [
                {
                    "actor_path": "nautilus_quants.actors.factor_engine:FactorEngineActor",
                    "config_path": "nautilus_quants.actors.factor_engine:FactorEngineActorConfig",
                    "config": {
                        "factor_config_path": "config/fmz/factors.yaml",
                        "max_history": 500,
                        "publish_signals": True,
                    },
                }
            ],
            "strategies": [
                {
                    "strategy_path": "nautilus_quants.strategies.fmz.strategy:FMZFactorStrategy",
                    "config_path": "nautilus_quants.strategies.fmz.strategy:FMZFactorStrategyConfig",
                    "config": {
                        "instrument_ids": [],
                        "n_long": 10,
                        "n_short": 10,
                    },
                }
            ],
        },
    }


class TestParseLiveConfig:
    """Tests for parse_live_config."""

    def test_parse_okx(self) -> None:
        config_dict = _make_config_dict("OKX")
        result = parse_live_config(config_dict)

        assert result.venue.name == "OKX"
        assert result.venue.instrument_type == "SWAP"
        assert result.instruments.bar_spec == "1h"
        assert len(result.instruments.instrument_ids) == 2

    def test_parse_binance(self) -> None:
        config_dict = _make_config_dict("binance", ["BTCUSDT", "ETHUSDT"])
        result = parse_live_config(config_dict)

        assert result.venue.name == "BINANCE"
        assert len(result.instruments.instrument_ids) == 2

    def test_missing_venue_raises(self) -> None:
        with pytest.raises(LiveConfigError, match="Missing 'venue' section"):
            parse_live_config({"instruments": {"bar_spec": "1h", "instrument_ids": ["X"]}})

    def test_missing_venue_name_raises(self) -> None:
        with pytest.raises(LiveConfigError, match="Missing 'venue.name'"):
            parse_live_config({
                "venue": {"instrument_type": "SWAP"},
                "instruments": {"bar_spec": "1h", "instrument_ids": ["X"]},
            })

    def test_missing_instruments_raises(self) -> None:
        with pytest.raises(LiveConfigError, match="Missing 'instruments' section"):
            parse_live_config({"venue": {"name": "OKX"}})

    def test_empty_instrument_ids_raises(self) -> None:
        with pytest.raises(LiveConfigError, match="Missing or empty"):
            parse_live_config({
                "venue": {"name": "OKX"},
                "instruments": {"bar_spec": "1h", "instrument_ids": []},
            })

    def test_missing_bar_spec_raises(self) -> None:
        with pytest.raises(LiveConfigError, match="Missing 'instruments.bar_spec'"):
            parse_live_config({
                "venue": {"name": "OKX"},
                "instruments": {"instrument_ids": ["BTC-USDT-SWAP"]},
            })


class TestExtractDataConfigs:
    """Tests for extract_data_configs."""

    def test_extract_okx(self) -> None:
        config_dict = _make_config_dict("OKX", ["BTC-USDT-SWAP", "ETH-USDT-SWAP"], "1h")
        result = extract_data_configs(config_dict)

        assert len(result) == 2
        assert result[0]["instrument_id"] == "BTC-USDT-SWAP.OKX"
        assert result[0]["bar_type"] == "BTC-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"
        assert result[0]["bar_spec"] == "1-HOUR-LAST"
        assert result[1]["instrument_id"] == "ETH-USDT-SWAP.OKX"
        assert result[1]["bar_type"] == "ETH-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"

    def test_extract_okx_with_suffixed_ids(self) -> None:
        config_dict = _make_config_dict("OKX", ["BTC-USDT-SWAP.OKX"], "1h")
        result = extract_data_configs(config_dict)

        assert len(result) == 1
        assert result[0]["instrument_id"] == "BTC-USDT-SWAP.OKX"
        assert result[0]["bar_type"] == "BTC-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"

    def test_extract_binance(self) -> None:
        config_dict = _make_config_dict("BINANCE", ["BTCUSDT", "ETHUSDT"], "4h")
        result = extract_data_configs(config_dict)

        assert len(result) == 2
        assert result[0]["bar_type"] == "BTCUSDT.BINANCE-4-HOUR-LAST-EXTERNAL"

    def test_extract_empty_instruments(self) -> None:
        config_dict = _make_config_dict()
        config_dict["instruments"]["instrument_ids"] = []
        result = extract_data_configs(config_dict)
        assert result == []

    def test_extract_missing_venue(self) -> None:
        config_dict = {"instruments": {"bar_spec": "1h", "instrument_ids": ["X"]}}
        result = extract_data_configs(config_dict)
        assert result == []


class TestInjectDataConfigs:
    """Tests for inject_data_configs."""

    def test_inject_bar_types_into_actors(self) -> None:
        config_dict = _make_config_dict()
        data_configs = extract_data_configs(config_dict)
        result = inject_data_configs(config_dict, data_configs)

        actor_config = result["engine"]["actors"][0]["config"]
        assert len(actor_config["bar_types"]) == 2
        assert "BTC-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL" in actor_config["bar_types"]

    def test_inject_bar_types_into_strategies(self) -> None:
        config_dict = _make_config_dict()
        data_configs = extract_data_configs(config_dict)
        result = inject_data_configs(config_dict, data_configs)

        strategy_config = result["engine"]["strategies"][0]["config"]
        assert len(strategy_config["bar_types"]) == 2
        assert len(strategy_config["instrument_ids"]) == 2
        assert "BTC-USDT-SWAP.OKX" in strategy_config["instrument_ids"]

    def test_inject_preserves_existing_bar_types(self) -> None:
        config_dict = _make_config_dict()
        config_dict["engine"]["actors"][0]["config"]["bar_types"] = ["EXISTING.BAR-TYPE"]
        data_configs = extract_data_configs(config_dict)
        result = inject_data_configs(config_dict, data_configs)

        actor_config = result["engine"]["actors"][0]["config"]
        assert actor_config["bar_types"] == ["EXISTING.BAR-TYPE"]

    def test_inject_skips_non_factor_actor_without_bar_types_field(self) -> None:
        config_dict = _make_config_dict()
        config_dict["engine"]["actors"].append(
            {
                "actor_path": "nautilus_quants.strategies.cs.decision_engine:DecisionEngineActor",
                "config_path": "nautilus_quants.strategies.cs.config:DecisionEngineActorConfig",
                "config": {"n_long": 8, "n_short": 8},
            }
        )
        data_configs = extract_data_configs(config_dict)
        result = inject_data_configs(config_dict, data_configs)

        decision_config = result["engine"]["actors"][1]["config"]
        assert "bar_types" not in decision_config

    def test_inject_preserves_existing_instrument_ids(self) -> None:
        config_dict = _make_config_dict()
        config_dict["engine"]["strategies"][0]["config"]["instrument_ids"] = ["CUSTOM-ID"]
        data_configs = extract_data_configs(config_dict)
        result = inject_data_configs(config_dict, data_configs)

        strategy_config = result["engine"]["strategies"][0]["config"]
        assert strategy_config["instrument_ids"] == ["CUSTOM-ID"]

    def test_inject_does_not_mutate_original(self) -> None:
        config_dict = _make_config_dict()
        data_configs = extract_data_configs(config_dict)
        inject_data_configs(config_dict, data_configs)

        # Original should be unchanged
        actor_config = config_dict["engine"]["actors"][0]["config"]
        assert "bar_types" not in actor_config

    def test_inject_empty_data_configs(self) -> None:
        config_dict = _make_config_dict()
        result = inject_data_configs(config_dict, [])
        assert result is config_dict  # No copy when empty


class TestGetNautilusConfigDict:
    """Tests for get_nautilus_config_dict."""

    def test_extracts_engine_only(self) -> None:
        config_dict = _make_config_dict()
        result = get_nautilus_config_dict(config_dict)

        assert "trader_id" in result
        assert "actors" in result
        assert "strategies" in result
        assert "venue" not in result
        assert "instruments" not in result

    def test_empty_engine(self) -> None:
        result = get_nautilus_config_dict({"venue": {"name": "OKX"}})
        assert result == {}
