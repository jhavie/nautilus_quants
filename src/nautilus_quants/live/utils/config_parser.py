"""Configuration parsing utilities for the live trading module."""

from __future__ import annotations

import copy
import os
from typing import Any

from nautilus_quants.utils.bar_spec import format_bar_spec
from nautilus_quants.live.config import InstrumentsConfig, LiveConfig, VenueConfig
from nautilus_quants.live.exceptions import LiveConfigError


def parse_live_config(config_dict: dict[str, Any]) -> LiveConfig:
    """Parse venue, instruments from YAML dict.

    Args:
        config_dict: Full YAML config dictionary.

    Returns:
        LiveConfig dataclass.

    Raises:
        LiveConfigError: If required fields are missing or invalid.
    """
    # Parse venue
    venue_section = config_dict.get("venue")
    if not venue_section:
        raise LiveConfigError("Missing 'venue' section in config")

    venue_name = venue_section.get("name")
    if not venue_name:
        raise LiveConfigError("Missing 'venue.name' in config")

    venue = VenueConfig(
        name=venue_name.upper(),
        instrument_type=venue_section.get("instrument_type", "SWAP"),
        contract_type=venue_section.get("contract_type", "LINEAR"),
        margin_mode=venue_section.get("margin_mode", "CROSS"),
        is_demo=venue_section.get("is_demo", True),
        http_timeout_secs=venue_section.get("http_timeout_secs", 10),
        use_fills_channel=venue_section.get("use_fills_channel", False),
    )

    # Parse instruments
    instruments_section = config_dict.get("instruments")
    if not instruments_section:
        raise LiveConfigError("Missing 'instruments' section in config")

    instrument_ids = instruments_section.get("instrument_ids", [])
    if not instrument_ids:
        raise LiveConfigError("Missing or empty 'instruments.instrument_ids' in config")

    bar_spec = instruments_section.get("bar_spec", "")
    if not bar_spec:
        raise LiveConfigError("Missing 'instruments.bar_spec' in config")

    instruments = InstrumentsConfig(
        bar_spec=bar_spec,
        instrument_ids=list(instrument_ids),
    )

    return LiveConfig(venue=venue, instruments=instruments)


def extract_data_configs(config_dict: dict[str, Any]) -> list[dict[str, str]]:
    """Extract bar_type list from instruments section.

    Unlike the backtest version, live mode does not need data_cls or catalog_path.
    Simply generates bar_types from instrument_ids + bar_spec + venue.

    Args:
        config_dict: Full YAML config dictionary.

    Returns:
        List of dicts with instrument_id, bar_type, bar_spec for each instrument.
    """
    instruments_section = config_dict.get("instruments", {})
    venue_section = config_dict.get("venue", {})

    instrument_ids = instruments_section.get("instrument_ids", [])
    bar_spec = instruments_section.get("bar_spec", "")
    venue_name = venue_section.get("name", "").upper()

    if not instrument_ids or not bar_spec or not venue_name:
        return []

    try:
        native_spec = format_bar_spec(bar_spec, include_source=False)
    except ValueError:
        native_spec = bar_spec

    result = []
    for inst_id in instrument_ids:
        # Support both "BTC-USDT-SWAP" and "BTC-USDT-SWAP.OKX" formats
        if "." in inst_id:
            # Already has venue suffix (e.g. "BTC-USDT-SWAP.OKX")
            nautilus_id = inst_id
        else:
            nautilus_id = f"{inst_id}.{venue_name}"
        bar_type = f"{nautilus_id}-{native_spec}-EXTERNAL"
        result.append(
            {
                "instrument_id": nautilus_id,
                "bar_spec": native_spec,
                "bar_type": bar_type,
            }
        )

    return result


def inject_data_configs(
    config_dict: dict[str, Any],
    data_configs: list[dict[str, str]],
) -> dict[str, Any]:
    """Inject bar_types and instrument_ids into actors/strategies config.

    Mirrors backtest inject_data_configs() naming, but simplified for live mode:
    no data_cls filtering needed since live receives standard Bar objects.

    Args:
        config_dict: Full YAML config dictionary.
        data_configs: List of extracted data configs (from extract_data_configs).

    Returns:
        Modified config dict with injected bar_type info.
    """
    if not data_configs:
        return config_dict

    config_dict = copy.deepcopy(config_dict)
    engine = config_dict.get("engine", {})

    all_bar_types = [dc["bar_type"] for dc in data_configs]
    all_instrument_ids = [dc["instrument_id"] for dc in data_configs]

    # Inject into actors
    for actor in engine.get("actors", []):
        actor_config = actor.get("config", {})
        actor_path = actor.get("actor_path", "")
        should_inject_bar_types = (
            "bar_types" in actor_config
            or actor_path.endswith("factor_engine:FactorEngineActor")
            or actor_path.endswith("risk_model:RiskModelActor")
        )
        if should_inject_bar_types and (
            "bar_types" not in actor_config or not actor_config["bar_types"]
        ):
            actor_config["bar_types"] = all_bar_types
        actor["config"] = actor_config

    # Inject into strategies
    for strategy in engine.get("strategies", []):
        strategy_config = strategy.get("config", {})
        if "bar_types" not in strategy_config or not strategy_config["bar_types"]:
            strategy_config["bar_types"] = all_bar_types
        if not strategy_config.get("instrument_ids"):
            strategy_config["instrument_ids"] = all_instrument_ids
        strategy["config"] = strategy_config

    return config_dict


def get_nautilus_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip non-Nautilus keys, returning only engine configuration.

    Removes venue, instruments, and other live-specific keys
    that are not part of TradingNodeConfig.

    Args:
        config_dict: Full YAML config dictionary.

    Returns:
        Dict with only Nautilus-compatible engine keys.
    """
    engine = config_dict.get("engine", {})
    return dict(engine)


def _get_env_credential(key: str, required: bool = True) -> str | None:
    """Read a credential from environment variables.

    Args:
        key: Environment variable name.
        required: If True, raise LiveConfigError when missing.

    Returns:
        Environment variable value.

    Raises:
        LiveConfigError: If required and not set.
    """
    value = os.environ.get(key)
    if required and not value:
        raise LiveConfigError(
            f"Missing required environment variable: {key}. "
            f"Set it before running live trading."
        )
    return value
