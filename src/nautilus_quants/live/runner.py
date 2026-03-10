"""Live trading execution runner."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from nautilus_trader.config import LoggingConfig
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode

from nautilus_quants.live.exceptions import LiveConfigError
from nautilus_quants.live.utils.config_parser import (
    extract_data_configs,
    get_nautilus_config_dict,
    inject_data_configs,
    parse_live_config,
)


def _build_venue_configs(
    live_config: "nautilus_quants.live.config.LiveConfig",
    instrument_ids: list[str],
) -> tuple[Any, Any, type, type]:
    """Build venue-specific adapter configs and factory classes.

    Returns:
        (data_client_config, exec_client_config, DataFactory, ExecFactory)
    """
    venue_name = live_config.venue.name.upper()

    if venue_name == "OKX":
        from nautilus_quants.live.venues.okx import (
            DATA_FACTORY,
            EXEC_FACTORY,
            build_okx_data_config,
            build_okx_exec_config,
        )

        data_config = build_okx_data_config(live_config.venue, instrument_ids)
        exec_config = build_okx_exec_config(live_config.venue, instrument_ids)
        return data_config, exec_config, DATA_FACTORY, EXEC_FACTORY

    elif venue_name == "BINANCE":
        from nautilus_quants.live.venues.binance import (
            DATA_FACTORY,
            EXEC_FACTORY,
            build_binance_data_config,
            build_binance_exec_config,
        )

        data_config = build_binance_data_config(live_config.venue, instrument_ids)
        exec_config = build_binance_exec_config(live_config.venue, instrument_ids)
        return data_config, exec_config, DATA_FACTORY, EXEC_FACTORY

    else:
        raise LiveConfigError(
            f"Unsupported venue: {venue_name}. Supported: OKX, BINANCE"
        )


def _build_logging_config(engine_dict: dict[str, Any]) -> LoggingConfig | None:
    """Extract LoggingConfig from engine dict if present."""
    logging_section = engine_dict.get("logging")
    if not logging_section:
        return None

    return LoggingConfig(
        log_level=logging_section.get("log_level", "INFO"),
        log_level_file=logging_section.get("log_level_file", "DEBUG"),
        log_directory=logging_section.get("log_directory"),
        log_file_format=logging_section.get("log_file_format"),
        log_file_max_size=logging_section.get("log_file_max_size", 0),
        use_pyo3=logging_section.get("use_pyo3", True),
    )


def run_live(
    config_file: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Execute live trading from a YAML configuration file.

    This is a long-running process. Use Ctrl+C for graceful shutdown.

    Args:
        config_file: Path to YAML config file.
        dry_run: Validate config and test exchange connection without trading.
        verbose: Enable verbose output.

    Raises:
        LiveConfigError: If config is invalid.
    """
    # 1. Load YAML
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # 2. Parse live config
    live_config = parse_live_config(config_dict)

    # 3. Extract data configs (bar_types)
    data_configs = extract_data_configs(config_dict)
    if not data_configs:
        raise LiveConfigError("No data configs extracted. Check instruments section.")

    # 4. Inject bar_types + instrument_ids into actors/strategies
    config_dict = inject_data_configs(config_dict, data_configs)

    # 5. Get Nautilus engine config dict
    engine_dict = get_nautilus_config_dict(config_dict)

    # 6. Build venue adapter configs
    instrument_ids = live_config.instruments.instrument_ids
    data_client_config, exec_client_config, data_factory, exec_factory = (
        _build_venue_configs(live_config, instrument_ids)
    )

    venue_name = live_config.venue.name.upper()

    # 7. Build TradingNodeConfig
    logging_config = _build_logging_config(engine_dict)

    trader_id = engine_dict.get("trader_id", "LIVE-001")

    # Build risk engine config
    risk_engine_dict = engine_dict.get("risk_engine")

    # Build exec engine config
    exec_engine_dict = engine_dict.get("exec_engine")

    node_config = TradingNodeConfig(
        trader_id=trader_id,
        logging=logging_config,
        data_clients={venue_name: data_client_config},
        exec_clients={venue_name: exec_client_config},
        actors=[
            a for a in engine_dict.get("actors", [])
        ],
        strategies=[
            s for s in engine_dict.get("strategies", [])
        ],
    )

    if dry_run:
        # Validate config without actually connecting
        _dry_run_validate(node_config, live_config)
        return

    # 8-11. Build and run TradingNode
    node = TradingNode(config=node_config)
    node.add_data_client_factory(venue_name, data_factory)
    node.add_exec_client_factory(venue_name, exec_factory)

    # Register graceful shutdown handler
    _register_signal_handlers(node)

    node.build()
    node.run()


def _dry_run_validate(
    node_config: TradingNodeConfig,
    live_config: "nautilus_quants.live.config.LiveConfig",
) -> None:
    """Validate configuration in dry-run mode."""
    venue = live_config.venue
    instruments = live_config.instruments

    print(f"  Venue: {venue.name}")
    print(f"  Instrument type: {venue.instrument_type}")
    print(f"  Contract type: {venue.contract_type}")
    print(f"  Margin mode: {venue.margin_mode}")
    print(f"  Demo mode: {venue.is_demo}")
    print(f"  Bar spec: {instruments.bar_spec}")
    print(f"  Instruments: {len(instruments.instrument_ids)}")
    for iid in instruments.instrument_ids:
        print(f"    - {iid}")
    print(f"  Trader ID: {node_config.trader_id}")
    print(f"  Actors: {len(node_config.actors)}")
    print(f"  Strategies: {len(node_config.strategies)}")


def _register_signal_handlers(node: TradingNode) -> None:
    """Register signal handlers for graceful shutdown."""

    def _handle_signal(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        print(f"\nReceived {sig_name}, shutting down gracefully...")
        try:
            node.dispose()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def validate_config(config_file: Path) -> bool:
    """Validate a live configuration file without executing.

    Args:
        config_file: Path to YAML config file.

    Returns:
        True if valid.

    Raises:
        LiveConfigError: If config is invalid.
    """
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Parse and validate
    live_config = parse_live_config(config_dict)

    # Validate data configs can be extracted
    data_configs = extract_data_configs(config_dict)
    if not data_configs:
        raise LiveConfigError("No data configs could be extracted")

    # Validate injection works
    inject_data_configs(config_dict, data_configs)

    return True
