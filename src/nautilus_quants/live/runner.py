"""Live trading execution runner."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from nautilus_trader.common.config import ImportableActorConfig
from nautilus_trader.config import CacheConfig, DatabaseConfig
from nautilus_trader.config import ImportableExecAlgorithmConfig, LoggingConfig
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.trading.config import ImportableControllerConfig, ImportableStrategyConfig

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


def _build_cache_config(engine_dict: dict[str, Any]) -> CacheConfig | None:
    """Extract CacheConfig from engine dict if present."""
    cache_section = engine_dict.get("cache")
    if not cache_section:
        return None

    # Handle nested database config
    cache_section = dict(cache_section)  # Make a copy to avoid mutating original
    database_config = None
    if "database" in cache_section:
        db_section = cache_section.pop("database")
        database_config = DatabaseConfig(**db_section)

    return CacheConfig(database=database_config, **cache_section)



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
    cache_config = _build_cache_config(engine_dict)

    trader_id = engine_dict.get("trader_id", "LIVE-001")

    # Build risk engine config (live requires LiveRiskEngineConfig)
    risk_engine_config = None
    risk_engine_dict = engine_dict.get("risk_engine")
    if risk_engine_dict:
        from nautilus_trader.config import LiveRiskEngineConfig

        risk_engine_config = LiveRiskEngineConfig(**risk_engine_dict)

    # Build exec engine config (live requires LiveExecEngineConfig)
    exec_engine_config = None
    exec_engine_dict = engine_dict.get("exec_engine")
    if exec_engine_dict:
        from nautilus_trader.config import LiveExecEngineConfig

        exec_engine_config = LiveExecEngineConfig(**exec_engine_dict)

    # Build cache config (optional, supports Redis persistence)
    cache_config = None
    cache_dict = engine_dict.get("cache")
    if cache_dict:
        from nautilus_trader.config import CacheConfig, DatabaseConfig

        cache_payload = dict(cache_dict)
        database_dict = cache_payload.get("database")
        if database_dict:
            cache_payload["database"] = DatabaseConfig(**database_dict)
        cache_config = CacheConfig(**cache_payload)

    # Build exec algorithms from config
    exec_algorithms = [
        ImportableExecAlgorithmConfig(**ea)
        for ea in engine_dict.get("exec_algorithms", [])
    ]

    # Build controller config if present
    controller_config = None
    controller_dict = engine_dict.get("controller")
    if controller_dict:
        controller_config = ImportableControllerConfig(**controller_dict)

    # Build node config kwargs, only include non-None engine configs
    node_kwargs: dict[str, Any] = {
        "trader_id": trader_id,
        "logging": logging_config,
        "data_clients": {venue_name: data_client_config},
        "exec_clients": {venue_name: exec_client_config},
        "actors": [ImportableActorConfig(**a) for a in engine_dict.get("actors", [])],
        "strategies": [ImportableStrategyConfig(**s) for s in engine_dict.get("strategies", [])],
        "exec_algorithms": exec_algorithms,
    }
    if cache_config is not None:
        node_kwargs["cache"] = cache_config
    if risk_engine_config is not None:
        node_kwargs["risk_engine"] = risk_engine_config
    if exec_engine_config is not None:
        node_kwargs["exec_engine"] = exec_engine_config
    if controller_config is not None:
        node_kwargs["controller"] = controller_config

    node_config = TradingNodeConfig(**node_kwargs)

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
    """Register signal handlers for graceful shutdown.

    Sequence: SIGTERM → node.stop() (trader.stop → on_stop → close positions
    → await residuals → disconnect) → node.dispose() → exit.
    """
    _shutdown_in_progress = False

    def _handle_signal(signum: int, frame: Any) -> None:
        nonlocal _shutdown_in_progress
        if _shutdown_in_progress:
            print("\nForce exit (second signal).")
            sys.exit(1)

        _shutdown_in_progress = True
        sig_name = signal.Signals(signum).name
        print(f"\nReceived {sig_name}, shutting down gracefully...")
        try:
            node.stop()     # trader.stop → on_stop → close_all_positions → await residuals
            node.dispose()  # cleanup resources, cancel tasks, close loop
        except Exception as e:
            print(f"Error during shutdown: {e}")
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
