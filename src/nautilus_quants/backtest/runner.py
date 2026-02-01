"""Backtest execution runner."""

import glob
from datetime import datetime
from pathlib import Path
from typing import Any

import msgspec
import yaml

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import BacktestRunConfig

from nautilus_quants.backtest.config import ReportConfig
from nautilus_quants.backtest.exceptions import BacktestConfigError
from nautilus_quants.backtest.reports import ReportGenerator
from nautilus_quants.backtest.utils.config_parser import (
    extract_data_configs,
    get_nautilus_config_dict,
    inject_data_configs,
    inject_logging_config,
    parse_report_config,
)
from nautilus_quants.backtest.utils.reporting import create_output_directory, generate_run_id


def _load_funding_data(config_dict: dict, node: BacktestNode) -> None:
    """Load funding rate data based on configuration.

    Files are discovered at:
        {raw_data_path}/{venue}/{SYMBOL}/funding/{SYMBOL}_funding_*.csv

    Config format:
        funding:
          enabled: true
          raw_data_path: "/path/to/raw"  # optional, defaults to catalog_path/../raw
          instrument_ids: [...]  # optional, defaults to data section instrument_ids
    """
    funding_config = config_dict.get("funding", {})
    if not funding_config.get("enabled", False):
        return

    from nautilus_quants.data.transform.funding import load_funding_rates

    data_configs = config_dict.get("data", [])
    if not data_configs:
        return

    # raw_data_path: prefer funding config, else infer from catalog_path/../raw
    raw_data_path = funding_config.get("raw_data_path")
    if not raw_data_path:
        catalog_path = Path(data_configs[0].get("catalog_path", ""))
        raw_data_path = catalog_path.parent / "raw"
    raw_data_path = Path(raw_data_path)

    # instrument_ids: prefer funding config, else use from data configs
    instrument_ids = funding_config.get("instrument_ids")
    if not instrument_ids:
        instrument_ids = []
        for dc in data_configs:
            instrument_ids.extend(dc.get("instrument_ids", []))

    if not instrument_ids:
        return

    # Build engines first so we can add data
    node.build()

    # Load funding data for each instrument
    for inst_id in instrument_ids:
        # Parse instrument_id: "ETHUSDT.BINANCE" -> symbol="ETHUSDT", venue="binance"
        parts = inst_id.split(".")
        symbol = parts[0]
        venue = parts[1].lower() if len(parts) > 1 else "binance"

        # Discover funding files
        funding_dir = raw_data_path / venue / symbol / "funding"
        pattern = str(funding_dir / f"{symbol}_funding_*.csv")
        files = glob.glob(pattern)

        if files:
            # Use the latest file (sorted by name)
            file_path = Path(sorted(files)[-1])
            funding_rates = load_funding_rates(file_path, inst_id)
            for engine in node.get_engines():
                engine.add_data(funding_rates)
            print(f"Loaded {len(funding_rates)} funding rate updates for {inst_id}")
        else:
            print(f"Warning: Funding data not found: {pattern}")


class RunnerResult:
    """Result of a backtest run from runner."""

    def __init__(
        self,
        run_id: str,
        output_dir: Path | None,
        total_positions: int,
        total_orders: int,
        statistics: dict[str, Any],
        reports: dict[str, Path],
        duration: float,
    ) -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        self.total_positions = total_positions
        self.total_orders = total_orders
        self.statistics = statistics
        self.reports = reports
        self.duration = duration


def run_backtest(
    config_file: Path,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> RunnerResult | None:
    """Execute a backtest from a YAML configuration file.

    Args:
        config_file: Path to YAML config file
        dry_run: Validate config without executing
        verbose: Enable verbose output
        quiet: Suppress non-error output

    Returns:
        RunnerResult if successful, None if dry_run

    Raises:
        BacktestConfigError: If config is invalid
    """
    start_time = datetime.now()

    # Load YAML config
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Parse project-specific report config (before stripping for Nautilus)
    report_config = parse_report_config(config_dict)

    # Generate run_id and output_dir BEFORE running backtest (needed for logging)
    run_id = generate_run_id()
    output_dir = None
    if report_config:
        output_dir = create_output_directory(report_config.output_dir, run_id)

    # Extract data configs for injection
    data_configs = extract_data_configs(config_dict)

    # Inject data configs into actors/strategies
    config_dict = inject_data_configs(config_dict, data_configs)

    # Inject logging config to write log file to output directory
    if output_dir:
        config_dict = inject_logging_config(config_dict, output_dir)

    # Extract only Nautilus-compatible config
    nautilus_config = get_nautilus_config_dict(config_dict)

    if dry_run:
        # Try to parse to validate
        json_bytes = msgspec.json.encode(nautilus_config)
        BacktestRunConfig.parse(json_bytes)
        return None

    # Parse and run
    json_bytes = msgspec.json.encode(nautilus_config)
    run_config = BacktestRunConfig.parse(json_bytes)

    node = BacktestNode(configs=[run_config])

    # Load funding rate data if configured (before run)
    _load_funding_data(config_dict, node)

    node.run()

    # Get results
    engines = node.get_engines()
    if not engines:
        raise BacktestConfigError("No engines returned from backtest")

    engine = engines[0]
    # Get ALL positions (open + closed), not just currently open ones
    open_positions = len(engine.cache.positions())
    closed_positions = len(engine.cache.positions_closed())
    total_positions = open_positions + closed_positions
    orders = len(engine.cache.orders())

    # Generate reports if configured
    reports: dict[str, Path] = {}
    statistics: dict[str, Any] = {}

    if report_config and output_dir:
        report_generator = ReportGenerator(
            engine=engine,
            output_dir=output_dir,
            config=report_config,
        )
        reports = report_generator.generate_all()
        statistics = report_generator.generate_statistics()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    return RunnerResult(
        run_id=run_id,
        output_dir=output_dir,
        total_positions=total_positions,
        total_orders=orders,
        statistics=statistics,
        reports=reports,
        duration=duration,
    )


def validate_config(config_file: Path) -> bool:
    """Validate a configuration file without executing.

    Args:
        config_file: Path to YAML config file

    Returns:
        True if valid

    Raises:
        Exception: If config is invalid
    """
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    json_bytes = msgspec.json.encode(config_dict)
    BacktestRunConfig.parse(json_bytes)
    return True
