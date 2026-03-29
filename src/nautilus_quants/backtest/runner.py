"""Backtest execution runner."""

from datetime import datetime
from pathlib import Path
from typing import Any

import msgspec
import yaml

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import BacktestRunConfig

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
        # Get metadata renderer from config (explicit) or use default
        from nautilus_quants.utils.registry import RendererRegistry

        renderer_name = None
        if report_config.position_viz:
            renderer_name = report_config.position_viz.metadata_renderer

        metadata_renderer = RendererRegistry.get(renderer_name)

        report_generator = ReportGenerator(
            engine=engine,
            output_dir=output_dir,
            config=report_config,
            metadata_renderer=metadata_renderer,
        )
        reports = report_generator.generate_all()
        statistics = report_generator.generate_statistics()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Dispose engine now that reports are generated (we set
    # dispose_on_completion=False to keep cache data alive for reports).
    engine.dispose()

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
