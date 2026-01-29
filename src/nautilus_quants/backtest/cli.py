"""CLI entry point for backtest module.

For Actor-decoupled backtesting, use Nautilus native BacktestRunConfig directly.
See: config/backtest_factor.yaml for example usage.
"""

import sys
from datetime import datetime
from pathlib import Path

import click
import msgspec
import yaml

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import BacktestRunConfig

from nautilus_quants.backtest.config import (
    BacktestResult,
    QuantStatsConfig,
    ReportConfig,
    TearsheetConfig,
)
from nautilus_quants.backtest.exceptions import BacktestConfigError
from nautilus_quants.backtest.reports import ReportGenerator
from nautilus_quants.backtest.utils.bar_spec import format_bar_spec
from nautilus_quants.backtest.utils.reporting import create_output_directory, generate_run_id


@click.group()
@click.version_option()
def cli() -> None:
    """Nautilus Quants - Backtest CLI.

    Configuration-driven backtesting with nautilus_trader.
    """
    pass


def _parse_report_config(config_dict: dict) -> ReportConfig | None:
    """Parse report configuration from YAML dict.
    
    Args:
        config_dict: Full YAML config dictionary
        
    Returns:
        ReportConfig if report section exists, None otherwise
    """
    report_section = config_dict.get("report")
    if not report_section:
        return None
    
    # Parse tearsheet config if present
    tearsheet_config = None
    tearsheet_section = report_section.get("tearsheet")
    if tearsheet_section:
        tearsheet_config = TearsheetConfig(
            enabled=tearsheet_section.get("enabled", True),
            title=tearsheet_section.get("title", "Backtest Results"),
            theme=tearsheet_section.get("theme", "plotly_dark"),
            height=tearsheet_section.get("height", 1500),
            show_logo=tearsheet_section.get("show_logo", True),
            include_benchmark=tearsheet_section.get("include_benchmark", False),
            benchmark_name=tearsheet_section.get("benchmark_name", "Benchmark"),
            charts=tearsheet_section.get("charts", [
                "run_info", "stats_table", "equity", "drawdown",
                "monthly_returns", "distribution", "rolling_sharpe", "yearly_returns"
            ]),
        )

    # Parse quantstats config if present
    quantstats_config = None
    quantstats_section = report_section.get("quantstats")
    if quantstats_section:
        quantstats_config = QuantStatsConfig(
            enabled=quantstats_section.get("enabled", False),
            title=quantstats_section.get("title", "QuantStats Report"),
            benchmark=quantstats_section.get("benchmark"),
            output_format=quantstats_section.get("output_format", ["html"]),
            charts=quantstats_section.get("charts", [
                "returns", "log_returns", "yearly_returns", "monthly_heatmap",
                "drawdown", "rolling_sharpe", "rolling_volatility"
            ]),
        )

    return ReportConfig(
        output_dir=report_section.get("output_dir", "logs/backtest_runs"),
        formats=report_section.get("formats", ["csv", "html"]),
        tearsheet=tearsheet_config,
        quantstats=quantstats_config,
    )


def _get_nautilus_config_dict(config_dict: dict) -> dict:
    """Extract only Nautilus-compatible config keys.
    
    Removes project-specific keys (report, logging) that are not part of
    Nautilus BacktestRunConfig schema.
    
    Args:
        config_dict: Full YAML config dictionary
        
    Returns:
        Dict with only Nautilus-compatible keys
    """
    # Keys that Nautilus BacktestRunConfig expects
    nautilus_keys = {"venues", "data", "engine", "batch_size_bytes"}
    return {k: v for k, v in config_dict.items() if k in nautilus_keys}


def _extract_data_configs(config_dict: dict) -> list[dict]:
    """Extract instrument_ids and bar_spec from data section.

    Expects catalog format with instrument_ids (plural), catalog_path, and bar_spec.

    Args:
        config_dict: Full YAML config dictionary

    Returns:
        List of dicts with instrument_id and bar_spec for each data source
    """
    data_section = config_dict.get("data", [])
    result = []

    for data_config in data_section:
        instrument_ids = data_config.get("instrument_ids", [])
        catalog_path = data_config.get("catalog_path", "")
        bar_spec = data_config.get("bar_spec", "")

        if not instrument_ids or not catalog_path or not bar_spec:
            continue

        # Convert bar_spec to native format (e.g., "1h" -> "1-HOUR-LAST")
        try:
            native_spec = format_bar_spec(bar_spec, include_source=False)
        except ValueError:
            # Assume already in native format
            native_spec = bar_spec

        for inst_id in instrument_ids:
            bar_type = f"{inst_id}-{native_spec}-EXTERNAL"
            result.append({
                "instrument_id": inst_id,
                "bar_spec": native_spec,
                "bar_type": bar_type,
            })

    return result


def _inject_data_configs(config_dict: dict, data_configs: list[dict]) -> dict:
    """Inject data configs into actor and strategy configurations.
    
    This allows actors and strategies to know which bar types to subscribe to
    without querying the cache (which is empty at on_start time).
    
    Args:
        config_dict: Full YAML config dictionary
        data_configs: List of extracted data configs
        
    Returns:
        Modified config dict with injected bar_type info
    """
    if not data_configs:
        return config_dict
    
    # Deep copy to avoid modifying original
    import copy
    config_dict = copy.deepcopy(config_dict)
    
    engine = config_dict.get("engine", {})
    
    # Inject into actors
    actors = engine.get("actors", [])
    for actor in actors:
        actor_config = actor.get("config", {})
        # Inject bar_types list if not already present
        if "bar_types" not in actor_config:
            actor_config["bar_types"] = [dc["bar_type"] for dc in data_configs]
        actor["config"] = actor_config
    
    # Inject into strategies
    strategies = engine.get("strategies", [])
    for strategy in strategies:
        strategy_config = strategy.get("config", {})
        # Inject bar_types list if not already present (for multi-instrument strategies)
        if "bar_types" not in strategy_config:
            strategy_config["bar_types"] = [dc["bar_type"] for dc in data_configs]
        # Inject bar_type if not already present and we have data configs (for single-instrument strategies)
        if "bar_type" not in strategy_config and data_configs:
            # Find matching data config by instrument_id
            instrument_id = strategy_config.get("instrument_id")
            for dc in data_configs:
                if dc["instrument_id"] == instrument_id:
                    strategy_config["bar_type"] = dc["bar_type"]
                    break
            else:
                # Default to first data config if no match
                strategy_config["bar_type"] = data_configs[0]["bar_type"]
        strategy["config"] = strategy_config
    
    return config_dict


def _inject_logging_config(config_dict: dict, output_dir: Path) -> dict:
    """Inject logging configuration to write log file to output directory.
    
    Args:
        config_dict: Full YAML config dictionary
        output_dir: Output directory for log file
        
    Returns:
        Modified config dict with logging config
    """
    import copy
    config_dict = copy.deepcopy(config_dict)
    
    engine = config_dict.get("engine", {})
    logging_config = engine.get("logging", {})
    
    # Set log file configuration
    logging_config["log_level_file"] = logging_config.get("log_level", "INFO")
    logging_config["log_directory"] = str(output_dir)
    logging_config["log_file_name"] = "nautilus.log"
    
    engine["logging"] = logging_config
    config_dict["engine"] = engine
    
    return config_dict


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help="Validate config without executing",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress non-error output",
)
def run(
    config_file: Path,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Execute a backtest from a YAML configuration file.
    
    Supports Nautilus native BacktestRunConfig format with optional
    project-specific report configuration.
    """
    start_time = datetime.now()
    
    try:
        # Load YAML config
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        if not quiet:
            click.echo("=" * 80)
            click.echo("Nautilus Quants - Backtest")
            click.echo("=" * 80)
            click.echo(f"Config: {config_file}")
            click.echo("=" * 80)
            click.echo()

        # Parse project-specific report config (before stripping for Nautilus)
        report_config = _parse_report_config(config_dict)
        
        # Generate run_id and output_dir BEFORE running backtest (needed for logging)
        run_id = generate_run_id()
        output_dir = None
        if report_config:
            output_dir = create_output_directory(report_config.output_dir, run_id)
        
        # Extract data configs for injection
        data_configs = _extract_data_configs(config_dict)
        
        # Inject data configs into actors/strategies
        config_dict = _inject_data_configs(config_dict, data_configs)
        
        # Inject logging config to write log file to output directory
        if output_dir:
            config_dict = _inject_logging_config(config_dict, output_dir)
        
        # Extract only Nautilus-compatible config
        nautilus_config = _get_nautilus_config_dict(config_dict)

        if dry_run:
            # Try to parse to validate
            json_bytes = msgspec.json.encode(nautilus_config)
            BacktestRunConfig.parse(json_bytes)
            click.echo("✓ Configuration valid (dry run)")
            if report_config:
                click.echo(f"  Report output: {report_config.output_dir}")
            return

        # Parse and run
        json_bytes = msgspec.json.encode(nautilus_config)
        run_config = BacktestRunConfig.parse(json_bytes)

        node = BacktestNode(configs=[run_config])
        node.run()

        # Get results
        engines = node.get_engines()
        if engines:
            engine = engines[0]
            # Get ALL positions (open + closed), not just currently open ones
            open_positions = len(engine.cache.positions())
            closed_positions = len(engine.cache.positions_closed())
            total_positions = open_positions + closed_positions
            orders = len(engine.cache.orders())
            
            if not quiet:
                click.echo()
                click.echo("✓ Backtest completed")
                click.echo(f"  Total positions: {total_positions}")
                click.echo(f"  Total orders: {orders}")
            
            # Generate reports if configured
            if report_config and output_dir:
                if not quiet:
                    click.echo()
                    click.echo("Generating reports...")
                
                report_generator = ReportGenerator(
                    engine=engine,
                    output_dir=output_dir,
                    config=report_config,
                )
                reports = report_generator.generate_all()
                statistics = report_generator.generate_statistics()
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if not quiet:
                    click.echo()
                    click.echo("=" * 80)
                    click.echo("BACKTEST RESULTS")
                    click.echo("=" * 80)
                    click.echo(f"  Run ID: {run_id}")
                    click.echo(f"  Duration: {duration:.2f}s")
                    click.echo()
                    
                    # Display key statistics
                    if statistics:
                        click.echo("Performance Metrics:")
                        if "PnL (total)" in statistics:
                            click.echo(f"  PnL (total): {statistics['PnL (total)']:.2f}")
                        if "PnL% (total)" in statistics:
                            # Note: Nautilus returns PnL% as percentage value (e.g., 16.61 means 16.61%)
                            click.echo(f"  PnL% (total): {statistics['PnL% (total)']:.2f}%")
                        if "Win Rate" in statistics:
                            # Win Rate is a ratio (0.24 means 24%), so use :.2%
                            click.echo(f"  Win Rate: {statistics['Win Rate']:.2%}")
                        if "Sharpe Ratio" in statistics:
                            click.echo(f"  Sharpe Ratio: {statistics['Sharpe Ratio']:.4f}")
                        if "Max Drawdown" in statistics:
                            click.echo(f"  Max Drawdown: {statistics['Max Drawdown']:.2%}")
                        if "Avg Winner" in statistics:
                            click.echo(f"  Avg Winner: {statistics['Avg Winner']:.2f}")
                        if "Avg Loser" in statistics:
                            click.echo(f"  Avg Loser: {statistics['Avg Loser']:.2f}")
                        click.echo()
                    
                    click.echo("Reports generated:")
                    for report_type, report_path in reports.items():
                        click.echo(f"  {report_type}: {report_path}")
                    click.echo()
                    click.echo(f"Output directory: {output_dir}")
                    click.echo("=" * 80)
            else:
                if not quiet:
                    click.echo("=" * 80)
                    click.echo("  (No report config - skipping report generation)")

        sys.exit(0)

    except BacktestConfigError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(4)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
def validate(config_file: Path) -> None:
    """Validate a configuration file without executing."""
    try:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        json_bytes = msgspec.json.encode(config_dict)
        BacktestRunConfig.parse(json_bytes)

        click.echo(f"✓ Configuration valid: {config_file}")

    except Exception as e:
        click.echo(f"✗ Configuration invalid: {config_file}", err=True)
        click.echo(f"\nErrors:\n  {e}", err=True)
        sys.exit(1)


@cli.command("list")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show strategy parameters",
)
def list_strategies(verbose: bool) -> None:
    """List available strategies."""
    try:
        from nautilus_quants.strategies import STRATEGY_REGISTRY
    except ImportError:
        click.echo("No strategies registered.", err=True)
        sys.exit(1)

    click.echo("Available Strategies:")

    for name, (strategy_class, config_class) in STRATEGY_REGISTRY.items():
        doc = strategy_class.__doc__ or "No description"
        doc_line = doc.split("\n")[0].strip()

        if verbose:
            click.echo(f"\n{name} - {doc_line}")
            click.echo("  Parameters:")
            try:
                import inspect

                sig = inspect.signature(config_class)
                for param_name, param in sig.parameters.items():
                    if param_name in ("self", "instrument_id"):
                        continue
                    default = (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else "required"
                    )
                    click.echo(f"    {param_name} = {default}")
            except Exception:
                click.echo("    (could not extract parameters)")
        else:
            click.echo(f"  {name:12} {doc_line}")


if __name__ == "__main__":
    cli()
