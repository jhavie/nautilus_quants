"""CLI entry point for backtest module.

For Actor-decoupled backtesting, use Nautilus native BacktestRunConfig directly.
See: config/backtest_factor.yaml for example usage.
"""

import sys
from pathlib import Path

import click

from nautilus_quants.backtest.exceptions import BacktestConfigError


@click.group()
@click.version_option()
def cli() -> None:
    """Nautilus Quants - Backtest CLI.

    Configuration-driven backtesting with nautilus_trader.
    """
    pass


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
    from nautilus_quants.backtest.runner import run_backtest

    try:
        if not quiet:
            click.echo("=" * 80)
            click.echo("Nautilus Quants - Backtest")
            click.echo("=" * 80)
            click.echo(f"Config: {config_file}")
            click.echo("=" * 80)
            click.echo()

        result = run_backtest(
            config_file=config_file,
            dry_run=dry_run,
            verbose=verbose,
            quiet=quiet,
        )

        if dry_run:
            click.echo("Configuration valid (dry run)")
            sys.exit(0)

        if result is None:
            sys.exit(1)

        if not quiet:
            click.echo()
            click.echo("Backtest completed")
            click.echo(f"  Total positions: {result.total_positions}")
            click.echo(f"  Total orders: {result.total_orders}")

            if result.reports:
                click.echo()
                click.echo("=" * 80)
                click.echo("BACKTEST RESULTS")
                click.echo("=" * 80)
                click.echo(f"  Run ID: {result.run_id}")
                click.echo(f"  Duration: {result.duration:.2f}s")
                click.echo()

                # Display key statistics
                if result.statistics:
                    _print_statistics(result.statistics)

                click.echo("Reports generated:")
                for report_type, report_path in result.reports.items():
                    click.echo(f"  {report_type}: {report_path}")
                click.echo()
                click.echo(f"Output directory: {result.output_dir}")
                click.echo("=" * 80)
            else:
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


def _print_statistics(statistics: dict) -> None:
    """Print performance statistics to console."""
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


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
def validate(config_file: Path) -> None:
    """Validate a configuration file without executing."""
    from nautilus_quants.backtest.runner import validate_config

    try:
        validate_config(config_file)
        click.echo(f"Configuration valid: {config_file}")
    except Exception as e:
        click.echo(f"Configuration invalid: {config_file}", err=True)
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
