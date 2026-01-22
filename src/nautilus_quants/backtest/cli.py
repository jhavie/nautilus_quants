"""CLI entry point for backtest module."""

import sys
from pathlib import Path

import click

from nautilus_quants.backtest.config import BacktestConfig, BacktestResult
from nautilus_quants.backtest.exceptions import BacktestConfigError
from nautilus_quants.backtest.runner import BacktestRunner


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
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Override output directory",
)
@click.option(
    "-s",
    "--start-date",
    type=str,
    help="Override start date (YYYY-MM-DD)",
)
@click.option(
    "-e",
    "--end-date",
    type=str,
    help="Override end date (YYYY-MM-DD)",
)
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
    output_dir: Path | None,
    start_date: str | None,
    end_date: str | None,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Execute a backtest from a YAML configuration file."""
    try:
        # Load config
        config = BacktestConfig.from_yaml(config_file)

        # Apply overrides
        config_dict = config.to_dict()

        if output_dir:
            config_dict["report"]["output_dir"] = str(output_dir)
        if start_date:
            config_dict["backtest"]["start_date"] = start_date
        if end_date:
            config_dict["backtest"]["end_date"] = end_date

        config = BacktestConfig.from_dict(config_dict)

        if not quiet:
            _print_header(config, config_file)

        if dry_run:
            click.echo("✓ Configuration valid (dry run)")
            return

        # Run backtest
        runner = BacktestRunner(config)
        result = runner.run()

        if result.success:
            if not quiet:
                _print_results(result, verbose)
            sys.exit(0)
        else:
            click.echo(f"✗ Backtest failed: {result.errors}", err=True)
            sys.exit(4)

    except BacktestConfigError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(4)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-d",
    "--check-data",
    is_flag=True,
    help="Also verify data files exist",
)
@click.option(
    "-s",
    "--check-strategy",
    is_flag=True,
    help="Also verify strategy can be instantiated",
)
def validate(
    config_file: Path,
    check_data: bool,
    check_strategy: bool,
) -> None:
    """Validate a configuration file without executing."""
    try:
        config = BacktestConfig.from_yaml(config_file)

        click.echo(f"✓ Configuration valid: {config_file}")
        click.echo()
        click.echo(f"Strategy: {config.strategy.type}")
        click.echo(f"Instrument: {config.strategy.instrument_id}")
        click.echo(
            f"Period: {config.backtest.start_date} to {config.backtest.end_date}"
        )
        click.echo(
            f"Venue: {config.venue.name} "
            f"({config.venue.account_type}, {config.venue.default_leverage}x leverage)"
        )
        click.echo(f"Output: {config.report.output_dir}/{{run_id}}/")

        if check_data:
            _validate_data(config)

        if check_strategy:
            _validate_strategy(config)

    except BacktestConfigError as e:
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
            # Try to extract config fields
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


@cli.command()
@click.argument(
    "output_file",
    type=click.Path(path_type=Path),
    default="backtest.yaml",
)
@click.option(
    "-s",
    "--strategy",
    type=str,
    default="breakout",
    help="Strategy type to use",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Overwrite existing file",
)
def init(output_file: Path, strategy: str, force: bool) -> None:
    """Generate a sample configuration file."""
    if output_file.exists() and not force:
        click.echo(f"Error: {output_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    sample_config = f"""# Backtest Configuration
# Generated by: python -m nautilus_quants.backtest init

strategy:
  type: "{strategy}"
  instrument_id: "BTCUSDT"
  params:
    breakout_period: 60
    sma_period: 200
    position_size_pct: 0.10
    max_positions: 1
    stop_loss_pct: 0.05
    take_profit_pct: 0.10

backtest:
  catalog_path: "/path/to/your/catalog"  # Update this path
  start_date: "2025-01-01"
  end_date: "2025-12-31"
  bar_spec: "1h"
  warmup_days: 30

venue:
  name: "BINANCE"
  oms_type: "NETTING"
  account_type: "MARGIN"
  base_currency: "USDT"
  starting_balance: "100000 USDT"
  default_leverage: 5

  # Optional: Fill simulation
  # fill_model:
  #   prob_fill_on_limit: 1.0
  #   prob_slippage: 0.0

  # Optional: Fee structure
  fee_model:
    type: "maker_taker"
    maker_fee: 0.0002
    taker_fee: 0.0004

report:
  output_dir: "logs/backtest_runs"
  formats: [csv, html]
  tearsheet:
    enabled: true
    title: "Backtest Results"
    theme: "plotly_dark"
    height: 1500

logging:
  level: "INFO"
  log_to_file: true
  bypass_logging: false
"""

    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(sample_config)
    click.echo(f"✓ Created: {output_file}")
    click.echo(f"\nEdit the configuration, then run:")
    click.echo(f"  python -m nautilus_quants.backtest run {output_file}")


def _print_header(config: BacktestConfig, config_file: Path) -> None:
    """Print backtest header."""
    click.echo("=" * 80)
    click.echo("Nautilus Quants - Backtest")
    click.echo("=" * 80)
    click.echo(f"Config: {config_file}")
    click.echo(
        f"Strategy: {config.strategy.type} ({config.strategy.instrument_id})"
    )
    click.echo(
        f"Period: {config.backtest.start_date} to {config.backtest.end_date}"
    )
    click.echo("=" * 80)
    click.echo()


def _print_results(result: BacktestResult, verbose: bool) -> None:
    """Print backtest results."""
    if not isinstance(result, BacktestResult):
        return

    click.echo()
    click.echo(f"✓ Backtest completed in {result.duration_seconds:.1f} seconds")
    click.echo()

    # Print statistics
    if result.statistics:
        click.echo("Performance Summary:")
        if result.total_pnl:
            pnl_str = f"{result.total_pnl:+,.2f}"
            pct_str = f"{result.total_pnl_pct:+.2%}" if result.total_pnl_pct else ""
            click.echo(f"  Total PnL: {pnl_str} {pct_str}")
        if result.sharpe_ratio:
            click.echo(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        if result.max_drawdown:
            click.echo(f"  Max Drawdown: {result.max_drawdown:.2%}")
        if result.win_rate:
            click.echo(f"  Win Rate: {result.win_rate:.1%}")

    click.echo()
    click.echo("=" * 80)
    click.echo(f"✓ Complete! Results saved to: {result.output_dir}/")
    click.echo("=" * 80)


def _validate_data(config: BacktestConfig) -> None:
    """Validate data catalog exists."""
    from pathlib import Path

    catalog_path = Path(config.backtest.catalog_path)
    if not catalog_path.exists():
        click.echo(f"✗ Data validation failed: Catalog path does not exist: {catalog_path}", err=True)
        sys.exit(2)
    click.echo("✓ Catalog path found")


def _validate_strategy(config: BacktestConfig) -> None:
    """Validate strategy can be instantiated."""
    try:
        from nautilus_quants.strategies import STRATEGY_REGISTRY

        if config.strategy.type not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {config.strategy.type}")
        click.echo("✓ Strategy found in registry")
    except Exception as e:
        click.echo(f"✗ Strategy validation failed: {e}", err=True)
        sys.exit(3)


if __name__ == "__main__":
    cli()
