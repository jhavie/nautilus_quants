"""CLI entry point for the live trading module."""

import sys
from pathlib import Path

import click

from nautilus_quants.live.exceptions import LiveConfigError


@click.group()
@click.version_option()
def cli() -> None:
    """Nautilus Quants - Live Trading CLI.

    Configuration-driven live trading with nautilus_trader.
    """


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help="Validate config without connecting to exchange",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def run(config_file: Path, dry_run: bool, verbose: bool) -> None:
    """Start live trading from a YAML configuration file.

    Use Ctrl+C for graceful shutdown.
    """
    from nautilus_quants.live.runner import run_live

    try:
        click.echo("=" * 80)
        click.echo("Nautilus Quants - Live Trading")
        click.echo("=" * 80)
        click.echo(f"Config: {config_file}")
        if dry_run:
            click.echo("Mode: DRY RUN (no actual trading)")
        click.echo("=" * 80)
        click.echo()

        run_live(
            config_file=config_file,
            dry_run=dry_run,
            verbose=verbose,
        )

        if dry_run:
            click.echo()
            click.echo("Configuration valid (dry run)")
            sys.exit(0)

    except LiveConfigError as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nShutdown requested...")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(4)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
def validate(config_file: Path) -> None:
    """Validate a live configuration file without executing."""
    from nautilus_quants.live.runner import validate_config

    try:
        validate_config(config_file)
        click.echo(f"Configuration valid: {config_file}")
    except LiveConfigError as e:
        click.echo(f"Configuration invalid: {config_file}", err=True)
        click.echo(f"\nErrors:\n  {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Configuration invalid: {config_file}", err=True)
        click.echo(f"\nErrors:\n  {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
