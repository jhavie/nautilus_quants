"""CLI entry point for backtest module.

For Actor-decoupled backtesting, use Nautilus native BacktestRunConfig directly.
See: config/factor/backtest.yaml for example usage.
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
@click.option("--no-registry", is_flag=True, help="Skip registry write")
@click.option("--env", "env_name", default=None, help="Registry environment")
def run(
    config_file: Path,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
    no_registry: bool,
    env_name: str | None,
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

        # ── Registry auto-persist ──
        if not no_registry and not dry_run and result is not None:
            try:
                import yaml as _yaml

                from datetime import datetime, timezone

                from nautilus_quants.alpha.registry.backtest_repository import (
                    BacktestRepository,
                )

                import math

                def _now_iso() -> str:
                    return datetime.now(timezone.utc).isoformat(
                        timespec="seconds",
                    )

                def _safe_stat(
                    stats: dict, key: str,
                ) -> float | None:
                    v = stats.get(key)
                    if v is None:
                        return None
                    if isinstance(v, float) and math.isnan(v):
                        return None
                    return float(v)
                from nautilus_quants.alpha.registry.database import RegistryDatabase
                from nautilus_quants.alpha.registry.environment import (
                    parse_registry_config,
                    resolve_env,
                )
                from nautilus_quants.alpha.registry.models import BacktestRunRecord
                from nautilus_quants.alpha.registry.repository import FactorRepository
                from nautilus_quants.factors.config import (
                    generate_factor_id,
                    load_factor_config,
                )

                with open(config_file, encoding="utf-8") as _f:
                    bt_config_dict = _yaml.safe_load(_f)

                reg_cfg = parse_registry_config(bt_config_dict.get("registry"))
                if not reg_cfg.enabled:
                    raise RuntimeError("disabled")

                env = resolve_env(env_name, reg_cfg.env)
                reg_db = RegistryDatabase.for_environment(env, reg_cfg.db_dir)
                repo = FactorRepository(reg_db)
                bt_repo = BacktestRepository(reg_db)

                # Save backtest config snapshot
                bt_cfg_id = repo.save_config_snapshot(
                    bt_config_dict, "backtest",
                    config_name=str(config_file.stem),
                    file_path=str(config_file),
                )

                # Find factor config path from backtest config
                factor_cfg_id = ""
                factor_ids: list[str] = []
                composite_id: str | None = None
                actors = (
                    bt_config_dict.get("engine", {}).get("actors", [])
                )
                fc = None
                for actor in actors:
                    actor_cfg = actor.get("config", {})
                    fcp = actor_cfg.get("factor_config_path")
                    if fcp:
                        with open(fcp, encoding="utf-8") as _f2:
                            fc_dict = _yaml.safe_load(_f2)
                        factor_cfg_id = repo.save_config_snapshot(
                            fc_dict, "factors",
                            file_path=fcp,
                        )
                        fc = load_factor_config(fcp)
                        reg_result = repo.register_factors_from_config(fc)
                        for fdef in fc.factors:
                            fid = reg_result.name_map.get(
                                fdef.name,
                                generate_factor_id(fc.source, fdef.name),
                            )
                            factor_ids.append(fid)
                    # composite_factor may be on a different actor
                    cf = actor_cfg.get("composite_factor")
                    if cf:
                        source = fc.source if fc else ""
                        composite_id = generate_factor_id(source, cf)

                # Extract data configs for instrument count & timeframe
                data_cfgs = bt_config_dict.get("data", [])
                inst_count = 0
                timeframe = ""
                if data_cfgs:
                    inst_count = len(data_cfgs[0].get("instrument_ids", []))
                    timeframe = data_cfgs[0].get("bar_spec", "")

                record = BacktestRunRecord(
                    backtest_id=result.run_id,
                    config_id=bt_cfg_id,
                    factor_config_id=factor_cfg_id,
                    output_dir=str(result.output_dir or ""),
                    strategy_name=bt_config_dict.get("engine", {}).get(
                        "trader_id", "",
                    ),
                    instrument_count=inst_count,
                    timeframe=timeframe,
                    started_at=_now_iso(),
                    duration_seconds=result.duration,
                    total_pnl=_safe_stat(result.statistics, "total_pnl"),
                    total_pnl_pct=_safe_stat(
                        result.statistics, "total_pnl_pct",
                    ),
                    sharpe_ratio=_safe_stat(
                        result.statistics, "sharpe_ratio",
                    ),
                    max_drawdown=_safe_stat(
                        result.statistics, "max_drawdown",
                    ),
                    win_rate=_safe_stat(result.statistics, "win_rate"),
                    statistics_json=result.statistics,
                    reports_json={
                        k: str(v) for k, v in result.reports.items()
                    },
                )
                bt_repo.save_backtest(
                    record, factor_ids=factor_ids,
                    composite_id=composite_id,
                )
                reg_db.close()

                if not quiet:
                    click.echo(
                        f"  Registry: backtest saved to {env}.duckdb "
                        f"({len(factor_ids)} factors linked)"
                    )
            except Exception as e:
                if str(e) != "disabled":
                    click.echo(
                        f"  Warning: Registry write failed: {e}",
                        err=True,
                    )

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
