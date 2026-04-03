"""CLI entry point for alpha module.

Usage:
    python -m nautilus_quants.alpha analyze config/fmz/alpha.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import pandas as pd


def _run_alphalens_worker(
    factor_name: str,
    factor_series,
    forward_returns,
    config,
) -> tuple[str, dict | str]:
    """Worker function for parallel alphalens analysis.

    Runs in a subprocess via ProcessPoolExecutor. Returns (name, result_dict)
    on success or (name, error_string) on failure.
    """
    from nautilus_quants.alpha.analysis.evaluator import run_alphalens_with_forward_returns

    try:
        result = run_alphalens_with_forward_returns(
            factor_series, forward_returns, config.quantiles, config.max_loss,
        )
        return factor_name, result
    except Exception as e:
        return factor_name, str(e)

def _generate_charts_worker(
    factor_name: str,
    factor_data,
    charts: list[str],
    output_format: tuple[str, ...],
    output_dir: str,
    pricing=None,
) -> tuple[str, int | str]:
    """Worker function for parallel chart generation.

    Each subprocess gets its own matplotlib instance with Agg backend,
    avoiding thread-safety issues.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nautilus_quants.alpha.analysis.report import CHART_REGISTRY

    try:
        factor_dir = Path(output_dir) / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)

        period_cols = [
            c for c in factor_data.columns
            if c not in ("factor", "factor_quantile")
        ]
        period = period_cols[0] if period_cols else "1h"
        count = 0

        for chart_name in charts:
            chart_func = CHART_REGISTRY.get(chart_name)
            if chart_func is None:
                continue
            try:
                plt.figure(figsize=(12, 6))
                chart_func(factor_data, period, pricing=pricing)
                for fmt in output_format:
                    path = factor_dir / f"{chart_name}.{fmt}"
                    plt.savefig(path, bbox_inches="tight", dpi=100)
                    count += 1
                plt.close("all")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Chart {chart_name} failed for {factor_name}: {e}",
                )
                plt.close("all")

        return factor_name, count
    except Exception as e:
        return factor_name, str(e)



@click.group()
def cli() -> None:
    """Nautilus Quants - Alpha Factor Analysis."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
@click.option("--no-registry", is_flag=True, help="Skip writing to registry database")
@click.option("--env", "env_name", default=None, help="Registry environment (test/dev/prod)")
def analyze(
    config_file: Path, verbose: bool, quiet: bool,
    no_registry: bool, env_name: str | None,
) -> None:
    """Execute factor analysis from a YAML configuration file.

    Loads bar data from catalog, computes factor values using FactorEngine,
    and evaluates factor quality using alphalens IC/ICIR analysis.
    """
    from nautilus_quants.alpha.analysis.config import load_analysis_config
    from nautilus_quants.alpha.analysis.evaluator import FactorEvaluator
    from nautilus_quants.alpha.analysis.report import AnalysisReportGenerator
    from nautilus_quants.alpha.data_loader import CatalogDataLoader
    from nautilus_quants.backtest.utils.reporting import (
        create_output_directory,
        generate_run_id,
    )
    from nautilus_quants.factors.config import load_factor_config

    start_time = time.time()

    try:
        # 1. Load config
        config = load_analysis_config(config_file)

        if not quiet:
            click.echo("=" * 80)
            click.echo("Nautilus Quants - Alpha Factor Analysis")
            click.echo("=" * 80)
            click.echo(f"Config: {config_file}")
            click.echo("=" * 80)
            click.echo()

        # 2. Generate run ID and create output directory
        run_id = generate_run_id()
        output_dir = create_output_directory(config.output_dir, run_id)

        # 3. Load bar data from catalog
        if not quiet:
            click.echo(f"Loading bar data from {config.catalog_path}...")

        loader = CatalogDataLoader(config.catalog_path, config.bar_spec)
        bars_by_instrument = loader.load_bars(config.instrument_ids)

        loaded_count = sum(len(bars) for bars in bars_by_instrument.values())
        instruments_with_data = sum(
            1 for bars in bars_by_instrument.values() if bars
        )

        if not quiet:
            click.echo(
                f"Loaded {loaded_count} bars across "
                f"{instruments_with_data} instruments"
            )

        if loaded_count == 0:
            click.echo("Error: No bar data loaded. Check catalog path and instrument IDs.", err=True)
            sys.exit(1)

        # 4. Load factor config and create evaluator
        factor_config = load_factor_config(config.factor_config_path)
        evaluator = FactorEvaluator(factor_config)

        # 5. Compute factors (with lazy cache support)
        from nautilus_quants.factors.cache import (
            compute_config_hash,
            has_cache,
            load_as_factor_series,
            save_factor_cache,
            validate_cache,
        )

        config_hash = compute_config_hash(factor_config)
        use_cache = False

        if config.factor_cache_path and has_cache(config.factor_cache_path):
            valid, warnings = validate_cache(
                config.factor_cache_path,
                config_hash,
                expected_instruments=set(config.instrument_ids),
            )
            for w in warnings:
                click.echo(f"  Warning: {w}", err=True)
            if valid:
                use_cache = True
            elif not quiet:
                click.echo("  Factor config changed, re-computing...")

        if use_cache:
            if not quiet:
                click.echo()
                click.echo(
                    f"Loading factors from cache: {config.factor_cache_path}"
                )
            factor_series = load_as_factor_series(config.factor_cache_path)
            # pricing is always needed for forward returns (not cached)
            import pandas as pd
            pricing = pd.DataFrame(
                {
                    inst_id: CatalogDataLoader.bars_to_dataframe(bars)["close"]
                    for inst_id, bars in bars_by_instrument.items()
                    if bars
                }
            )
        else:
            if not quiet:
                click.echo()
                click.echo("Computing factors...")
            factor_series, pricing = evaluator.evaluate(bars_by_instrument)
            # Save to cache if path configured
            if config.factor_cache_path and factor_series:
                save_factor_cache(
                    factor_series,
                    config.factor_cache_path,
                    factor_config_path=config.factor_config_path,
                    config_hash=config_hash,
                )
                if not quiet:
                    click.echo(
                        f"  Factor cache saved: {config.factor_cache_path}"
                    )

        if not factor_series:
            click.echo("Error: No factor values computed. Check factor configuration.", err=True)
            sys.exit(1)

        # 6. Filter to requested factors
        computed_factor_names = list(factor_series.keys())
        if config.factors:
            factor_series = {
                k: v for k, v in factor_series.items() if k in config.factors
            }

        if not factor_series:
            click.echo(
                f"Error: None of the requested factors {config.factors} "
                f"were found in computed factors: {computed_factor_names}",
                err=True,
            )
            sys.exit(1)

        if not quiet:
            click.echo(
                f"Analyzing {len(factor_series)} factors across "
                f"{instruments_with_data} instruments..."
            )

        # 7. Pre-compute forward returns once (shared across all factors)
        if not quiet:
            click.echo("  Pre-computing forward returns...")

        t0 = time.time()
        # Use any factor's series to seed the index for forward returns
        any_factor_series = next(iter(factor_series.values()))
        forward_returns = evaluator.compute_forward_returns(
            any_factor_series, pricing, config,
        )
        if verbose:
            click.echo(f"  Forward returns computed in {time.time() - t0:.2f}s")

        # 8. Run alphalens per factor in parallel, then generate charts serially
        from concurrent.futures import ProcessPoolExecutor

        report_gen = AnalysisReportGenerator(config)
        ic_results: dict[str, pd.DataFrame] = {}
        al_results: dict[str, dict] = {}

        # Phase 1: Parallel alphalens analysis (CPU-bound, no matplotlib)
        n_workers = min(len(factor_series), 4)
        if not quiet:
            click.echo(f"  Running alphalens in parallel ({n_workers} workers)...")

        t1 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _run_alphalens_worker,
                    factor_name,
                    series,
                    forward_returns,
                    config,
                ): factor_name
                for factor_name, series in factor_series.items()
            }
            for future in futures:
                factor_name = futures[future]
                name, result = future.result()
                if isinstance(result, str):
                    click.echo(f"  Warning: {name} failed: {result}", err=True)
                else:
                    al_results[name] = result
                    ic_results[name] = result["ic"]

        if verbose:
            click.echo(f"  Alphalens analysis completed in {time.time() - t1:.2f}s")

        # Phase 2: Parallel chart generation (each subprocess has own matplotlib)
        if config.charts and al_results:
            t2 = time.time()
            if not quiet:
                click.echo(f"  Generating charts in parallel ({len(al_results)} workers)...")

            chart_futures = {}
            with ProcessPoolExecutor(max_workers=min(len(al_results), 4)) as executor:
                for factor_name, al_result in al_results.items():
                    chart_futures[executor.submit(
                        _generate_charts_worker,
                        factor_name,
                        al_result["factor_data"],
                        config.charts,
                        config.output_format,
                        str(output_dir),
                        pricing,
                    )] = factor_name

                for future in chart_futures:
                    fname = chart_futures[future]
                    name, result = future.result()
                    if isinstance(result, str):
                        click.echo(f"  Warning: {name} chart failed: {result}", err=True)

            if verbose:
                click.echo(f"  Charts generated in {time.time() - t2:.2f}s")

        # 9. Generate summary
        if ic_results:
            report_gen.generate_summary(ic_results, output_dir, factor_series=factor_series)

        # 9a. Extended factor signal quality metrics
        factor_metrics_results = None

        if config.metrics.factor_metrics and al_results:
            from nautilus_quants.alpha.analysis.report import compute_all_factor_metrics

            if not quiet:
                click.echo("  Computing factor signal quality metrics...")
            factor_metrics_results = {}
            for fname, al_result in al_results.items():
                factor_metrics_results[fname] = compute_all_factor_metrics(
                    al_result["factor_data"],
                    ic_results[fname],
                    total_timestamps=len(pricing),
                    total_assets=len(pricing.columns),
                )

        if factor_metrics_results:
            report_gen.generate_extended_summary(
                factor_metrics_results, output_dir,
            )

        # ── Registry auto-persist ──
        registry_enabled = config.registry_enabled and not no_registry
        if registry_enabled:
            try:
                import yaml as _yaml

                from nautilus_quants.alpha.analysis.report import (
                    build_analysis_metrics,
                    compute_ic_summary,
                )
                from nautilus_quants.alpha.registry.database import RegistryDatabase
                from nautilus_quants.alpha.registry.environment import resolve_env
                from nautilus_quants.alpha.registry.repository import FactorRepository
                from nautilus_quants.factors.config import generate_factor_id

                env = resolve_env(env_name, config.registry_env)
                reg_db = RegistryDatabase.for_environment(
                    env, config.registry_db_dir,
                )
                repo = FactorRepository(reg_db)

                # Save config snapshots
                with open(config.factor_config_path, encoding="utf-8") as _f:
                    factor_yaml_dict = _yaml.safe_load(_f)
                factor_cfg_id = repo.save_config_snapshot(
                    factor_yaml_dict, "factors",
                    config_name=factor_config.name,
                    file_path=str(config.factor_config_path),
                )

                with open(config_file, encoding="utf-8") as _f:
                    analysis_yaml_dict = _yaml.safe_load(_f)
                analysis_cfg_id = repo.save_config_snapshot(
                    analysis_yaml_dict, "analysis",
                    config_name=str(config_file.stem),
                    file_path=str(config_file),
                )

                # Register factors
                repo.register_factors_from_config(factor_config)

                # Save analysis metrics
                all_metrics = []
                for fname, ic_df in ic_results.items():
                    fid = generate_factor_id(factor_config.source, fname)
                    ic_summary = compute_ic_summary(ic_df)
                    fm = (
                        factor_metrics_results.get(fname)
                        if factor_metrics_results else None
                    )
                    all_metrics.extend(build_analysis_metrics(
                        run_id=run_id,
                        factor_id=fid,
                        timeframe=config.bar_spec,
                        ic_summary=ic_summary,
                        metrics_result=fm,
                        factor_config_id=factor_cfg_id,
                        analysis_config_id=analysis_cfg_id,
                        output_dir=str(output_dir),
                    ))

                repo.save_metrics(all_metrics)
                reg_db.close()

                if not quiet:
                    click.echo(
                        f"  Registry: {len(all_metrics)} metrics saved to "
                        f"{env}.duckdb"
                    )
            except Exception as e:
                click.echo(
                    f"  Warning: Registry write failed: {e}", err=True,
                )

        duration = time.time() - start_time

        # 10. Print results
        if not quiet:
            click.echo()
            click.echo("Factor Analysis Summary:")
            report_gen.print_summary_table(ic_results, factor_series=factor_series)

            if factor_metrics_results:
                report_gen.print_extended_summary(factor_metrics_results)

            click.echo()
            click.echo("=" * 80)
            click.echo("ANALYSIS RESULTS")
            click.echo("=" * 80)
            click.echo(f"  Run ID: {run_id}")
            click.echo(f"  Duration: {duration:.2f}s")
            click.echo()

            click.echo("Reports generated:")
            click.echo(f"  summary: {output_dir / 'summary.txt'}")
            for factor_name in ic_results:
                factor_dir = output_dir / factor_name
                chart_count = len(list(factor_dir.iterdir())) if factor_dir.exists() else 0
                click.echo(f"  {factor_name}: {factor_dir}/ ({chart_count} charts)")
            click.echo()
            click.echo(f"Output directory: {output_dir}")
            click.echo("=" * 80)

        sys.exit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Registry commands (v2)
# ---------------------------------------------------------------------------

_ENV_OPTION = click.option(
    "--env", "env_name", default=None,
    type=click.Choice(["test", "dev", "prod"]),
    help="Registry environment (default: test).",
)
_DB_DIR_OPTION = click.option(
    "--db-dir", default="logs/registry",
    help="Registry database directory.",
)


def _open_repo(env_name: str | None, db_dir: str):
    """Open a FactorRepository for the given environment."""
    from nautilus_quants.alpha.registry.database import RegistryDatabase
    from nautilus_quants.alpha.registry.environment import resolve_env
    from nautilus_quants.alpha.registry.repository import FactorRepository

    env = resolve_env(env_name)
    db = RegistryDatabase.for_environment(env, db_dir)
    return FactorRepository(db), db


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@_ENV_OPTION
@_DB_DIR_OPTION
def register(config_file: Path, env_name: str | None, db_dir: str) -> None:
    """Register factors from a YAML config file into the registry."""
    from nautilus_quants.factors.config import load_factor_config

    try:
        config = load_factor_config(config_file)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    repo, db = _open_repo(env_name, db_dir)
    try:
        new, updated, unchanged = repo.register_factors_from_config(config)
        click.echo(
            f"Registered {new + updated + unchanged} factors "
            f"({new} new, {updated} updated, {unchanged} unchanged)"
        )
    finally:
        db.close()


@cli.command("list")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--source", default=None, help="Filter by source.")
@click.option("--prototype", default=None, help="Filter by prototype.")
@click.option("--limit", default=None, type=int, help="Max rows.")
@_ENV_OPTION
@_DB_DIR_OPTION
def list_factors(
    status: str | None,
    source: str | None,
    prototype: str | None,
    limit: int | None,
    env_name: str | None,
    db_dir: str,
) -> None:
    """List factors in the registry."""
    repo, db = _open_repo(env_name, db_dir)
    try:
        factors = repo.list_factors(
            status=status, source=source, prototype=prototype, limit=limit,
        )
        if not factors:
            click.echo("(no factors found)")
            return

        click.echo(
            f"{'factor_id':<30} {'prototype':<14} {'status':<12} "
            f"{'source':<10} {'tags'}"
        )
        click.echo("-" * 80)
        for f in factors:
            tags_str = ", ".join(f.tags) if f.tags else "-"
            click.echo(
                f"{f.factor_id:<30} {f.prototype:<14} {f.status:<12} "
                f"{f.source:<10} {tags_str}"
            )
        click.echo(f"({len(factors)} factors)")
    finally:
        db.close()


@cli.command()
@click.argument("factor_id")
@_ENV_OPTION
@_DB_DIR_OPTION
def inspect(factor_id: str, env_name: str | None, db_dir: str) -> None:
    """Inspect a factor's details and latest metrics."""
    repo, db = _open_repo(env_name, db_dir)
    try:
        f = repo.get_factor(factor_id)
        if f is None:
            click.echo(f"Error: factor not found: {factor_id}", err=True)
            sys.exit(1)

        click.echo(f"Factor: {f.factor_id}")
        click.echo(f"Expression: {f.expression}")
        click.echo(f"Prototype: {f.prototype or '(none)'}")
        click.echo(f"Source: {f.source or '(none)'}")
        click.echo(f"Status: {f.status}")
        click.echo(f"Tags: {', '.join(f.tags) if f.tags else '(none)'}")
        click.echo(f"Parameters: {f.parameters}")
        click.echo(f"Variables: {f.variables}")

        metrics = repo.get_metrics(factor_id)
        if metrics:
            click.echo(f"\nAnalysis metrics ({len(metrics)} records):")
            for m in metrics[:12]:
                icir_str = f"{m.icir:.4f}" if m.icir is not None else "-"
                ic_str = f"{m.ic_mean:.4f}" if m.ic_mean is not None else "-"
                click.echo(
                    f"  run={m.run_id} period={m.period} "
                    f"IC={ic_str} ICIR={icir_str} timeframe={m.timeframe}"
                )

        from nautilus_quants.alpha.registry.backtest_repository import (
            BacktestRepository,
        )

        bt_repo = BacktestRepository(db)
        runs = bt_repo.list_backtests(factor_id=factor_id)
        if runs:
            click.echo(f"\nBacktests ({len(runs)} records):")
            for r in runs:
                def _v(v: float | None, fmt: str = ".4f") -> str:
                    return f"{v:{fmt}}" if v is not None else "-"
                dd = f"{r.max_drawdown:.2%}" if r.max_drawdown else "-"
                factors = bt_repo.get_backtest_factors(r.backtest_id)
                fids = ", ".join(bf.factor_id for bf in factors) if factors else "-"
                click.echo(
                    f"  {r.backtest_id}  sharpe={_v(r.sharpe_ratio)} "
                    f"pnl%={_v(r.total_pnl_pct, '.2f')} "
                    f"max_dd={dd} timeframe={r.timeframe} "
                    f"instr={r.instrument_count} "
                    f"factors=[{fids}]"
                )
    finally:
        db.close()


@cli.command()
@click.argument("factor_id")
@click.argument("new_status")
@_ENV_OPTION
@_DB_DIR_OPTION
def status(
    factor_id: str, new_status: str,
    env_name: str | None, db_dir: str,
) -> None:
    """Change a factor's status (candidate/active/archived)."""
    repo, db = _open_repo(env_name, db_dir)
    try:
        f = repo.get_factor(factor_id)
        old_status = f.status if f else "?"
        repo.set_status(factor_id, new_status)
        click.echo(f"{factor_id}: {old_status} → {new_status}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.argument("factor_id")
@click.option("--timeframe", default=None, help="Filter by timeframe.")
@_ENV_OPTION
@_DB_DIR_OPTION
def metrics(
    factor_id: str, timeframe: str | None,
    env_name: str | None, db_dir: str,
) -> None:
    """Show analysis metrics for a factor."""
    repo, db = _open_repo(env_name, db_dir)
    try:
        results = repo.get_metrics(factor_id, timeframe=timeframe)
        if not results:
            click.echo(f"No metrics found for {factor_id}")
            return

        def _f(v: float | None, w: int = 8, d: int = 4) -> str:
            return f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}"

        click.echo(
            f"{'run_id':<18} {'period':<6} "
            f"{'IC':>8} {'ICIR':>8} {'t(NW)':>8} {'p(NW)':>10} "
            f"{'mono':>6} {'win%':>7} {'IC_lin':>7} "
            f"{'IC_skew':>8} {'IC_kur':>8} {'AR1':>7} {'N':>6}"
        )
        click.echo("-" * 112)
        for m in results:
            wr = f"{m.win_rate * 100:>6.1f}%" if m.win_rate is not None else f"{'-':>7}"
            p_nw = f"{m.p_value_nw:>10.2e}" if m.p_value_nw is not None else f"{'-':>10}"
            n = f"{m.n_samples:>6}" if m.n_samples is not None else f"{'-':>6}"
            click.echo(
                f"{m.run_id:<18} {m.period:<6} "
                f"{_f(m.ic_mean)} {_f(m.icir)} {_f(m.t_stat_nw, 8, 2)} {p_nw} "
                f"{_f(m.monotonicity, 6, 2)} {wr} {_f(m.ic_linearity, 7, 3)} "
                f"{_f(m.ic_skew)} {_f(m.ic_kurtosis)} {_f(m.ic_ar1, 7, 3)} {n}"
            )
    finally:
        db.close()


@cli.command("backtests")
@click.option("--factor-id", default=None, help="Filter by factor_id.")
@click.option("--limit", default=20, type=int, help="Max rows.")
@_ENV_OPTION
@_DB_DIR_OPTION
def backtests(
    factor_id: str | None, limit: int,
    env_name: str | None, db_dir: str,
) -> None:
    """List backtest runs from the registry."""
    from nautilus_quants.alpha.registry.backtest_repository import BacktestRepository
    from nautilus_quants.alpha.registry.database import RegistryDatabase
    from nautilus_quants.alpha.registry.environment import resolve_env

    env = resolve_env(env_name)
    db = RegistryDatabase.for_environment(env, db_dir)
    bt_repo = BacktestRepository(db)
    try:
        runs = bt_repo.list_backtests(factor_id=factor_id, limit=limit)
        if not runs:
            click.echo("(no backtests found)")
            return

        def _f(v: float | None, w: int = 8, d: int = 4) -> str:
            return f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}"

        click.echo(
            f"{'backtest_id':<18} {'strategy':<20} {'timeframe':<10} "
            f"{'instr':>5} {'sharpe':>8} {'pnl%':>8} {'max_dd':>9} "
            f"{'win_rate':>8} {'dur(s)':>7}  {'factors'}"
        )
        click.echo("-" * 125)
        for r in runs:
            wr = f"{r.win_rate * 100:>7.1f}%" if r.win_rate else f"{'-':>8}"
            dd = f"{r.max_drawdown * 100:>8.2f}%" if r.max_drawdown else f"{'-':>9}"
            factors = bt_repo.get_backtest_factors(r.backtest_id)
            fids = ", ".join(bf.factor_id for bf in factors) if factors else "-"
            click.echo(
                f"{r.backtest_id:<18} {r.strategy_name:<20} {r.timeframe:<10} "
                f"{r.instrument_count:>5} {_f(r.sharpe_ratio)} "
                f"{_f(r.total_pnl_pct)} {dd} "
                f"{wr} {r.duration_seconds:>7.1f}  {fids}"
            )
        click.echo(f"({len(runs)} backtests)")
    finally:
        db.close()


@cli.command("export-factors")
@click.option("--context-id", default="", help="Config snapshot ID for variables/parameters.")
@click.option("--method", default="equal", help="Composite weighting (equal/icir_weight).")
@click.option("--top", "top_n", default=30, type=int, help="Max factors.")
@click.option("--transform", default="cs_rank", help="Transform (cs_rank/cs_zscore/raw).")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(path_type=Path))
@_ENV_OPTION
@_DB_DIR_OPTION
def export_factors(
    context_id: str, method: str, top_n: int, transform: str,
    output_path: Path, env_name: str | None, db_dir: str,
) -> None:
    """Export active factors + composite to a factors.yaml file."""
    from nautilus_quants.alpha.registry.export import export_factors_yaml

    repo, db = _open_repo(env_name, db_dir)
    try:
        export_factors_yaml(
            repo, output_path,
            context_id=context_id,
            composite_method=method,
            composite_top_n=top_n,
            composite_transform=transform,
        )
        click.echo(f"Exported to {output_path}")
    finally:
        db.close()


@cli.command()
@click.option(
    "--source-env", default="test",
    type=click.Choice(["test", "dev", "prod"]),
    help="Source registry environment.",
)
@click.option(
    "--target-env", default="dev",
    type=click.Choice(["test", "dev", "prod"]),
    help="Target registry environment.",
)
@click.option(
    "--config", "config_path",
    default="config/scoring.yaml",
    type=click.Path(exists=True, path_type=Path),
    help="Scoring configuration file.",
)
@click.option("--dry-run", is_flag=True, help="Score and rank without migrating.")
@click.option("--skip-corr", is_flag=True, help="Skip correlation computation (fingerprint dedup only).")
@click.option("--max-factors", default=None, type=int, help="Override max factors to promote.")
@_DB_DIR_OPTION
def promote(
    source_env: str,
    target_env: str,
    config_path: Path,
    dry_run: bool,
    skip_corr: bool,
    max_factors: int | None,
    db_dir: str,
) -> None:
    """Score, deduplicate, decorrelate, and promote factors to target env.

    Full pipeline:
      1. Load metrics from source env
      2. Apply hard filters (per-period)
      3. Score factors (5-dimension)
      4. Fingerprint deduplication
      5. Spearman correlation dedup + greedy selection (unless --skip-corr)
      6. Migrate to target env (unless --dry-run)

    \b
    Examples:
      python -m nautilus_quants.alpha promote --source-env test --target-env dev
      python -m nautilus_quants.alpha promote --source-env test --target-env dev --dry-run
      python -m nautilus_quants.alpha promote --source-env test --target-env dev --skip-corr
      python -m nautilus_quants.alpha promote --config config/scoring.yaml --max-factors 50
    """
    from datetime import datetime, timezone

    from nautilus_quants.alpha.registry.database import RegistryDatabase
    from nautilus_quants.alpha.registry.scoring import (
        apply_hard_filters,
        compute_factor_correlation,
        dedup_by_fingerprint,
        greedy_select,
        load_scoring_config,
        load_scoring_data,
        migrate_factors,
        score_factors,
    )

    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        # 0. Load scoring config
        scoring_cfg = load_scoring_config(config_path)
        if max_factors is not None:
            # Override from CLI
            scoring_cfg = scoring_cfg.__class__(
                periods=scoring_cfg.periods,
                hard_filters=scoring_cfg.hard_filters,
                weights=scoring_cfg.weights,
                sub_weights=scoring_cfg.sub_weights,
                dedup=scoring_cfg.dedup,
                promote=scoring_cfg.promote.__class__(
                    max_factors=max_factors,
                    target_status=scoring_cfg.promote.target_status,
                ),
                data=scoring_cfg.data,
            )

        click.echo("=" * 80)
        click.echo("Nautilus Quants - Factor Promotion Pipeline")
        click.echo("=" * 80)
        click.echo(f"  Source: {source_env}.duckdb")
        click.echo(f"  Target: {target_env}.duckdb")
        click.echo(f"  Config: {config_path}")
        click.echo(f"  Periods: {scoring_cfg.periods}")
        click.echo(f"  Dry run: {dry_run}")
        click.echo(f"  Skip corr: {skip_corr}")
        click.echo(f"  Max factors: {scoring_cfg.promote.max_factors}")
        click.echo("=" * 80)
        click.echo()

        # 1. Load data from source
        click.echo("Phase 1: Loading scoring data...")
        source_db = RegistryDatabase.for_environment(source_env, db_dir)
        df = load_scoring_data(source_db, scoring_cfg.periods)

        if df.empty:
            click.echo("Error: No metrics found in source database.", err=True)
            source_db.close()
            sys.exit(1)

        click.echo(f"  Loaded {len(df)} factors with metrics across {scoring_cfg.periods}")

        # 2. Hard filters
        click.echo()
        click.echo("Phase 1.1: Applying hard filters...")
        n_before = len(df)
        df = apply_hard_filters(df, scoring_cfg.hard_filters, scoring_cfg.periods)
        n_after = len(df)
        click.echo(f"  {n_before} → {n_after} factors ({n_before - n_after} eliminated)")

        if df.empty:
            click.echo("Error: No factors passed hard filters.", err=True)
            source_db.close()
            sys.exit(1)

        # Show valid period distribution
        period_counts = df["n_valid_periods"].value_counts().sort_index()
        for n_periods, count in period_counts.items():
            click.echo(f"    {n_periods} valid periods: {count} factors")

        # 3. Fingerprint dedup
        click.echo()
        click.echo("Phase 2.1: Fingerprint deduplication...")
        n_before_dedup = len(df)
        df = dedup_by_fingerprint(
            df, scoring_cfg.periods,
            threshold=scoring_cfg.dedup.fingerprint_threshold,
        )
        n_removed = n_before_dedup - len(df)
        click.echo(f"  {n_before_dedup} → {len(df)} factors ({n_removed} duplicates removed)")

        # 4. Scoring
        click.echo()
        click.echo("Phase 1.2: Computing 5-dimension scores...")
        df = score_factors(df, scoring_cfg)

        # Print top factors table
        display_cols = ["final_score", "avg_period_score", "consistency",
                        "turnover_friendliness", "n_valid_periods"]
        available = [c for c in display_cols if c in df.columns]

        click.echo()
        click.echo(f"  Top {min(30, len(df))} factors by final_score:")
        click.echo(f"  {'factor_id':<40} {'score':>7} {'avg_pp':>7} {'cons':>6} "
                   f"{'turn':>6} {'#pd':>4}")
        click.echo("  " + "-" * 72)
        for i, (fid, row) in enumerate(df.head(30).iterrows()):
            click.echo(
                f"  {fid:<40} {row.get('final_score', 0):>7.4f} "
                f"{row.get('avg_period_score', 0):>7.4f} "
                f"{row.get('consistency', 0):>6.3f} "
                f"{row.get('turnover_friendliness', 0):>6.3f} "
                f"{int(row.get('n_valid_periods', 0)):>4}"
            )

        # 5. Correlation-based greedy selection
        if not skip_corr:
            click.echo()
            click.echo("Phase 2.2: Computing factor correlations...")
            candidate_ids = df.index.tolist()
            try:
                corr_matrix = compute_factor_correlation(candidate_ids, scoring_cfg)

                if not corr_matrix.empty:
                    click.echo(
                        f"  Correlation matrix: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}"
                    )

                    # Save correlation matrix CSV
                    corr_dir = Path("logs/scoring")
                    corr_dir.mkdir(parents=True, exist_ok=True)
                    corr_csv = corr_dir / f"correlation_matrix_{timestamp}.csv"
                    corr_matrix.to_csv(corr_csv)
                    click.echo(f"  Saved: {corr_csv}")

                    # Save heatmap
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        fig, ax = plt.subplots(figsize=(20, 16))
                        sns.heatmap(
                            corr_matrix, vmin=-1, vmax=1, center=0,
                            cmap="RdBu_r", ax=ax,
                            xticklabels=True, yticklabels=True,
                        )
                        ax.set_title("Factor Spearman Correlation (cross-sectional avg)")
                        plt.tight_layout()
                        heatmap_path = corr_dir / f"correlation_heatmap_{timestamp}.png"
                        fig.savefig(heatmap_path, dpi=100)
                        plt.close(fig)
                        click.echo(f"  Saved: {heatmap_path}")
                    except Exception as e:
                        click.echo(f"  Warning: Heatmap generation failed: {e}", err=True)

                    # Greedy selection
                    click.echo()
                    click.echo("Phase 2.3: Greedy selection (max corr "
                               f"≤ {scoring_cfg.dedup.max_corr})...")
                    selected_ids = greedy_select(
                        df, corr_matrix,
                        max_corr=scoring_cfg.dedup.max_corr,
                        max_factors=scoring_cfg.promote.max_factors,
                    )
                    click.echo(f"  Selected {len(selected_ids)} factors")
                else:
                    click.echo("  Warning: Empty correlation matrix, skipping greedy selection")
                    selected_ids = df.index.tolist()[:scoring_cfg.promote.max_factors]
            except Exception as e:
                click.echo(f"  Warning: Correlation computation failed: {e}", err=True)
                click.echo("  Falling back to score-only selection")
                selected_ids = df.index.tolist()[:scoring_cfg.promote.max_factors]
        else:
            click.echo()
            click.echo("Phase 2.2: Skipping correlation (--skip-corr)")
            selected_ids = df.index.tolist()[:scoring_cfg.promote.max_factors]

        # Save scores CSV
        scores_dir = Path("logs/scoring")
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_csv = scores_dir / f"factor_scores_{timestamp}.csv"
        score_cols = ["final_score", "avg_period_score", "consistency",
                      "turnover_friendliness", "n_valid_periods"]
        save_cols = [c for c in score_cols if c in df.columns]
        df[save_cols].to_csv(scores_csv)
        click.echo(f"\n  Scores saved: {scores_csv}")

        # Summary
        click.echo()
        click.echo("=" * 80)
        click.echo("PROMOTION SUMMARY")
        click.echo("=" * 80)
        click.echo(f"  Candidates after hard filter: {n_after}")
        click.echo(f"  After fingerprint dedup: {n_after - n_removed}")
        click.echo(f"  Final selected: {len(selected_ids)}")
        click.echo()

        # Print selected factors
        click.echo(f"  Selected factors ({len(selected_ids)}):")
        for i, fid in enumerate(selected_ids, 1):
            score = df.loc[fid, "final_score"] if fid in df.index else 0
            click.echo(f"    {i:3d}. {fid:<40} score={score:.4f}")

        # 6. Migrate (unless dry run)
        if dry_run:
            click.echo()
            click.echo("  [DRY RUN] No migration performed.")
        else:
            click.echo()
            click.echo(f"Phase 3: Migrating {len(selected_ids)} factors "
                       f"to {target_env}.duckdb...")
            target_db = RegistryDatabase.for_environment(target_env, db_dir)
            try:
                counts = migrate_factors(
                    source_db, target_db, selected_ids,
                    target_status=scoring_cfg.promote.target_status,
                )
                click.echo(f"  Migrated: {counts['factors']} factors, "
                           f"{counts['metrics']} metrics, "
                           f"{counts['configs']} config snapshots")
            finally:
                target_db.close()

        source_db.close()
        duration = time.time() - start_time
        click.echo()
        click.echo(f"  Duration: {duration:.2f}s")
        click.echo("=" * 80)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
