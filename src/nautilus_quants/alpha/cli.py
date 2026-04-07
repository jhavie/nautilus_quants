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

                # Register only factors that have analysis results
                repo.register_factors_from_config(
                    factor_config, only_names=set(ic_results.keys()),
                )

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
@click.option("--transform", default="normalize", help="Transform (normalize/cs_rank/cs_zscore/raw).")
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
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
def regime(config_file: Path, verbose: bool, quiet: bool) -> None:
    """Run regime-conditional factor analysis (Jump Model vs EMA comparison).

    Detects market regimes using both Jump Model and EMA, computes per-factor
    per-regime IC/ICIR, and generates comparison charts (timeline, ICIR bars,
    L/S equity curves).
    """
    from nautilus_quants.alpha.analysis.config import load_analysis_config
    from nautilus_quants.alpha.analysis.evaluator import FactorEvaluator
    from nautilus_quants.alpha.data_loader import CatalogDataLoader
    from nautilus_quants.backtest.utils.reporting import (
        create_output_directory,
        generate_run_id,
    )
    from nautilus_quants.factors.config import load_factor_config

    start_time = time.time()

    try:
        # 1. Load config and validate regime section
        config = load_analysis_config(config_file)
        if config.regime is None:
            click.echo(
                "Error: No 'regime' section in config. "
                "Add regime: {...} to enable regime analysis.",
                err=True,
            )
            sys.exit(1)

        rc = config.regime

        if not quiet:
            click.echo("=" * 80)
            click.echo("Nautilus Quants - Regime Analysis (JM vs EMA)")
            click.echo("=" * 80)
            click.echo(f"Config: {config_file}")
            click.echo(
                f"  Regime instrument: {rc.regime_instrument}"
            )
            click.echo(
                f"  JM: n_states={rc.jump_model.n_states} "
                f"λ={rc.jump_model.jump_penalty} "
                f"features={rc.jump_model.feature_set}"
            )
            click.echo(f"  EMA: span={rc.ema.span}")
            click.echo("=" * 80)
            click.echo()

        # 2. Load bar data
        if not quiet:
            click.echo(f"Loading bar data from {config.catalog_path}...")

        loader = CatalogDataLoader(config.catalog_path, config.bar_spec)
        bars_by_instrument = loader.load_bars(config.instrument_ids)

        loaded_count = sum(len(bars) for bars in bars_by_instrument.values())
        if loaded_count == 0:
            click.echo(
                "Error: No bar data loaded.", err=True,
            )
            sys.exit(1)

        if not quiet:
            click.echo(
                f"Loaded {loaded_count} bars across "
                f"{sum(1 for b in bars_by_instrument.values() if b)} "
                f"instruments"
            )

        # 3. Load factors (with cache support, same as analyze)
        factor_config = load_factor_config(config.factor_config_path)
        evaluator = FactorEvaluator(factor_config)

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

        if use_cache:
            if not quiet:
                click.echo(
                    f"Loading factors from cache: "
                    f"{config.factor_cache_path}"
                )
            factor_series = load_as_factor_series(config.factor_cache_path)
            import pandas as _pd
            pricing = _pd.DataFrame(
                {
                    inst_id: CatalogDataLoader.bars_to_dataframe(bars)["close"]
                    for inst_id, bars in bars_by_instrument.items()
                    if bars
                }
            )
        else:
            if not quiet:
                click.echo("Computing factors...")
            factor_series, pricing = evaluator.evaluate(bars_by_instrument)
            if config.factor_cache_path and factor_series:
                save_factor_cache(
                    factor_series,
                    config.factor_cache_path,
                    factor_config_path=config.factor_config_path,
                    config_hash=config_hash,
                )
                if not quiet:
                    click.echo(
                        f"  Cache saved: {config.factor_cache_path}"
                    )

        if not factor_series:
            click.echo(
                "Error: No factor values computed.", err=True,
            )
            sys.exit(1)

        # 4. Filter factors if specified
        if config.factors:
            factor_series = {
                k: v for k, v in factor_series.items()
                if k in config.factors
            }
        if not factor_series:
            click.echo("Error: No matching factors.", err=True)
            sys.exit(1)

        # 5. Convert factor_series to panel DataFrames (T × N)
        factor_dfs: dict[str, pd.DataFrame] = {}
        for name, series in factor_series.items():
            factor_dfs[name] = series.unstack("asset")

        if not quiet:
            click.echo(
                f"Analyzing {len(factor_dfs)} factors "
                f"across {len(pricing.columns)} instruments..."
            )

        # 6. Run comparative regime analysis
        from nautilus_quants.alpha.regime.regime_ic_analysis import (
            run_comparative_analysis,
        )

        if not quiet:
            click.echo("Running regime detection and IC analysis...")

        report = run_comparative_analysis(
            factor_dfs=factor_dfs,
            pricing=pricing,
            regime_instrument=rc.regime_instrument,
            n_states=rc.jump_model.n_states,
            jump_penalty=rc.jump_model.jump_penalty,
            feature_set=rc.jump_model.feature_set,
            ema_span=rc.ema.span,
            forward_period=rc.forward_period,
            min_obs=rc.min_obs,
            min_weight=rc.min_weight,
            refit_window=rc.jump_model.refit_window,
            refit_interval=rc.jump_model.refit_interval,
            min_train=rc.jump_model.min_train,
        )

        # 7. Generate output directory and charts
        run_id = generate_run_id()
        output_dir = create_output_directory(
            config.output_dir, f"regime_{run_id}",
        )

        from nautilus_quants.alpha.regime.charts import (
            chart_ls_equity_curves,
            chart_regime_icir_bars,
            chart_regime_timeline,
        )

        if not quiet:
            click.echo("Generating charts...")

        chart_regime_timeline(report, output_dir / "regime_timeline.png")
        chart_regime_icir_bars(report, output_dir / "regime_icir_comparison.png")

        # L/S equity curves need forward returns + factor directions
        if rc.forward_period == 1:
            fwd_returns = pricing.pct_change(fill_method=None).shift(-1)
        else:
            fwd_returns = pricing.pct_change(
                periods=rc.forward_period, fill_method=None,
            ).shift(-rc.forward_period)

        # Build signed equal weights from factor config composite
        import yaml as _yaml_dir

        equal_weights: dict[str, float] = {}
        with open(config.factor_config_path, encoding="utf-8") as _f:
            raw_fc = _yaml_dir.safe_load(_f)
        composite_raw = raw_fc.get("composite", {}) or {}
        composite_weights = composite_raw.get("weights", {}) or {}
        if composite_weights:
            # Normalize to sum(abs)=1, preserving signs
            total_abs = sum(abs(v) for v in composite_weights.values())
            for name, w in composite_weights.items():
                equal_weights[name] = w / total_abs if total_abs > 0 else 0
        # Default +1/n for factors not in composite
        n_factors = len(report.factor_names)
        for name in report.factor_names:
            equal_weights.setdefault(name, 1.0 / n_factors)

        chart_ls_equity_curves(
            report, factor_dfs, fwd_returns,
            output_dir / "ls_equity_curves.png",
            equal_weights=equal_weights,
        )

        # 8. Export regime_config.yaml (weights + schedule)
        if rc.export_weights:
            from datetime import datetime as _dt

            import yaml as _yaml

            from nautilus_quants.alpha.regime.regime_ic_analysis import (
                extract_regime_schedule,
            )

            def _round_weights(
                wm: dict[str, dict[str, float]],
            ) -> dict[str, dict[str, float]]:
                return {
                    r: {k: round(v, 4) for k, v in ws.items()}
                    for r, ws in wm.items()
                }

            # Read default composite weights from factor config
            default_weights = {}
            composite_raw = raw_fc.get("composite", {}) or {}
            cw = composite_raw.get("weights", {}) or {}
            if cw:
                total_abs = sum(abs(v) for v in cw.values())
                if total_abs > 0:
                    default_weights = {
                        k: round(v / total_abs, 4) for k, v in cw.items()
                    }

            regime_data = {
                "metadata": {
                    "generated": _dt.now().isoformat(
                        timespec="seconds",
                    ),
                    "source_config": str(config_file),
                    "factor_config": str(config.factor_config_path),
                    "regime_instrument": rc.regime_instrument,
                    "data_range": [
                        str(report.btc_close.index.min().date()),
                        str(report.btc_close.index.max().date()),
                    ],
                    "default_weights": default_weights,
                },
                "jump_model": {
                    "params": {
                        "n_states": rc.jump_model.n_states,
                        "jump_penalty": rc.jump_model.jump_penalty,
                        "refit_interval": rc.jump_model.refit_interval,
                        "min_train": rc.jump_model.min_train,
                        "feature_set": rc.jump_model.feature_set,
                    },
                    "weights": _round_weights(report.jm_weights),
                    "schedule": extract_regime_schedule(
                        report.jm_regime,
                    ),
                },
                "ema": {
                    "params": {"span": rc.ema.span},
                    "weights": _round_weights(report.ema_weights),
                    "schedule": extract_regime_schedule(
                        report.ema_regime,
                    ),
                },
            }
            config_path = output_dir / "regime_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                _yaml.dump(
                    regime_data, f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            if not quiet:
                jm_n = len(regime_data["jump_model"]["schedule"])
                ema_n = len(regime_data["ema"]["schedule"])
                click.echo(
                    f"  Regime config: {config_path} "
                    f"(JM {jm_n} transitions, EMA {ema_n} transitions)"
                )

        # 9. Print summary table
        duration = time.time() - start_time

        if not quiet:
            click.echo()
            click.echo("Regime Detection Summary")

            # JM stats
            jm_counts = report.jm_regime.value_counts(normalize=True)
            jm_sw = int(
                (report.jm_regime != report.jm_regime.shift()).sum()
            )
            jm_parts = []
            for label in ["bear", "neutral", "bull"]:
                if label in jm_counts.index:
                    jm_parts.append(f"{label}={jm_counts[label]:.1%}")
            click.echo(
                f"  JM (n={rc.jump_model.n_states}, "
                f"λ={rc.jump_model.jump_penalty}): "
                f"{' '.join(jm_parts)} switches={jm_sw}"
            )

            # EMA stats
            ema_counts = report.ema_regime.value_counts(normalize=True)
            ema_sw = int(
                (report.ema_regime != report.ema_regime.shift()).sum()
            )
            ema_parts = []
            for label in ["bear", "bull"]:
                if label in ema_counts.index:
                    ema_parts.append(f"{label}={ema_counts[label]:.1%}")
            click.echo(
                f"  EMA (span={rc.ema.span}): "
                f"{' '.join(ema_parts)} switches={ema_sw}"
            )

            # ICIR comparison table
            click.echo()
            jm_regimes = ["bear", "neutral", "bull"]
            ema_regimes = ["bear", "bull"]

            header = (
                f"{'Factor':<30} "
                + "".join(f"{'JM '+r:>10}" for r in jm_regimes)
                + "".join(f"{'EMA '+r:>10}" for r in ema_regimes)
                + f"{'JM Spread':>10}"
            )
            click.echo(header)
            click.echo("-" * len(header))

            jm_spreads = []
            ema_spreads = []
            for name in report.factor_names:
                jm_r = next(
                    (r for r in report.jm_results
                     if r.factor_name == name), None,
                )
                ema_r = next(
                    (r for r in report.ema_results
                     if r.factor_name == name), None,
                )
                if jm_r is None or ema_r is None:
                    continue

                jm_vals = " ".join(
                    f"{jm_r.icir(r):>10.4f}" for r in jm_regimes
                )
                ema_vals = " ".join(
                    f"{ema_r.icir(r):>10.4f}" for r in ema_regimes
                )
                jm_sp = jm_r.icir_spread()
                ema_sp = ema_r.icir_spread()
                jm_spreads.append(jm_sp)
                ema_spreads.append(ema_sp)

                click.echo(
                    f"{name:<30} {jm_vals} {ema_vals} "
                    f"{jm_sp:>10.4f}"
                )

            click.echo("-" * len(header))
            avg_jm = (
                sum(jm_spreads) / len(jm_spreads) if jm_spreads else 0.0
            )
            avg_ema = (
                sum(ema_spreads) / len(ema_spreads) if ema_spreads else 0.0
            )
            ratio = avg_jm / avg_ema if avg_ema > 0.001 else 0.0
            click.echo(
                f"Average ICIR Spread: "
                f"JM={avg_jm:.4f}  EMA={avg_ema:.4f}  "
                f"(JM/EMA={ratio:.1f}x)"
            )

            click.echo()
            click.echo("=" * 80)
            click.echo(f"  Run ID: regime_{run_id}")
            click.echo(f"  Duration: {duration:.2f}s")
            click.echo(f"  Output: {output_dir}")
            click.echo("=" * 80)

        sys.exit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
