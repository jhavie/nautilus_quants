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
@click.option("--force-reanalyze", is_flag=True,
              help="Re-analyze even if metrics already exist for the expression")
@click.option("--workers", type=int, default=None,
              help="Max parallel workers for alphalens (default: min(factors, 4))")
def analyze(
    config_file: Path, verbose: bool, quiet: bool,
    no_registry: bool, env_name: str | None, force_reanalyze: bool,
    workers: int | None,
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

        # 6b. Skip factors with existing metrics (default behavior)
        registry_enabled = config.registry_enabled and not no_registry
        if registry_enabled and not force_reanalyze:
            from nautilus_quants.alpha.registry.database import RegistryDatabase
            from nautilus_quants.alpha.registry.environment import resolve_env
            from nautilus_quants.factors.expression.normalize import (
                expression_hash as _expr_hash,
            )

            _env = resolve_env(env_name, config.registry_env)
            _db = RegistryDatabase.for_environment(_env, config.registry_db_dir)
            already: set[str] = set()
            for fname in list(factor_series.keys()):
                fdef = factor_config.get_factor(fname)
                if fdef is None:
                    continue
                try:
                    h = _expr_hash(fdef.expression)
                except Exception:
                    continue
                row = _db.fetch_one(
                    "SELECT f.factor_id FROM factors f "
                    "JOIN alpha_analysis_metrics m "
                    "ON f.factor_id = m.factor_id "
                    "WHERE f.expression_hash = ? LIMIT 1",
                    [h],
                )
                if row is not None:
                    already.add(fname)
            _db.close()

            if already:
                factor_series = {
                    k: v for k, v in factor_series.items()
                    if k not in already
                }
                if not quiet:
                    click.echo(
                        f"  Skipped {len(already)} factors "
                        f"with existing metrics"
                    )
            if not factor_series:
                if not quiet:
                    click.echo(
                        "All factors already analyzed. Nothing to do."
                    )
                return

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
        n_workers = workers if workers is not None else min(len(factor_series), 4)
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
    from nautilus_quants.alpha.formatters import print_register_result
    from nautilus_quants.factors.config import load_factor_config

    try:
        config = load_factor_config(config_file)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    repo, db = _open_repo(env_name, db_dir)
    try:
        new, updated, unchanged, duplicate = repo.register_factors_from_config(
            config,
        )
        print_register_result(new, updated, unchanged, duplicate)
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
    from nautilus_quants.alpha.formatters import print_factor_list

    repo, db = _open_repo(env_name, db_dir)
    try:
        factors = repo.list_factors(
            status=status, source=source, prototype=prototype, limit=limit,
        )
        if not factors:
            click.echo("(no factors found)")
            return

        has_scores = any(
            f.parameters.get("promote_score") is not None for f in factors
        )
        print_factor_list(factors, has_scores)
    finally:
        db.close()


@cli.command()
@click.argument("factor_id", required=False, default=None)
@click.option("--prototype", "proto_name", default=None,
              help="Inspect all factors sharing this prototype.")
@_ENV_OPTION
@_DB_DIR_OPTION
def inspect(
    factor_id: str | None, proto_name: str | None,
    env_name: str | None, db_dir: str,
) -> None:
    """Inspect a factor or prototype group.

    With FACTOR_ID: show factor details + metrics + backtests.
    With --prototype: show template expression + all parameter variants.
    """
    if not factor_id and not proto_name:
        click.echo("Error: provide FACTOR_ID or --prototype NAME", err=True)
        sys.exit(1)

    repo, db = _open_repo(env_name, db_dir)
    try:
        if proto_name:
            _inspect_prototype(repo, proto_name)
        else:
            _inspect_factor(repo, db, factor_id)  # type: ignore[arg-type]
    finally:
        db.close()


def _inspect_prototype(repo, proto_name: str) -> None:
    """Show prototype template + parameter variants."""
    from nautilus_quants.alpha.formatters import print_prototype_group
    from nautilus_quants.factors.expression.normalize import expression_template

    factors = repo.list_factors(prototype=proto_name)
    if not factors:
        click.echo(f"No factors with prototype '{proto_name}'")
        return

    sources = {f.source for f in factors if f.source}

    # Compute template from first factor
    template = None
    try:
        tmpl, _ = expression_template(factors[0].expression)
        template = tmpl
    except Exception:
        pass

    # Collect parameter names from all factors
    all_params: list[tuple[str, dict[str, float], str]] = []
    for f in factors:
        try:
            _, vals = expression_template(f.expression)
            params = {f"p{i}": v for i, v in enumerate(vals)}
        except Exception:
            params = {}
        all_params.append((f.factor_id, params, f.status))

    p_keys = sorted({k for _, p, _ in all_params for k in p_keys_of(p)})
    print_prototype_group(proto_name, sources, template, all_params, p_keys)


def p_keys_of(params: dict) -> list[str]:
    """Extract sorted p0, p1, ... keys from a params dict."""
    return sorted(
        (k for k in params if k.startswith("p") and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )


def _inspect_factor(repo, db, factor_id: str) -> None:
    """Show single factor details + metrics + backtests."""
    from nautilus_quants.alpha.formatters import print_factor_detail
    from nautilus_quants.alpha.registry.backtest_repository import (
        BacktestRepository,
    )

    f = repo.get_factor(factor_id)
    if f is None:
        click.echo(f"Error: factor not found: {factor_id}", err=True)
        sys.exit(1)

    metrics = repo.get_metrics(factor_id)

    bt_repo = BacktestRepository(db)
    runs = bt_repo.list_backtests(factor_id=factor_id)
    backtests = None
    if runs:
        backtests = [
            (r, bt_repo.get_backtest_factors(r.backtest_id))
            for r in runs
        ]

    print_factor_detail(f, metrics=metrics or None, backtests=backtests)


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
    from nautilus_quants.alpha.formatters import print_metrics_table

    repo, db = _open_repo(env_name, db_dir)
    try:
        results = repo.get_metrics(factor_id, timeframe=timeframe)
        if not results:
            click.echo(f"No metrics found for {factor_id}")
            return

        print_metrics_table(factor_id, results)
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
    from nautilus_quants.alpha.formatters import print_backtests_table
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

        bt_with_factors = [
            (r, bt_repo.get_backtest_factors(r.backtest_id))
            for r in runs
        ]
        print_backtests_table(bt_with_factors)
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
    default="config/examples/scoring.yaml",
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
      python -m nautilus_quants.alpha promote --config config/examples/scoring.yaml --max-factors 50
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

        from nautilus_quants.alpha.formatters import (
            console as _console,
            print_promote_header,
            print_promote_summary,
            print_selected_factors,
            print_top_scores,
        )

        print_promote_header(
            source_env, target_env, str(config_path),
            scoring_cfg.periods, dry_run, skip_corr,
            scoring_cfg.promote.max_factors,
        )

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
        print_top_scores(df, n=30)

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
        score_cols = [
            "final_score", "avg_period_score",
            "pred_score", "stab_score", "mono_score",
            "consistency", "turnover_friendliness",
            "avg_icir", "avg_t_stat_nw", "avg_win_rate",
            "avg_ic_linearity", "avg_monotonicity",
            "n_valid_periods",
        ]
        save_cols = [c for c in score_cols if c in df.columns]
        df[save_cols].to_csv(scores_csv)
        click.echo(f"\n  Scores saved: {scores_csv}")

        # Summary
        print_selected_factors(selected_ids, df)

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
                    scores=df,
                )
                click.echo(f"  Migrated: {counts['factors']} factors, "
                           f"{counts['metrics']} metrics, "
                           f"{counts['configs']} config snapshots")
            finally:
                target_db.close()

        source_db.close()
        duration = time.time() - start_time
        print_promote_summary(
            n_after, n_after - n_removed, len(selected_ids),
            duration, dry_run,
        )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@_ENV_OPTION
@_DB_DIR_OPTION
def audit(env_name: str | None, db_dir: str) -> None:
    """Audit the factor registry for duplicates and prototype issues."""
    from nautilus_quants.alpha.formatters import (
        console as _console,
        print_audit_duplicates,
        print_audit_template_groups,
    )
    from nautilus_quants.alpha.registry.audit import (
        find_expression_duplicates,
        suggest_prototype_groups,
    )

    repo, db = _open_repo(env_name, db_dir)
    try:
        factors = repo.list_factors()
        _console.print(f"Total factors: [bold]{len(factors)}[/bold]")

        dup_groups = find_expression_duplicates(repo)
        print_audit_duplicates(dup_groups)

        groups = suggest_prototype_groups(repo)
        print_audit_template_groups(groups)
    finally:
        db.close()


@cli.command()
@click.option("--execute", is_flag=True, default=False, help="Actually apply.")
@_ENV_OPTION
@_DB_DIR_OPTION
def backfill(execute: bool, env_name: str | None, db_dir: str) -> None:
    """Backfill expression_hash, prototype, and parameters for all factors.

    For each factor:
    1. Compute expression_hash (for dedup).
    2. Fix prototype from builtin libraries (ta_factors.py).
    3. Extract numbers via expression_template → {p0, p1, ...} parameters.
    """
    from nautilus_quants.alpha.registry.audit import backfill_factors

    from nautilus_quants.alpha.formatters import print_backfill_result

    dry_run = not execute
    repo, db = _open_repo(env_name, db_dir)
    try:
        counts = backfill_factors(repo, dry_run=dry_run)
        print_backfill_result(counts, dry_run)
    finally:
        db.close()


@cli.command()
@click.option(
    "--keep-source", default=None,
    help="Preferred source to keep (e.g. alpha101).",
)
@click.option("--dry-run", is_flag=True, default=True, help="Preview only.")
@click.option("--execute", is_flag=True, default=False, help="Actually delete.")
@_ENV_OPTION
@_DB_DIR_OPTION
def dedup(
    keep_source: str | None,
    dry_run: bool,
    execute: bool,
    env_name: str | None,
    db_dir: str,
) -> None:
    """Remove duplicate factors by expression hash."""
    from nautilus_quants.alpha.registry.audit import dedup_factors

    from nautilus_quants.alpha.formatters import print_dedup_result

    actual_dry_run = not execute
    repo, db = _open_repo(env_name, db_dir)
    try:
        removed = dedup_factors(
            repo, keep_source=keep_source, dry_run=actual_dry_run,
        )
        print_dedup_result(removed, actual_dry_run)
    finally:
        db.close()


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("--rounds", default=5, type=int, help="Number of mining rounds.")
@click.option("--factors-per-round", default=None, type=int,
              help="Factors to generate per round (default: from config or 8).")
@click.option("--model", default=None, help="Claude model (sonnet/opus, default: from config or sonnet).")
@click.option("--hypothesis", default=None, help="Initial hypothesis direction for factor generation.")
@click.option("--no-analyze", is_flag=True, help="Generate factors only, skip IC analysis.")
def mine(
    config_file: Path,
    rounds: int,
    factors_per_round: int | None,
    model: str | None,
    hypothesis: str | None,
    no_analyze: bool,
) -> None:
    """LLM-driven alpha factor mining via Claude Code CLI.

    Generates factor expressions using Claude, validates them against
    the expression parser, and optionally runs alphalens analysis.

    \b
    Examples:
      python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml
      python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml --rounds 3
      python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml --hypothesis "volume divergence"
      python -m nautilus_quants.alpha mine config/cs/alpha_101.yaml --no-analyze --rounds 1
    """
    from nautilus_quants.alpha.mining.agent.miner import AlphaMiner, MiningConfig

    config = MiningConfig.from_yaml(
        config_file,
        factors_per_round=factors_per_round,
        model=model,
        hypothesis=hypothesis,
        auto_analyze=not no_analyze,
    )
    miner = AlphaMiner(config)
    miner.run(rounds=rounds)


if __name__ == "__main__":
    cli()
