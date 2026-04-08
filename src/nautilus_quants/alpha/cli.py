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
        evaluator = FactorEvaluator(factor_config, analysis_config=config)

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

            try:
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
            except Exception:
                # expression_hash column may not exist in older DBs — skip dedup
                already = set()
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
        skipped_factors: dict[str, str] = {}  # name → reason

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
                    skipped_factors[name] = result
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
        if ic_results or skipped_factors:
            report_gen.generate_summary(
                ic_results, output_dir,
                factor_series=factor_series,
                skipped_factors=skipped_factors,
            )

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
                reg_result = repo.register_factors_from_config(
                    factor_config, only_names=set(ic_results.keys()),
                )

                # Save analysis metrics
                all_metrics = []
                for fname, ic_df in ic_results.items():
                    fid = reg_result.name_map.get(
                        fname,
                        generate_factor_id(factor_config.source, fname),
                    )
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
        result = repo.register_factors_from_config(config)
        print_register_result(
            result.new, result.updated, result.unchanged, result.renamed,
        )
    finally:
        db.close()


@cli.command("list")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--source", default=None, help="Filter by source.")
@click.option("--prototype", default=None, help="Filter by prototype.")
@click.option("--tag", default=None, help="Filter by tag.")
@click.option("--limit", default=None, type=int, help="Max rows.")
@_ENV_OPTION
@_DB_DIR_OPTION
def list_factors(
    status: str | None,
    source: str | None,
    prototype: str | None,
    tag: str | None,
    limit: int | None,
    env_name: str | None,
    db_dir: str,
) -> None:
    """List factors in the registry."""
    from nautilus_quants.alpha.formatters import print_factor_list

    repo, db = _open_repo(env_name, db_dir)
    try:
        factors = repo.list_factors(
            status=status, source=source, prototype=prototype,
            tag=tag, limit=limit,
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


@cli.command("config")
@click.argument("backtest_id")
@click.option(
    "--type",
    "config_type",
    default="backtest",
    type=click.Choice(["backtest", "factors", "all"]),
    help="Config type to show.",
)
@_ENV_OPTION
@_DB_DIR_OPTION
def config_cmd(
    backtest_id: str,
    config_type: str,
    env_name: str | None,
    db_dir: str,
) -> None:
    """Show config snapshot linked to a backtest run."""
    import json as json_mod

    from nautilus_quants.alpha.registry.backtest_repository import BacktestRepository
    from nautilus_quants.alpha.registry.database import RegistryDatabase
    from nautilus_quants.alpha.registry.environment import resolve_env
    from nautilus_quants.alpha.registry.repository import FactorRepository

    env = resolve_env(env_name)
    db = RegistryDatabase.for_environment(env, db_dir)
    bt_repo = BacktestRepository(db)
    repo = FactorRepository(db)
    try:
        run = bt_repo.get_backtest(backtest_id)
        if run is None:
            click.echo(f"Backtest not found: {backtest_id}", err=True)
            raise SystemExit(1)

        ids: list[tuple[str, str]] = []
        if config_type in ("backtest", "all") and run.config_id:
            ids.append(("backtest", run.config_id))
        if config_type in ("factors", "all") and run.factor_config_id:
            ids.append(("factors", run.factor_config_id))

        if not ids:
            click.echo("(no config snapshots linked)")
            return

        for label, cid in ids:
            snap = repo.get_config_snapshot(cid)
            if snap is None:
                click.echo(f"[{label}] config_id={cid} — not found")
                continue
            click.echo(f"── {label} (config_id={cid}) ──")
            click.echo(json_mod.dumps(snap.config_json, indent=2, ensure_ascii=False))
            click.echo()
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


@cli.command()
@click.option(
    "--config", "config_path",
    default="config/examples/scoring.yaml",
    type=click.Path(exists=True, path_type=Path),
    help="Scoring configuration file (all behavior driven by config).",
)
def promote(config_path: Path) -> None:
    """Config-driven factor promote pipeline.

    \b
    All behavior controlled by scoring.yaml (each layer has enabled toggle):
      [hard_filters]       — eliminate garbage factors
      [scoring]            — 4D quality ranking
      [dedup]              — fingerprint + correlation greedy
      [clustering]         — HDBSCAN + PCA → Super Alpha
      [orthogonalization]  — Löwdin weight optimization
      [migrate]            — promote.enabled controls DB migration

    \b
    Examples:
      python -m nautilus_quants.alpha promote --config config/examples/scoring.yaml
    """
    import warnings
    from datetime import datetime, timezone

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from nautilus_quants.alpha.registry.database import RegistryDatabase
    from nautilus_quants.alpha.registry.scoring import (
        apply_hard_filters,
        compute_factor_correlation,
        dedup_by_fingerprint,
        greedy_select,
        load_scoring_config,
        load_scoring_data,
        migrate_factors,
        retire_evicted_factors,
        score_factors,
    )

    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        scoring_cfg = load_scoring_config(config_path)

        from nautilus_quants.alpha.formatters import (
            print_promote_summary,
            print_selected_factors,
            print_top_scores,
        )

        source_db_path = scoring_cfg.promote.source_db_path
        target_db_path = scoring_cfg.promote.target_db_path
        competitive = scoring_cfg.promote.competitive
        diag_dir = Path(scoring_cfg.diagnostics.output_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)

        if not source_db_path:
            click.echo("Error: promote.source_db_path required.", err=True)
            sys.exit(1)

        click.echo(f"Config: {config_path}")
        click.echo(f"Source: {source_db_path}")
        if scoring_cfg.promote.enabled:
            click.echo(f"Target: {target_db_path}")
        click.echo()

        # ── Load ──
        click.echo("Loading scoring data...")
        source_db = RegistryDatabase(Path(source_db_path))
        df = load_scoring_data(source_db, scoring_cfg.periods)

        if df.empty:
            click.echo("Error: No metrics found.", err=True)
            source_db.close()
            sys.exit(1)

        n_loaded = len(df)
        n_after_filter = n_loaded
        n_after_dedup = n_loaded
        click.echo(f"  Loaded {n_loaded} factors")

        # ── [hard_filters] ──
        click.echo()
        if scoring_cfg.hard_filters.enabled:
            click.echo("[hard_filters]...")
            n_before = len(df)
            df = apply_hard_filters(
                df, scoring_cfg.hard_filters, scoring_cfg.periods,
            )
            n_after_filter = len(df)
            click.echo(
                f"  {n_before} → {n_after_filter} "
                f"({n_before - n_after_filter} eliminated)",
            )
            if df.empty:
                click.echo("Error: No factors passed.", err=True)
                source_db.close()
                sys.exit(1)
            period_counts = df["n_valid_periods"].value_counts().sort_index()
            for n_p, cnt in period_counts.items():
                click.echo(f"    {n_p} valid periods: {cnt} factors")
        else:
            click.echo("[hard_filters] — SKIPPED")

        # ── [scoring] ──
        click.echo()
        if scoring_cfg.weights.enabled:
            click.echo("[scoring] (4-dimension)...")
            df = score_factors(df, scoring_cfg)
            print_top_scores(df, n=30)
        else:
            click.echo("[scoring] — SKIPPED")

        # ── [dedup] ──
        click.echo()
        corr_matrix = None
        factor_panels: dict[str, object] | None = None
        if scoring_cfg.dedup.enabled:
            # Fingerprint dedup
            click.echo("[dedup] Fingerprint dedup...")
            n_before_fp = len(df)
            df = dedup_by_fingerprint(
                df, scoring_cfg.periods,
                threshold=scoring_cfg.dedup.fingerprint_threshold,
            )
            n_after_dedup = len(df)
            click.echo(
                f"  {n_before_fp} → {n_after_dedup} "
                f"({n_before_fp - n_after_dedup} duplicates)",
            )

            # Load target existing factors
            from nautilus_quants.alpha.registry.repository import FactorRepository

            target_existing_ids: list[str] = []
            if target_db_path:
                target_db_query = RegistryDatabase(Path(target_db_path))
                try:
                    tgt_repo = FactorRepository(target_db_query)
                    existing = tgt_repo.list_factors(status="active")
                    target_existing_ids = [f.factor_id for f in existing]
                finally:
                    target_db_query.close()

            if target_existing_ids:
                mode = "competitive" if competitive else "gatekeeper"
                click.echo(
                    f"  Target: {len(target_existing_ids)} existing [{mode}]",
                )

            # Correlation + greedy
            click.echo("[dedup] Computing correlations...")
            candidate_ids = df.index.tolist()
            all_corr_ids = list(
                dict.fromkeys(candidate_ids + target_existing_ids),
            )
            try:
                target_db_for_corr = RegistryDatabase(Path(target_db_path))
                try:
                    corr_matrix, factor_panels = compute_factor_correlation(
                        all_corr_ids, scoring_cfg,
                        registry_dbs=[source_db, target_db_for_corr],
                        return_panels=True,
                    )
                finally:
                    target_db_for_corr.close()

                if not corr_matrix.empty:
                    click.echo(
                        f"  Matrix: {corr_matrix.shape[0]}×{corr_matrix.shape[1]}",
                    )
                    # Save CSV
                    corr_csv = diag_dir / f"correlation_matrix_{timestamp}.csv"
                    corr_matrix.to_csv(corr_csv)
                    click.echo(f"  Saved: {corr_csv}")

                    # Heatmap
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
                        ax.set_title("Factor Spearman Correlation")
                        plt.tight_layout()
                        hm = diag_dir / f"correlation_heatmap_{timestamp}.png"
                        fig.savefig(hm, dpi=100)
                        plt.close(fig)
                        click.echo(f"  Saved: {hm}")
                    except Exception as e:
                        click.echo(f"  Warning: Heatmap failed: {e}", err=True)

                    # Greedy selection
                    click.echo()
                    if competitive:
                        click.echo(
                            f"[dedup] Competitive greedy "
                            f"(max_corr≤{scoring_cfg.dedup.max_corr})...",
                        )
                        selected_ids = greedy_select(
                            df, corr_matrix,
                            max_corr=scoring_cfg.dedup.max_corr,
                            max_factors=scoring_cfg.promote.max_factors,
                        )
                    else:
                        click.echo(
                            f"[dedup] Greedy "
                            f"(max_corr≤{scoring_cfg.dedup.max_corr})...",
                        )
                        selected_ids = greedy_select(
                            df, corr_matrix,
                            max_corr=scoring_cfg.dedup.max_corr,
                            max_factors=scoring_cfg.promote.max_factors,
                            existing_ids=target_existing_ids,
                        )
                    click.echo(f"  Selected {len(selected_ids)} factors")

                    # Competitive diff report
                    if competitive and target_existing_ids:
                        sel_set = set(selected_ids)
                        ex_set = set(target_existing_ids)
                        click.echo(
                            f"  Competitive: {len(sel_set & ex_set)} retained, "
                            f"{len(sel_set - ex_set)} new, "
                            f"{len(ex_set - sel_set)} evict",
                        )
                        for fid in sorted(ex_set - sel_set):
                            click.echo(f"    - evict: {fid}")
                else:
                    click.echo("  Warning: Empty corr matrix")
                    selected_ids = df.index.tolist()[
                        :scoring_cfg.promote.max_factors
                    ]
            except Exception as e:
                click.echo(f"  Warning: Correlation failed: {e}", err=True)
                selected_ids = df.index.tolist()[
                    :scoring_cfg.promote.max_factors
                ]
        else:
            click.echo("[dedup] — SKIPPED")
            target_existing_ids = []
            selected_ids = df.index.tolist()[
                :scoring_cfg.promote.max_factors
            ]

        # ── [clustering] ──
        click.echo()
        if scoring_cfg.clustering.enabled:
            algo = scoring_cfg.clustering.algorithm
            comp = scoring_cfg.clustering.composition_method
            click.echo(f"[clustering] {algo} + {comp}...")
            # Need corr matrix + factor panels — compute if dedup was skipped
            if corr_matrix is None or corr_matrix.empty:
                click.echo("  Computing correlation matrix for clustering...")
                dbs_for_corr = [source_db]
                if target_db_path:
                    target_db_for_corr = RegistryDatabase(Path(target_db_path))
                    dbs_for_corr.append(target_db_for_corr)
                try:
                    corr_matrix, factor_panels = compute_factor_correlation(
                        selected_ids, scoring_cfg,
                        registry_dbs=dbs_for_corr,
                        return_panels=True,
                    )
                finally:
                    if target_db_path:
                        target_db_for_corr.close()

            if corr_matrix is not None and not corr_matrix.empty:
                from nautilus_quants.alpha.registry.clustering import (
                    build_super_alphas,
                    cluster_factors,
                    plot_cluster_heatmap,
                )
                from nautilus_quants.alpha.registry.repository import (
                    FactorRepository,
                )

                # Get factor expressions from source DB
                src_repo = FactorRepository(source_db)
                factor_exprs: dict[str, str] = {}
                factor_tags_map: dict[str, list[str]] = {}
                for fid in selected_ids:
                    rec = src_repo.get_factor(fid)
                    if rec:
                        factor_exprs[fid] = rec.expression
                        factor_tags_map[fid] = rec.tags

                # Build clustering config
                from nautilus_quants.alpha.registry.clustering import (
                    ClusterConfig,
                )
                cluster_cfg = ClusterConfig(
                    enabled=True,
                    algorithm=scoring_cfg.clustering.algorithm,
                    composition_method=scoring_cfg.clustering.composition_method,
                    single_factor_passthrough=scoring_cfg.clustering.single_factor_passthrough,
                    noise_passthrough=scoring_cfg.clustering.noise_passthrough,
                    super_alpha_prefix=scoring_cfg.clustering.super_alpha_prefix,
                    component_status=scoring_cfg.clustering.component_status,
                    # HDBSCAN
                    min_cluster_size=scoring_cfg.clustering.min_cluster_size,
                    cluster_selection_method=scoring_cfg.clustering.cluster_selection_method,
                    min_samples=scoring_cfg.clustering.min_samples or None,
                    # Agglomerative
                    linkage=scoring_cfg.clustering.linkage,
                    distance_threshold=scoring_cfg.clustering.distance_threshold,
                    abs_correlation=scoring_cfg.clustering.abs_correlation,
                )

                super_alphas, noise = build_super_alphas(
                    factor_ids=selected_ids,
                    factor_expressions=factor_exprs,
                    corr_matrix=corr_matrix,
                    config=cluster_cfg,
                    factor_panels=factor_panels,
                    factor_tags=factor_tags_map,
                )

                # Cluster heatmap
                clusters_for_plot, noise_for_plot = cluster_factors(
                    corr_matrix.loc[
                        [f for f in selected_ids if f in corr_matrix.index],
                        [f for f in selected_ids if f in corr_matrix.columns],
                    ],
                    cluster_cfg,
                )
                heatmap_path = diag_dir / f"cluster_heatmap_{timestamp}.png"
                try:
                    plot_cluster_heatmap(
                        corr_matrix, clusters_for_plot, noise_for_plot,
                        output_path=str(heatmap_path),
                    )
                    click.echo(f"  Saved: {heatmap_path}")
                except Exception as e:
                    click.echo(f"  Warning: Heatmap failed: {e}", err=True)

                # Save cluster assignment CSV
                cluster_csv_path = diag_dir / f"cluster_assignments_{timestamp}.csv"
                rows = []
                for sa in super_alphas:
                    for member in sa.members:
                        rows.append({
                            "factor_id": member,
                            "super_alpha": sa.name,
                            "method": sa.method,
                            "weight": sa.weights.get(member, 0.0),
                            "cluster_id": sa.cluster_id,
                            "tags": "|".join(sa.tags),
                        })
                for nid in noise:
                    rows.append({
                        "factor_id": nid,
                        "super_alpha": "",
                        "method": "noise_discarded",
                        "weight": 0.0,
                        "cluster_id": -1,
                        "tags": "",
                    })
                import csv as _csv
                with open(cluster_csv_path, "w", newline="") as f:
                    writer = _csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                click.echo(f"  Saved: {cluster_csv_path}")

                n_passthrough = sum(
                    1 for sa in super_alphas if sa.method == "passthrough"
                )
                n_composed = len(super_alphas) - n_passthrough
                noise_msg = (
                    f"{len(noise)} noise discarded"
                    if noise
                    else f"{n_passthrough} noise passthrough"
                )
                click.echo(
                    f"  {len(super_alphas)} Super Alphas "
                    f"({n_composed} composed, {n_passthrough} passthrough), "
                    f"{noise_msg}",
                )
                for sa in super_alphas:
                    click.echo(
                        f"    {sa.name}: {len(sa.members)} members "
                        f"[{sa.method}] tags={sa.tags}",
                    )
            else:
                click.echo("  Warning: No corr matrix, skipping clustering")
        else:
            click.echo("[clustering] — SKIPPED")

        # ── [orthogonalization] ──
        click.echo()
        if scoring_cfg.orthogonalization.enabled:
            click.echo("[orthogonalization] Löwdin...")
            click.echo("  (S^{-1/2} computation — to be applied by regime command)")
            # TODO: compute S^{-1/2}, save to diag_dir
            click.echo(f"  Output dir: {diag_dir}")
        else:
            click.echo("[orthogonalization] — SKIPPED")

        # ── Save scores CSV (only when scoring enabled) ──
        if scoring_cfg.weights.enabled:
            scores_csv = diag_dir / f"factor_scores_{timestamp}.csv"
            save_cols = [
                c for c in [
                    "final_score", "avg_period_score", "pred_score",
                    "mono_score", "consistency", "turnover_friendliness",
                    "avg_icir", "avg_ic_mean", "avg_monotonicity",
                    "n_valid_periods",
                ] if c in df.columns
            ]
            df[save_cols].to_csv(scores_csv)
            click.echo(f"\n  Scores saved: {scores_csv}")
            print_selected_factors(selected_ids, df)
        else:
            click.echo(f"\n  Selected {len(selected_ids)} factors (no scoring)")

        # ── [migrate] ──
        click.echo()
        if scoring_cfg.promote.enabled:
            if not target_db_path:
                click.echo(
                    "Error: promote.target_db_path required.", err=True,
                )
                sys.exit(1)

            target_db = RegistryDatabase(Path(target_db_path))
            try:
                if competitive and target_existing_ids:
                    target_db.connection.begin()
                try:
                    if competitive and target_existing_ids:
                        evicted_ids = retire_evicted_factors(
                            target_db, selected_ids, target_existing_ids,
                        )
                        if evicted_ids:
                            click.echo(
                                f"[migrate] Archived {len(evicted_ids)} "
                                f"evicted factors",
                            )

                    label = "[migrate] Syncing" if competitive else "[migrate] Migrating"
                    click.echo(
                        f"{label} {len(selected_ids)} factors "
                        f"to {target_db_path}...",
                    )
                    counts = migrate_factors(
                        source_db, target_db, selected_ids,
                        target_status=scoring_cfg.promote.target_status,
                        scores=df,
                    )
                    click.echo(
                        f"  Migrated: {counts['factors']} factors, "
                        f"{counts['metrics']} metrics, "
                        f"{counts['configs']} configs",
                    )

                    if competitive and target_existing_ids:
                        target_db.connection.commit()
                except Exception:
                    if competitive and target_existing_ids:
                        target_db.connection.rollback()
                        click.echo(
                            "  Error — rolled back.", err=True,
                        )
                    raise
            finally:
                target_db.close()
        else:
            click.echo("[migrate] — SKIPPED (promote.enabled=false)")

        source_db.close()
        duration = time.time() - start_time
        print_promote_summary(
            n_after_filter, n_after_dedup, len(selected_ids),
            duration, not scoring_cfg.promote.enabled,
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
@click.option("--execute", is_flag=True, default=False, help="Actually apply.")
@_ENV_OPTION
@_DB_DIR_OPTION
def repair(
    execute: bool,
    env_name: str | None,
    db_dir: str,
) -> None:
    """Repair factors with metrics from conflicting expressions.

    Detects factors whose metrics span multiple different expressions
    (caused by LLM mining name collisions) and either splits them into
    separate factors or deletes orphaned metrics.
    """
    from nautilus_quants.alpha.registry.audit import (
        find_conflicting_factors,
        repair_factors,
    )

    dry_run = not execute
    repo, db = _open_repo(env_name, db_dir)
    try:
        conflicts = find_conflicting_factors(repo)
        if not conflicts:
            click.echo("No conflicting factors found.")
            return

        # Print conflict report
        click.echo(
            f"Found {len(conflicts)} factors with conflicting metrics:\n"
        )
        total_orphan_runs = 0
        for c in conflicts:
            n_orphans = sum(len(g.run_ids) for g in c.orphan_groups)
            total_orphan_runs += n_orphans
            click.echo(
                f"  {c.factor_id} — "
                f"{len(c.orphan_groups)} orphan group(s), "
                f"{n_orphans} orphan run(s)"
            )
            for g in c.orphan_groups:
                click.echo(
                    f"    hash={g.expression_hash[:8]}  "
                    f"runs={len(g.run_ids)}  "
                    f"metrics={g.metric_count}"
                )
                click.echo(f"    expr: {g.expression[:90]}")

        click.echo(
            f"\nTotal: {len(conflicts)} factors, "
            f"{total_orphan_runs} orphan runs"
        )

        actions = repair_factors(repo, dry_run=dry_run)

        if dry_run:
            click.echo(f"\n[DRY RUN] {len(actions)} actions planned:")
        else:
            click.echo(f"\n[EXECUTED] {len(actions)} actions applied:")

        for a in actions:
            if a.action in ("split", "merge"):
                click.echo(
                    f"  {a.action}: {a.factor_id} → {a.new_factor_id} "
                    f"({len(a.run_ids)} runs)"
                )
            else:
                click.echo(
                    f"  {a.action}: {a.factor_id} — {a.detail}"
                )

        if dry_run:
            click.echo("\nRe-run with --execute to apply.")
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

    if config.directions and config.hypothesis:
        raise click.UsageError(
            "--hypothesis and mining.directions are mutually exclusive. "
            "Remove --hypothesis or remove the directions section from config."
        )

    miner = AlphaMiner(config)
    miner.run(rounds=rounds)


if __name__ == "__main__":
    cli()
