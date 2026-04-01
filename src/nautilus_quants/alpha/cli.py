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
def analyze(config_file: Path, verbose: bool, quiet: bool) -> None:
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

        duration = time.time() - start_time

        # 10. Print results
        if not quiet:
            click.echo()
            click.echo("Factor Analysis Summary:")
            report_gen.print_summary_table(ic_results, factor_series=factor_series)

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
# Registry commands (Feature 035)
# ---------------------------------------------------------------------------

_DB_OPTION = click.option(
    "--db", "db_path", default="data/factor_registry.duckdb",
    help="Path to DuckDB registry file.",
)


def _open_repo(db_path: str):
    """Lazy-import and open a FactorRepository."""
    from nautilus_quants.alpha.registry.database import RegistryDatabase
    from nautilus_quants.alpha.registry.repository import FactorRepository

    db = RegistryDatabase(db_path)
    return FactorRepository(db), db


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("--source", default="", help="Factor source label (e.g. alpha101, fmz, mined).")
@click.option("--context-id", default="", help="Config context ID (defaults to metadata.name).")
@_DB_OPTION
def register(config_file: Path, source: str, context_id: str, db_path: str) -> None:
    """Register factors from a YAML config file into the registry."""
    from nautilus_quants.factors.config import load_factor_config

    try:
        config = load_factor_config(config_file)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    repo, db = _open_repo(db_path)
    try:
        new, updated, unchanged = repo.import_from_config(
            config, source=source, context_id=context_id,
        )
        cid = context_id or config.name
        click.echo(
            f"已注册 {new + updated + unchanged} 个因子"
            f"（{new} 新增, {updated} 更新, {unchanged} 无变化）"
        )
        click.echo(f"已保存配置上下文: {cid}")
    finally:
        db.close()


@cli.command("list")
@click.option("--status", default=None, help="Filter by status (candidate/active/archived).")
@click.option("--sort", "sort_by", default="factor_id", help="Sort column.")
@click.option("--category", default=None, help="Filter by category.")
@click.option("--source", default=None, help="Filter by source.")
@click.option("--limit", default=None, type=int, help="Max rows.")
@_DB_OPTION
def list_factors(
    status: str | None,
    sort_by: str,
    category: str | None,
    source: str | None,
    limit: int | None,
    db_path: str,
) -> None:
    """List factors in the registry."""
    repo, db = _open_repo(db_path)
    try:
        factors = repo.list_factors(
            status=status, category=category, source=source,
            sort_by=sort_by, limit=limit,
        )
        if not factors:
            click.echo("(no factors found)")
            return

        # Header
        click.echo(
            f"{'factor_id':<16} {'category':<14} {'status':<12} "
            f"{'source':<10} {'ICIR':>8} {'score':>8}"
        )
        click.echo("-" * 70)
        for f in factors:
            icir_str = f"{f.icir:.4f}" if f.icir is not None else "-"
            score_str = f"{f.score:.1f}" if f.score is not None else "-"
            click.echo(
                f"{f.factor_id:<16} {f.category:<14} {f.status:<12} "
                f"{f.source:<10} {icir_str:>8} {score_str:>8}"
            )
        click.echo(f"({len(factors)} factors)")
    finally:
        db.close()


@cli.command()
@click.argument("factor_id")
@_DB_OPTION
def inspect(factor_id: str, db_path: str) -> None:
    """Inspect a factor's details and version history."""
    repo, db = _open_repo(db_path)
    try:
        f = repo.get_factor(factor_id)
        if f is None:
            click.echo(f"Error: factor not found: {factor_id}", err=True)
            sys.exit(1)

        click.echo(f"Factor: {f.factor_id}")
        click.echo(f"Expression: {f.expression}")
        click.echo(f"Category: {f.category or '(none)'}")
        click.echo(f"Source: {f.source or '(none)'}")
        click.echo(f"Status: {f.status}")
        click.echo(f"Created: {f.created_at}")
        click.echo(f"Updated: {f.updated_at}")

        if f.icir is not None:
            click.echo(f"ICIR: {f.icir:.4f}")
        if f.score is not None:
            click.echo(f"Score: {f.score:.1f}")

        versions = repo.get_versions(factor_id)
        if versions:
            click.echo(f"\nVersions ({len(versions)}):")
            for v in versions:
                reason = f"  {v.reason}" if v.reason else ""
                click.echo(f"  v{v.version}  {v.created_at}{reason}")

        analysis = repo.get_analysis(factor_id, f.bar_spec or "")
        if analysis:
            click.echo(f"\nAnalysis ({f.bar_spec}):")
            for a in analysis:
                icir_str = f"{a.icir:.4f}" if a.icir is not None else "-"
                click.echo(f"  period={a.period}  ICIR={icir_str}")
    finally:
        db.close()


@cli.command()
@click.argument("factor_id")
@click.argument("new_status")
@_DB_OPTION
def status(factor_id: str, new_status: str, db_path: str) -> None:
    """Change a factor's status (candidate/active/archived)."""
    repo, db = _open_repo(db_path)
    try:
        f = repo.get_factor(factor_id)
        old_status = f.status if f else "?"
        repo.set_status(factor_id, new_status)
        click.echo(f"{factor_id}: {old_status} → {new_status} ✓")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command("export-factors")
@click.option("--context-id", default="", help="Config context for variables/parameters.")
@click.option("--method", default="equal", help="Composite weighting method (equal/icir_weight).")
@click.option("--top", "top_n", default=30, type=int, help="Max factors in composite.")
@click.option("--transform", default="cs_rank", help="Transform function (cs_rank/cs_zscore/raw).")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(path_type=Path))
@_DB_OPTION
def export_factors(
    context_id: str,
    method: str,
    top_n: int,
    transform: str,
    output_path: Path,
    db_path: str,
) -> None:
    """Export active factors + composite to a factors.yaml file."""
    from nautilus_quants.alpha.registry.export import export_factors_yaml

    repo, db = _open_repo(db_path)
    try:
        export_factors_yaml(
            repo, output_path,
            context_id=context_id,
            composite_method=method,
            composite_top_n=top_n,
            composite_transform=transform,
        )
        click.echo(
            f"已导出到 {output_path}\n"
            f"method={method}, transform={transform}, context={context_id or '(none)'}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    cli()
