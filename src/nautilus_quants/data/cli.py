"""
CLI entry point for the data pipeline.

Provides subcommands for download, validate, process, transform, run, status, and clean.
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from nautilus_quants.data.config import (
    ConfigurationError,
    PipelineConfig,
    TardisPipelineConfig,
    config_to_dict,
    load_config,
    load_tardis_config,
    tardis_config_to_dict,
)
from nautilus_quants.data.download.binance import (
    BinanceDownloader,
    check_disk_space,
    estimate_download_size,
)
from nautilus_quants.data.download.bybit_futures_data import (
    BybitFundingRateDownloader,
    BybitOpenInterestDownloader,
)
from nautilus_quants.data.process.processors import ProcessConfig, process_data
from nautilus_quants.data.reporting import ReportWriter, create_log_dir, generate_run_id
from nautilus_quants.data.transform.parquet import transform_to_parquet
from nautilus_quants.data.validate.integrity import validate_file


# Exit codes per CLI contract
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_DOWNLOAD_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_PROCESS_ERROR = 4
EXIT_TRANSFORM_ERROR = 5
EXIT_FILE_NOT_FOUND = 10
EXIT_PERMISSION_ERROR = 11
EXIT_DISK_FULL = 12
EXIT_API_RATE_LIMIT = 20
EXIT_NETWORK_ERROR = 21


@click.group()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    default="config/examples/data.yaml",
    help="Configuration file path",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without executing"
)
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool, dry_run: bool) -> None:
    """Binance Data Pipeline - Download, validate, process, and transform market data."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run


@cli.command()
@click.option("--symbol", "-s", help="Symbol(s) to download, comma-separated")
@click.option("--timeframe", "-t", help="Timeframe(s), comma-separated")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--market-type",
    type=click.Choice(["spot", "futures"]),
    help="Market type",
)
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoint")
@click.option("--force", is_flag=True, help="Re-download even if data exists")
@click.option(
    "--funding-rate/--no-funding-rate",
    default=None,
    help="Download funding rate data",
)
@click.option(
    "--open-interest/--no-open-interest",
    default=None,
    help="Download open interest data",
)
@click.option("--oi-period", default=None, help="OI period (5m/15m/30m/1h/2h/4h/6h/12h/1d)")
@click.pass_context
def download(
    ctx: click.Context,
    symbol: Optional[str],
    timeframe: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    market_type: Optional[str],
    resume: bool,
    force: bool,
    funding_rate: Optional[bool],
    open_interest: Optional[bool],
    oi_period: Optional[str],
) -> None:
    """Download historical K-line, funding rate, and open interest data from Binance."""
    try:
        config = _load_config_with_overrides(
            ctx,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            market_type=market_type,
            funding_rate=funding_rate,
            open_interest=open_interest,
            oi_period=oi_period,
        )
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)

    if ctx.obj["dry_run"]:
        # Estimate download size for dry run
        total_estimated_bytes = 0
        for tf in config.download.timeframes:
            est_bytes = estimate_download_size(
                config.download.start_date,
                config.download.end_date,
                tf,
                len(config.download.symbols),
            )
            total_estimated_bytes += est_bytes

        click.echo("DRY RUN: Would download data with config:")
        click.echo(f"  Symbols: {config.download.symbols}")
        click.echo(f"  Timeframes: {config.download.timeframes}")
        click.echo(
            f"  Date range: {config.download.start_date} to {config.download.end_date}"
        )
        click.echo(f"  Market type: {config.download.market_type}")
        click.echo(f"  Estimated size: ~{total_estimated_bytes / (1024 * 1024):.1f} MB")
        if config.download.funding_rate:
            click.echo("  Funding rate: enabled")
        if config.download.open_interest:
            oi_p = config.download.oi_period or "4h"
            click.echo(f"  Open interest: enabled (period={oi_p})")

        # Check disk space even in dry run
        output_dir = Path(config.paths.raw_data) / "binance"
        has_space, space_msg = check_disk_space(output_dir, total_estimated_bytes)
        click.echo(f"  {space_msg}")
        return

    # Pre-flight disk space check
    output_dir = Path(config.paths.raw_data) / "binance"
    total_estimated_bytes = 0
    for tf in config.download.timeframes:
        est_bytes = estimate_download_size(
            config.download.start_date,
            config.download.end_date,
            tf,
            len(config.download.symbols),
        )
        total_estimated_bytes += est_bytes

    has_space, space_msg = check_disk_space(output_dir, total_estimated_bytes)
    if ctx.obj["verbose"]:
        click.echo(f"Disk space check: {space_msg}")

    if not has_space:
        click.echo(f"Error: {space_msg}", err=True)
        ctx.exit(EXIT_DISK_FULL)

    # Run download
    results = asyncio.run(
        _run_download(config, resume=resume and not force, verbose=ctx.obj["verbose"])
    )

    # Check results
    failed = [r for r in results if not r.success]
    if failed:
        click.echo(f"\nDownload failed for {len(failed)} symbol(s):", err=True)
        for r in failed:
            click.echo(f"  {r.symbol}: {r.errors}", err=True)
        ctx.exit(EXIT_DOWNLOAD_ERROR)

    click.echo(f"\nDownload complete: {len(results)} symbol(s) processed")


async def _run_download(
    config: PipelineConfig, resume: bool = True, verbose: bool = False
):
    """Run download for all configured symbols and timeframes."""
    output_dir = Path(config.paths.raw_data) / "binance"
    checkpoint_dir = output_dir / ".checkpoints"

    results = []

    # K-line download
    downloader = BinanceDownloader(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=config.download.checkpoint.batch_size,
    )

    for symbol in config.download.symbols:
        for timeframe in config.download.timeframes:
            click.echo(f"Downloading {symbol} {timeframe}...")

            result = await downloader.download(
                symbol=symbol,
                timeframe=timeframe,
                start_date=config.download.start_date,
                end_date=config.download.end_date,
                market_type=config.download.market_type,
                resume=resume,
            )

            if result.success:
                click.echo(f"  ✓ {result.rows_downloaded} rows downloaded")
                if result.resumed_from_checkpoint:
                    click.echo("    (resumed from checkpoint)")
            else:
                click.echo(f"  ✗ Failed: {result.errors}", err=True)

            results.append(result)

    # Funding rate download (Bybit, sync)
    if config.download.funding_rate:
        bybit_output_dir = Path(config.paths.raw_data) / "bybit"
        bybit_checkpoint_dir = bybit_output_dir / ".checkpoints"
        fr_downloader = BybitFundingRateDownloader(
            output_dir=bybit_output_dir,
            checkpoint_dir=bybit_checkpoint_dir,
            batch_size=config.download.checkpoint.batch_size,
        )
        for symbol in config.download.symbols:
            click.echo(f"Downloading {symbol} funding_rate (bybit)...")

            result = await asyncio.to_thread(
                fr_downloader.download,
                symbol=symbol,
                start_date=config.download.start_date,
                end_date=config.download.end_date,
            )

            if result.success:
                click.echo(f"  ✓ {result.rows_downloaded} rows downloaded")
                if result.resumed_from_checkpoint:
                    click.echo("    (resumed from checkpoint)")
            else:
                click.echo(f"  ✗ Failed: {result.errors}", err=True)

            results.append(result)

    # Open interest download (Bybit, sync)
    if config.download.open_interest:
        oi_period = config.download.oi_period or "4h"
        bybit_output_dir = Path(config.paths.raw_data) / "bybit"
        bybit_checkpoint_dir = bybit_output_dir / ".checkpoints"
        oi_downloader = BybitOpenInterestDownloader(
            output_dir=bybit_output_dir,
            checkpoint_dir=bybit_checkpoint_dir,
            batch_size=config.download.checkpoint.batch_size,
        )
        for symbol in config.download.symbols:
            click.echo(f"Downloading {symbol} oi_{oi_period} (bybit)...")

            result = await asyncio.to_thread(
                oi_downloader.download,
                symbol=symbol,
                period=oi_period,
                start_date=config.download.start_date,
                end_date=config.download.end_date,
            )

            if result.success:
                click.echo(f"  ✓ {result.rows_downloaded} rows downloaded")
                if result.resumed_from_checkpoint:
                    click.echo("    (resumed from checkpoint)")
            else:
                click.echo(f"  ✗ Failed: {result.errors}", err=True)

            results.append(result)

    return results


@cli.command()
@click.option("--symbol", "-s", help="Symbol(s) to validate, comma-separated")
@click.option("--timeframe", "-t", help="Timeframe(s), comma-separated")
@click.option(
    "--input-dir", type=click.Path(exists=True), help="Input directory (raw data)"
)
@click.option("--fail-on-warnings", is_flag=True, help="Treat warnings as errors")
@click.option("--output-json", type=click.Path(), help="Write JSON report to file")
@click.pass_context
def validate(
    ctx: click.Context,
    symbol: Optional[str],
    timeframe: Optional[str],
    input_dir: Optional[str],
    fail_on_warnings: bool,
    output_json: Optional[str],
) -> None:
    """Validate raw data integrity and consistency."""
    try:
        config = _load_config_with_overrides(ctx, symbol=symbol, timeframe=timeframe)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)

    if ctx.obj["dry_run"]:
        click.echo("DRY RUN: Would validate data with config:")
        click.echo(f"  Symbols: {config.download.symbols}")
        click.echo(f"  Timeframes: {config.download.timeframes}")
        return

    raw_dir = Path(input_dir) if input_dir else Path(config.paths.raw_data) / "binance"

    reports = []
    has_errors = False

    for sym in config.download.symbols:
        for tf in config.download.timeframes:
            # Find data files
            pattern = f"{sym}_{tf}_*.csv"
            data_dir = raw_dir / sym / tf
            files = list(data_dir.glob(pattern)) if data_dir.exists() else []

            if not files:
                click.echo(f"No data files found for {sym} {tf}")
                continue

            for file_path in files:
                click.echo(f"Validating {file_path.name}...")
                report = validate_file(
                    file_path,
                    max_gap_bars=config.process.max_gap_bars,
                )
                reports.append(report)

                if report.passed:
                    click.echo(f"  ✓ Passed ({report.total_rows} rows)")
                    if report.warning_count > 0:
                        click.echo(f"    Warnings: {report.warning_count}")
                else:
                    click.echo(
                        f"  ✗ Failed ({report.error_count} errors, {report.warning_count} warnings)"
                    )
                    has_errors = True

                if fail_on_warnings and report.warning_count > 0:
                    has_errors = True

    # Write JSON report if requested
    if output_json and reports:
        json_data = [
            {
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "file_path": r.file_path,
                "total_rows": r.total_rows,
                "passed": r.passed,
                "error_count": r.error_count,
                "warning_count": r.warning_count,
                "duplicate_count": r.duplicate_count,
                "gap_count": r.gap_count,
                "invalid_ohlc_count": r.invalid_ohlc_count,
            }
            for r in reports
        ]
        with open(output_json, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        click.echo(f"\nJSON report written to: {output_json}")

    if has_errors:
        ctx.exit(EXIT_VALIDATION_ERROR)

    click.echo(f"\nValidation complete: {len(reports)} file(s) validated")


@cli.command()
@click.option("--symbol", "-s", help="Symbol(s) to process, comma-separated")
@click.option("--timeframe", "-t", help="Timeframe(s), comma-separated")
@click.option(
    "--input-dir", type=click.Path(exists=True), help="Input directory (raw data)"
)
@click.option(
    "--output-dir", type=click.Path(), help="Output directory (processed data)"
)
@click.option("--max-gap-bars", type=int, help="Max gap size to fill")
@click.option("--force", is_flag=True, help="Overwrite existing processed files")
@click.pass_context
def process(
    ctx: click.Context,
    symbol: Optional[str],
    timeframe: Optional[str],
    input_dir: Optional[str],
    output_dir: Optional[str],
    max_gap_bars: Optional[int],
    force: bool,
) -> None:
    """Process validated data: remove duplicates, fill gaps."""
    try:
        config = _load_config_with_overrides(ctx, symbol=symbol, timeframe=timeframe)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)

    if ctx.obj["dry_run"]:
        click.echo("DRY RUN: Would process data with config:")
        click.echo(f"  Symbols: {config.download.symbols}")
        click.echo(f"  Timeframes: {config.download.timeframes}")
        return

    raw_dir = Path(input_dir) if input_dir else Path(config.paths.raw_data) / "binance"
    proc_dir = (
        Path(output_dir)
        if output_dir
        else Path(config.paths.processed_data) / "binance"
    )

    proc_config = ProcessConfig(
        remove_duplicates=config.process.remove_duplicates,
        keep_duplicate=config.process.keep_duplicate,
        fill_small_gaps=config.process.fill_small_gaps,
        max_gap_bars=max_gap_bars if max_gap_bars else config.process.max_gap_bars,
        remove_invalid_ohlc=config.process.remove_invalid_ohlc,
    )

    reports = []
    has_errors = False

    for sym in config.download.symbols:
        for tf in config.download.timeframes:
            # Find raw data files
            pattern = f"{sym}_{tf}_*.csv"
            data_dir = raw_dir / sym / tf
            files = list(data_dir.glob(pattern)) if data_dir.exists() else []

            if not files:
                click.echo(f"No raw data files found for {sym} {tf}")
                continue

            for file_path in files:
                # Output path
                out_dir = proc_dir / sym / tf
                out_path = out_dir / f"{sym}_{tf}_processed.csv"

                if out_path.exists() and not force:
                    click.echo(
                        f"Skipping {file_path.name} (processed file exists, use --force)"
                    )
                    continue

                click.echo(f"Processing {file_path.name}...")

                try:
                    report = process_data(file_path, out_path, config=proc_config)
                    reports.append(report)

                    click.echo(f"  ✓ {report.input_rows} → {report.output_rows} rows")
                    click.echo(f"    Duplicates removed: {report.duplicates_removed}")
                    click.echo(f"    Gaps filled: {report.gaps_filled}")
                    click.echo(f"    Invalid removed: {report.invalid_rows_removed}")
                except Exception as e:
                    click.echo(f"  ✗ Failed: {e}", err=True)
                    has_errors = True

    if has_errors:
        ctx.exit(EXIT_PROCESS_ERROR)

    click.echo(f"\nProcessing complete: {len(reports)} file(s) processed")


@cli.command()
@click.option("--symbol", "-s", help="Symbol(s) to transform, comma-separated")
@click.option("--timeframe", "-t", help="Timeframe(s), comma-separated")
@click.option(
    "--input-dir", type=click.Path(exists=True), help="Input directory (processed data)"
)
@click.option("--catalog-path", type=click.Path(), help="Parquet catalog output path")
@click.option(
    "--merge/--no-merge", default=True, help="Merge multiple files per symbol"
)
@click.option("--force", is_flag=True, help="Overwrite existing Parquet files")
@click.pass_context
def transform(
    ctx: click.Context,
    symbol: Optional[str],
    timeframe: Optional[str],
    input_dir: Optional[str],
    catalog_path: Optional[str],
    merge: bool,
    force: bool,
) -> None:
    """Transform processed CSV to Nautilus Parquet format."""
    try:
        config = _load_config_with_overrides(ctx, symbol=symbol, timeframe=timeframe)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)

    if ctx.obj["dry_run"]:
        click.echo("DRY RUN: Would transform data with config:")
        click.echo(f"  Symbols: {config.download.symbols}")
        click.echo(f"  Timeframes: {config.download.timeframes}")
        if config.download.funding_rate:
            click.echo("  Funding rate: enabled")
        if config.download.open_interest:
            oi_p = config.download.oi_period or "4h"
            click.echo(f"  Open interest: enabled (period={oi_p})")
        return

    proc_dir = (
        Path(input_dir) if input_dir else Path(config.paths.processed_data) / "binance"
    )
    cat_path = Path(catalog_path) if catalog_path else Path(config.paths.catalog)

    results = []
    has_errors = False

    # K-line transform
    for sym in config.download.symbols:
        for tf in config.download.timeframes:
            # Find processed data files
            pattern = f"{sym}_{tf}_processed.csv"
            data_dir = proc_dir / sym / tf
            files = list(data_dir.glob(pattern)) if data_dir.exists() else []

            if not files:
                click.echo(f"No processed data files found for {sym} {tf}")
                continue

            for file_path in files:
                click.echo(f"Transforming {file_path.name}...")

                # Get raw data path for precision lookup
                raw_data_path = Path(config.paths.raw_data) / "binance"

                try:
                    result = transform_to_parquet(
                        input_path=file_path,
                        catalog_path=cat_path,
                        symbol=sym,
                        timeframe=tf,
                        merge=merge,
                        raw_data_path=raw_data_path,
                        maker_fee=config.transform.maker_fee,
                        taker_fee=config.transform.taker_fee,
                        margin_init=config.transform.margin_init,
                        margin_maint=config.transform.margin_maint,
                        bar_class=config.transform.bar_class,
                    )
                    results.append(result)

                    if result.success:
                        click.echo(
                            f"  ✓ {result.rows_transformed} bars written to catalog"
                        )
                    else:
                        click.echo(f"  ✗ Failed: {result.errors}", err=True)
                        has_errors = True
                except Exception as e:
                    click.echo(f"  ✗ Failed: {e}", err=True)
                    has_errors = True

    # Funding rate transform (Bybit CSV → NT FundingRateUpdate → catalog)
    if config.download.funding_rate:
        from nautilus_quants.data.transform.funding_rate import (
            transform_funding_rates,
        )

        bybit_raw_dir = Path(config.paths.raw_data) / "bybit"
        click.echo("Transforming funding rates...")
        fr_results = transform_funding_rates(
            raw_dir=bybit_raw_dir,
            catalog_path=cat_path,
            symbols=config.download.symbols,
        )
        for r in fr_results:
            if r["success"]:
                click.echo(f"  ✓ {r['symbol']}: {r['count']} FundingRateUpdate records")
            else:
                click.echo(f"  ✗ {r['symbol']}: {r.get('error', 'unknown')}", err=True)
                has_errors = True

    # Open interest transform (Bybit CSV → standalone Parquet)
    if config.download.open_interest:
        from nautilus_quants.data.transform.open_interest import (
            transform_open_interest,
        )

        bybit_raw_dir = Path(config.paths.raw_data) / "bybit"
        oi_period = config.download.oi_period or "4h"
        click.echo(f"Transforming open interest ({oi_period})...")
        oi_results = transform_open_interest(
            raw_dir=bybit_raw_dir,
            catalog_path=cat_path,
            symbols=config.download.symbols,
            timeframe=oi_period,
        )
        for r in oi_results:
            if r["success"]:
                click.echo(f"  ✓ {r['symbol']}: {r['count']} OI records → parquet")
            else:
                click.echo(f"  ✗ {r['symbol']}: {r.get('error', 'unknown')}", err=True)
                has_errors = True

    if has_errors:
        ctx.exit(EXIT_TRANSFORM_ERROR)

    click.echo(f"\nTransform complete: {len(results)} file(s) transformed")


@cli.command()
@click.option("--symbol", "-s", help="Symbol(s), comma-separated")
@click.option("--timeframe", "-t", help="Timeframe(s), comma-separated")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--market-type", type=click.Choice(["spot", "futures"]), help="Market type"
)
@click.option("--skip-download", is_flag=True, help="Skip download step")
@click.option("--skip-validate", is_flag=True, help="Skip validation step")
@click.option("--skip-process", is_flag=True, help="Skip processing step")
@click.option("--force", is_flag=True, help="Overwrite existing output files")
@click.pass_context
def run(
    ctx: click.Context,
    symbol: Optional[str],
    timeframe: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    market_type: Optional[str],
    skip_download: bool,
    skip_validate: bool,
    skip_process: bool,
    force: bool,
) -> None:
    """Execute full pipeline: download → validate → process → transform."""
    try:
        config = _load_config_with_overrides(
            ctx,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            market_type=market_type,
        )
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)

    run_id = generate_run_id()
    log_dir = create_log_dir(config.paths.logs, run_id)
    start_time = datetime.now()

    click.echo("=" * 70)
    click.echo(f"DATA PIPELINE RUN: {run_id}")
    click.echo("=" * 70)
    click.echo(f"Symbols: {config.download.symbols}")
    click.echo(f"Timeframes: {config.download.timeframes}")
    click.echo(
        f"Date range: {config.download.start_date} to {config.download.end_date}"
    )
    click.echo(f"Log directory: {log_dir}")
    click.echo()

    if ctx.obj["dry_run"]:
        click.echo("DRY RUN: Would execute pipeline steps:")
        if not skip_download:
            click.echo("  1. Download")
        if not skip_validate:
            click.echo("  2. Validate")
        if not skip_process:
            click.echo("  3. Process")
        click.echo("  4. Transform")
        return

    step_results = []
    errors = []

    # Step 1: Download
    if not skip_download:
        click.echo("[1/4] DOWNLOAD")
        click.echo("-" * 40)
        step_start = datetime.now()
        try:
            download_results = asyncio.run(
                _run_download(config, resume=not force, verbose=ctx.obj["verbose"])
            )
            failed = [r for r in download_results if not r.success]
            step_results.append(
                {
                    "step_name": "download",
                    "success": len(failed) == 0,
                    "duration_seconds": (datetime.now() - step_start).total_seconds(),
                    "summary": {
                        "symbols_total": len(download_results),
                        "symbols_successful": len(download_results) - len(failed),
                        "total_rows": sum(r.rows_downloaded for r in download_results),
                    },
                }
            )
            if failed:
                for r in failed:
                    errors.extend(r.errors)
                click.echo(f"Download failed for {len(failed)} symbol(s)")
                ctx.exit(EXIT_DOWNLOAD_ERROR)
        except Exception as e:
            errors.append(str(e))
            step_results.append(
                {
                    "step_name": "download",
                    "success": False,
                    "duration_seconds": (datetime.now() - step_start).total_seconds(),
                    "summary": {},
                }
            )
            ctx.exit(EXIT_DOWNLOAD_ERROR)
        click.echo()

    # Step 2: Validate
    if not skip_validate:
        click.echo("[2/4] VALIDATE")
        click.echo("-" * 40)
        step_start = datetime.now()
        # Invoke validate command logic inline
        raw_dir = Path(config.paths.raw_data) / "binance"
        validation_reports = []
        has_errors = False

        for sym in config.download.symbols:
            for tf in config.download.timeframes:
                pattern = f"{sym}_{tf}_*.csv"
                data_dir = raw_dir / sym / tf
                files = list(data_dir.glob(pattern)) if data_dir.exists() else []

                for file_path in files:
                    report = validate_file(
                        file_path,
                        max_gap_bars=config.process.max_gap_bars,
                    )
                    validation_reports.append(report)
                    if report.passed:
                        click.echo(f"  ✓ {sym} {tf}: {report.total_rows} rows")
                    else:
                        click.echo(f"  ✗ {sym} {tf}: {report.error_count} errors")
                        has_errors = True

        step_results.append(
            {
                "step_name": "validate",
                "success": not has_errors,
                "duration_seconds": (datetime.now() - step_start).total_seconds(),
                "summary": {
                    "files_validated": len(validation_reports),
                    "files_passed": sum(1 for r in validation_reports if r.passed),
                },
            }
        )
        if has_errors:
            ctx.exit(EXIT_VALIDATION_ERROR)
        click.echo()

    # Step 3: Process
    if not skip_process:
        click.echo("[3/4] PROCESS")
        click.echo("-" * 40)
        step_start = datetime.now()
        raw_dir = Path(config.paths.raw_data) / "binance"
        proc_dir = Path(config.paths.processed_data) / "binance"

        proc_config = ProcessConfig(
            remove_duplicates=config.process.remove_duplicates,
            keep_duplicate=config.process.keep_duplicate,
            fill_small_gaps=config.process.fill_small_gaps,
            max_gap_bars=config.process.max_gap_bars,
            remove_invalid_ohlc=config.process.remove_invalid_ohlc,
        )

        processing_reports = []
        has_errors = False

        for sym in config.download.symbols:
            for tf in config.download.timeframes:
                pattern = f"{sym}_{tf}_*.csv"
                data_dir = raw_dir / sym / tf
                files = list(data_dir.glob(pattern)) if data_dir.exists() else []

                for file_path in files:
                    out_dir = proc_dir / sym / tf
                    out_path = out_dir / f"{sym}_{tf}_processed.csv"

                    try:
                        report = process_data(file_path, out_path, config=proc_config)
                        processing_reports.append(report)
                        click.echo(
                            f"  ✓ {sym} {tf}: {report.input_rows} → {report.output_rows} rows"
                        )
                    except Exception as e:
                        click.echo(f"  ✗ {sym} {tf}: {e}")
                        has_errors = True

        step_results.append(
            {
                "step_name": "process",
                "success": not has_errors,
                "duration_seconds": (datetime.now() - step_start).total_seconds(),
                "summary": {
                    "files_processed": len(processing_reports),
                    "total_duplicates_removed": sum(
                        r.duplicates_removed for r in processing_reports
                    ),
                    "total_gaps_filled": sum(r.gaps_filled for r in processing_reports),
                },
            }
        )
        if has_errors:
            ctx.exit(EXIT_PROCESS_ERROR)
        click.echo()

    # Step 4: Transform
    click.echo("[4/4] TRANSFORM")
    click.echo("-" * 40)
    step_start = datetime.now()
    proc_dir = Path(config.paths.processed_data) / "binance"
    cat_path = Path(config.paths.catalog)
    raw_dir = Path(config.paths.raw_data) / "binance"

    transform_results = []
    has_errors = False

    for sym in config.download.symbols:
        for tf in config.download.timeframes:
            pattern = f"{sym}_{tf}_processed.csv"
            data_dir = proc_dir / sym / tf
            files = list(data_dir.glob(pattern)) if data_dir.exists() else []

            for file_path in files:
                try:
                    result = transform_to_parquet(
                        input_path=file_path,
                        catalog_path=cat_path,
                        symbol=sym,
                        timeframe=tf,
                        merge=config.transform.merge_files,
                        raw_data_path=raw_dir,
                        maker_fee=config.transform.maker_fee,
                        taker_fee=config.transform.taker_fee,
                        margin_init=config.transform.margin_init,
                        margin_maint=config.transform.margin_maint,
                        bar_class=config.transform.bar_class,
                    )
                    transform_results.append(result)
                    if result.success:
                        click.echo(f"  ✓ {sym} {tf}: {result.rows_transformed} bars")
                    else:
                        click.echo(f"  ✗ {sym} {tf}: {result.errors}")
                        has_errors = True
                except Exception as e:
                    click.echo(f"  ✗ {sym} {tf}: {e}")
                    has_errors = True

    # Funding rate transform (Bybit CSV → NT FundingRateUpdate → catalog)
    if config.download.funding_rate:
        from nautilus_quants.data.transform.funding_rate import (
            transform_funding_rates,
        )

        bybit_raw_dir = Path(config.paths.raw_data) / "bybit"
        click.echo("Transforming funding rates...")
        fr_results = transform_funding_rates(
            raw_dir=bybit_raw_dir,
            catalog_path=cat_path,
            symbols=config.download.symbols,
        )
        for r in fr_results:
            if r["success"]:
                click.echo(f"  ✓ {r['symbol']}: {r['count']} FundingRateUpdate records")
            else:
                click.echo(f"  ✗ {r['symbol']}: {r.get('error', 'unknown')}", err=True)
                has_errors = True

    # Open interest transform (Bybit CSV → standalone Parquet)
    if config.download.open_interest:
        from nautilus_quants.data.transform.open_interest import (
            transform_open_interest,
        )

        bybit_raw_dir = Path(config.paths.raw_data) / "bybit"
        oi_period = config.download.oi_period or "4h"
        click.echo(f"Transforming open interest ({oi_period})...")
        oi_results = transform_open_interest(
            raw_dir=bybit_raw_dir,
            catalog_path=cat_path,
            symbols=config.download.symbols,
            timeframe=oi_period,
        )
        for r in oi_results:
            if r["success"]:
                click.echo(f"  ✓ {r['symbol']}: {r['count']} OI records → parquet")
            else:
                click.echo(f"  ✗ {r['symbol']}: {r.get('error', 'unknown')}", err=True)
                has_errors = True

    step_results.append(
        {
            "step_name": "transform",
            "success": not has_errors,
            "duration_seconds": (datetime.now() - step_start).total_seconds(),
            "summary": {
                "files_transformed": len(transform_results),
                "total_bars": sum(
                    r.rows_transformed for r in transform_results if r.success
                ),
            },
        }
    )

    # Write reports
    end_time = datetime.now()
    writer = ReportWriter(log_dir)

    writer.write_summary_report(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        config=config_to_dict(config),
        step_results=step_results,
        errors=errors if errors else None,
    )

    writer.write_details_json(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        config=config_to_dict(config),
        steps=step_results,
        errors=errors if errors else None,
    )

    # Summary
    click.echo()
    click.echo("=" * 70)
    duration = (end_time - start_time).total_seconds()
    all_success = all(s.get("success", False) for s in step_results)
    status = "SUCCESS" if all_success else "FAILED"
    click.echo(f"PIPELINE {status} in {duration:.1f} seconds")
    click.echo(f"Reports: {log_dir}")
    click.echo("=" * 70)

    if has_errors:
        ctx.exit(EXIT_TRANSFORM_ERROR)


@cli.command()
@click.option("--symbol", "-s", help="Filter by symbol(s)")
@click.option(
    "--format", "output_format", type=click.Choice(["table", "json"]), default="table"
)
@click.pass_context
def status(ctx: click.Context, symbol: Optional[str], output_format: str) -> None:
    """Show pipeline status and data file status."""
    try:
        config = _load_config_with_overrides(ctx, symbol=symbol)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)

    raw_dir = Path(config.paths.raw_data) / "binance"
    proc_dir = Path(config.paths.processed_data) / "binance"
    cat_dir = Path(config.paths.catalog)

    click.echo("Pipeline Status")
    click.echo("-" * 70)
    click.echo(
        f"{'Symbol':<12} {'Timeframe':<10} {'Raw':<12} {'Processed':<12} {'Catalog':<10}"
    )
    click.echo("-" * 70)

    status_data = []

    for sym in config.download.symbols:
        for tf in config.download.timeframes:
            raw_files = (
                list((raw_dir / sym / tf).glob("*.csv"))
                if (raw_dir / sym / tf).exists()
                else []
            )
            proc_files = (
                list((proc_dir / sym / tf).glob("*_processed.csv"))
                if (proc_dir / sym / tf).exists()
                else []
            )

            raw_rows = 0
            for f in raw_files:
                try:
                    import pandas as pd

                    raw_rows += len(pd.read_csv(f))
                except Exception:
                    pass

            proc_rows = 0
            for f in proc_files:
                try:
                    import pandas as pd

                    proc_rows += len(pd.read_csv(f))
                except Exception:
                    pass

            has_catalog = cat_dir.exists()  # Simplified check

            raw_str = f"{raw_rows} rows" if raw_rows else "-"
            proc_str = f"{proc_rows} rows" if proc_rows else "-"
            cat_str = "✓" if has_catalog and proc_rows else "-"

            click.echo(f"{sym:<12} {tf:<10} {raw_str:<12} {proc_str:<12} {cat_str:<10}")

            status_data.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "raw_rows": raw_rows,
                    "processed_rows": proc_rows,
                    "has_catalog": has_catalog and proc_rows > 0,
                }
            )

    if output_format == "json":
        click.echo()
        click.echo(json.dumps(status_data, indent=2))


@cli.command()
@click.option("--symbol", "-s", help="Filter by symbol(s)")
@click.option("--raw", is_flag=True, help="Remove raw data files")
@click.option("--processed", is_flag=True, help="Remove processed data files")
@click.option("--catalog", is_flag=True, help="Remove Parquet catalog files")
@click.option("--logs", is_flag=True, help="Remove log files")
@click.option("--checkpoints", is_flag=True, help="Remove checkpoint files")
@click.option("--all", "clean_all", is_flag=True, help="Remove all generated files")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def clean(
    ctx: click.Context,
    symbol: Optional[str],
    raw: bool,
    processed: bool,
    catalog: bool,
    logs: bool,
    checkpoints: bool,
    clean_all: bool,
    force: bool,
) -> None:
    """Clean generated files."""
    try:
        config = _load_config_with_overrides(ctx, symbol=symbol)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)

    if not (raw or processed or catalog or logs or checkpoints or clean_all):
        click.echo("No clean options specified. Use --help for options.", err=True)
        return

    if clean_all:
        raw = processed = catalog = logs = checkpoints = True

    if not force:
        targets = []
        if raw:
            targets.append("raw data")
        if processed:
            targets.append("processed data")
        if catalog:
            targets.append("catalog")
        if logs:
            targets.append("logs")
        if checkpoints:
            targets.append("checkpoints")

        if not click.confirm(f"Remove {', '.join(targets)}?"):
            click.echo("Aborted.")
            return

    symbols = config.download.symbols

    if raw:
        raw_dir = Path(config.paths.raw_data) / "binance"
        for sym in symbols:
            sym_dir = raw_dir / sym
            if sym_dir.exists():
                shutil.rmtree(sym_dir)
                click.echo(f"Removed raw data for {sym}")

    if processed:
        proc_dir = Path(config.paths.processed_data) / "binance"
        for sym in symbols:
            sym_dir = proc_dir / sym
            if sym_dir.exists():
                shutil.rmtree(sym_dir)
                click.echo(f"Removed processed data for {sym}")

    if catalog:
        cat_dir = Path(config.paths.catalog)
        if cat_dir.exists():
            shutil.rmtree(cat_dir)
            click.echo("Removed catalog")

    if logs:
        log_dir = Path(config.paths.logs)
        if log_dir.exists():
            shutil.rmtree(log_dir)
            click.echo("Removed logs")

    if checkpoints:
        cp_dir = Path(config.paths.raw_data) / "binance" / ".checkpoints"
        if cp_dir.exists():
            shutil.rmtree(cp_dir)
            click.echo("Removed checkpoints")

    click.echo("Clean complete.")


@cli.command("tardis-download")
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config/examples/tardis_data.yaml",
    help="Tardis configuration file path",
)
@click.option("--symbol", "-s", help="Override symbols, comma-separated")
@click.option("--from-date", help="Override start date (YYYY-MM-DD)")
@click.option("--to-date", help="Override end date (YYYY-MM-DD, non-inclusive)")
@click.option("--force", is_flag=True, help="Re-download all (delete existing first)")
@click.pass_context
def tardis_download(
    ctx: click.Context,
    config_path: str,
    symbol: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    force: bool,
) -> None:
    """Download tick-level trade data from Tardis.dev."""
    overrides: dict[str, str] = {}
    if symbol:
        overrides["download.symbols"] = symbol
    if from_date:
        overrides["download.from_date"] = from_date
    if to_date:
        overrides["download.to_date"] = to_date

    try:
        config = load_tardis_config(config_path, overrides if overrides else None)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)
        return

    if ctx.obj.get("dry_run"):
        click.echo("DRY RUN: Would download Tardis data with config:")
        click.echo(f"  Exchange: {config.download.exchange}")
        click.echo(f"  Symbols: {list(config.download.symbols)}")
        click.echo(f"  Data types: {list(config.download.data_types)}")
        click.echo(f"  Date range: {config.download.from_date} to {config.download.to_date}")
        click.echo(f"  Symbol workers: {config.download.max_symbol_workers}")
        return

    from nautilus_quants.data.download.tardis import TardisDownloader

    run_id = generate_run_id()
    log_dir = create_log_dir("logs/tardis_pipeline", run_id)
    start_time = datetime.now()

    click.echo("=" * 70)
    click.echo(f"TARDIS DOWNLOAD: {run_id}")
    click.echo("=" * 70)
    click.echo(f"Exchange: {config.download.exchange}")
    click.echo(f"Symbols: {len(config.download.symbols)}")
    click.echo(f"Data types: {list(config.download.data_types)}")
    click.echo(f"Date range: {config.download.from_date} to {config.download.to_date}")
    click.echo(f"Workers: {config.download.max_symbol_workers}, Concurrency: {config.download.concurrency}")
    click.echo(f"Log directory: {log_dir}")
    click.echo()

    downloader = TardisDownloader(config=config.download, paths=config.paths)

    if force:
        click.echo("Force mode: cleaning existing data...")
        downloader.clean()

    errors: list[str] = []
    try:
        results = downloader.download_all()
    except Exception as e:
        errors.append(f"Download crashed: {e}")
        results = []

    failed = [r for r in results if not r.success]
    succeeded = [r for r in results if r.success]
    for r in failed:
        errors.append(f"{r.symbol}: {r.error}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    all_success = len(failed) == 0 and not errors

    # Write reports
    writer = ReportWriter(log_dir)
    writer.write_step_report(
        step_name="download",
        run_id=run_id,
        summary={
            "total_tasks": len(results),
            "successful": len(succeeded),
            "failed": len(failed),
            "duration_seconds": round(duration, 1),
        },
        details_by_symbol=[
            {
                "symbol": r.symbol,
                "success": r.success,
                "files": 1,
                "size_mb": 0.0,
                "time_seconds": 0.0,
            }
            for r in results
        ],
        errors=errors if errors else None,
    )
    writer.write_details_json(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        config=tardis_config_to_dict(config),
        steps=[
            {
                "step_name": "download",
                "success": all_success,
                "duration_seconds": round(duration, 1),
                "summary": {
                    "total_tasks": len(results),
                    "successful": len(succeeded),
                    "failed": len(failed),
                },
            }
        ],
        errors=errors if errors else None,
    )

    # Terminal summary
    click.echo()
    click.echo("=" * 70)
    duration_str = f"{duration / 60:.1f} minutes" if duration >= 60 else f"{duration:.1f} seconds"
    status = "SUCCESS" if all_success else "FAILED"
    click.echo(f"TARDIS DOWNLOAD {status} in {duration_str}")
    click.echo(f"  Successful: {len(succeeded)}, Failed: {len(failed)}")
    click.echo(f"  Reports: {log_dir}")
    click.echo("=" * 70)

    if not all_success:
        ctx.exit(EXIT_DOWNLOAD_ERROR)


@cli.command("tardis-transform")
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config/examples/tardis_data.yaml",
    help="Tardis configuration file path",
)
@click.option("--symbol", "-s", help="Override symbols, comma-separated")
@click.option("--workers", "-w", type=int, default=None, help="Max concurrent transform threads (default: from config)")
@click.pass_context
def tardis_transform(
    ctx: click.Context,
    config_path: str,
    symbol: Optional[str],
    workers: int,
) -> None:
    """Transform Tardis CSV data to NautilusTrader Parquet format."""
    overrides: dict[str, str] = {}
    if symbol:
        overrides["download.symbols"] = symbol

    try:
        config = load_tardis_config(config_path, overrides if overrides else None)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)
        return

    data_types = list(config.download.data_types)
    if workers is None:
        workers = config.transform.max_workers

    if ctx.obj.get("dry_run"):
        click.echo("DRY RUN: Would transform Tardis data:")
        click.echo(f"  Symbols: {list(config.download.symbols)}")
        click.echo(f"  Data types: {data_types}")
        click.echo(f"  Catalog: {config.paths.catalog}")
        click.echo(f"  Workers: {workers}")
        return

    from nautilus_quants.data.transform.tardis import transform_all

    run_id = generate_run_id()
    log_dir = create_log_dir("logs/tardis_pipeline", run_id)
    start_time = datetime.now()

    catalog_path = Path(config.transform.catalog_path or config.paths.catalog)

    click.echo("=" * 70)
    click.echo(f"TARDIS TRANSFORM: {run_id}")
    click.echo("=" * 70)
    click.echo(f"Symbols: {len(config.download.symbols)}")
    click.echo(f"Data types: {data_types}")
    click.echo(f"Workers: {workers}")
    click.echo(f"Catalog: {catalog_path}")
    click.echo(f"Log directory: {log_dir}")
    click.echo()

    results = transform_all(
        raw_data_dir=Path(config.paths.raw_data),
        catalog_path=catalog_path,
        exchange=config.download.exchange,
        symbols=list(config.download.symbols),
        data_types=data_types,
        maker_fee=config.transform.maker_fee,
        taker_fee=config.transform.taker_fee,
        margin_init=config.transform.margin_init,
        margin_maint=config.transform.margin_maint,
        max_workers=workers,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_ticks = sum(r.total_ticks for r in succeeded)
    total_files = sum(r.files_processed for r in succeeded)
    errors = [err for r in failed for err in r.errors]
    all_success = len(failed) == 0

    # Write reports
    writer = ReportWriter(log_dir)
    transform_details = [
        {
            "symbol": r.symbol,
            "success": r.success,
            "files": r.files_processed,
            "size_mb": 0.0,
            "time_seconds": 0.0,
        }
        for r in results
    ]
    writer.write_step_report(
        step_name="transform",
        run_id=run_id,
        summary={
            "total_tasks": len(results),
            "successful": len(succeeded),
            "failed": len(failed),
            "total_ticks": total_ticks,
            "total_files_processed": total_files,
            "duration_seconds": round(duration, 1),
        },
        details_by_symbol=transform_details,
        errors=errors if errors else None,
    )
    writer.write_details_json(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        config=tardis_config_to_dict(config),
        steps=[
            {
                "step_name": "transform",
                "success": all_success,
                "duration_seconds": round(duration, 1),
                "summary": {
                    "total_tasks": len(results),
                    "successful": len(succeeded),
                    "failed": len(failed),
                    "total_ticks": total_ticks,
                    "total_files_processed": total_files,
                },
            }
        ],
        errors=errors if errors else None,
    )

    # Terminal summary
    click.echo()
    click.echo("=" * 70)
    duration_str = f"{duration / 60:.1f} minutes" if duration >= 60 else f"{duration:.1f} seconds"
    status = "SUCCESS" if all_success else "FAILED"
    click.echo(f"TARDIS TRANSFORM {status} in {duration_str}")
    click.echo(f"  Total ticks: {total_ticks}, Files: {total_files}")
    click.echo(f"  Reports: {log_dir}")
    click.echo("=" * 70)

    if not all_success:
        ctx.exit(EXIT_TRANSFORM_ERROR)


@cli.command("santiment-download")
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config/examples/data_santiment.yaml",
    help="Santiment configuration file path",
)
@click.option("--symbol", "-s", help="Override symbols, comma-separated")
@click.option("--from-date", help="Override start date (YYYY-MM-DD)")
@click.option("--to-date", help="Override end date (YYYY-MM-DD)")
@click.option("--force", is_flag=True, help="Re-download all (ignore checkpoints)")
@click.pass_context
def santiment_download(
    ctx: click.Context,
    config_path: str,
    symbol: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    force: bool,
) -> None:
    """Download funding rate and open interest data from SanAPI."""
    from nautilus_quants.data.config import (
        ConfigurationError,
        load_santiment_config,
    )

    overrides: dict[str, str] = {}
    if symbol:
        overrides["download.symbols"] = symbol
    if from_date:
        overrides["download.start_date"] = from_date
    if to_date:
        overrides["download.end_date"] = to_date

    try:
        config = load_santiment_config(config_path, overrides if overrides else None)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)
        return

    if force:
        # Disable checkpoint to re-download everything
        from dataclasses import replace
        config = replace(config, download=replace(config.download, checkpoint_enabled=False))

    if ctx.obj.get("dry_run"):
        click.echo("DRY RUN: Would download Santiment data:")
        click.echo(f"  Metrics: {list(config.download.metrics)}")
        click.echo(f"  Interval: {config.download.interval}")
        click.echo(f"  Date range: {config.download.start_date} to {config.download.end_date}")
        syms = list(config.download.symbols) if config.download.symbols else ["(AVAILABLE set)"]
        click.echo(f"  Symbols: {syms}")
        click.echo(f"  Output: {config.paths.raw_data}")
        return

    from nautilus_quants.data.download.santiment import SantimentDownloader

    run_id = generate_run_id()
    log_dir = create_log_dir(config.paths.logs, run_id)
    start_time = datetime.now()

    click.echo("=" * 70)
    click.echo(f"SANTIMENT DOWNLOAD: {run_id}")
    click.echo("=" * 70)
    click.echo(f"Metrics: {list(config.download.metrics)}")
    click.echo(f"Interval: {config.download.interval}")
    click.echo(f"Date range: {config.download.start_date} to {config.download.end_date}")
    n_syms = len(config.download.symbols) if config.download.symbols else 87
    click.echo(f"Symbols: {n_syms} tickers")
    click.echo(f"Log directory: {log_dir}")
    click.echo()

    try:
        downloader = SantimentDownloader(config=config.download, paths=config.paths)
        results = downloader.download_all()
    except Exception as e:
        click.echo(f"Download failed: {e}", err=True)
        ctx.exit(EXIT_DOWNLOAD_ERROR)
        return

    failed = [r for r in results if not r.success]
    succeeded = [r for r in results if r.success]
    total_rows = sum(r.rows for r in succeeded)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    click.echo()
    click.echo("=" * 70)
    duration_str = f"{duration / 60:.1f} min" if duration >= 60 else f"{duration:.0f}s"
    status = "SUCCESS" if not failed else "PARTIAL"
    click.echo(f"SANTIMENT DOWNLOAD {status} in {duration_str}")
    click.echo(f"  OK: {len(succeeded)}, Failed: {len(failed)}, Total rows: {total_rows}")
    if failed:
        for r in failed[:5]:
            click.echo(f"  ! {r.ticker}/{r.metric}: {r.error[:60]}")
    click.echo(f"  Reports: {log_dir}")
    click.echo("=" * 70)

    if failed:
        ctx.exit(EXIT_DOWNLOAD_ERROR)


@cli.command("santiment-transform")
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config/examples/data_santiment.yaml",
    help="Santiment configuration file path",
)
@click.option("--symbol", "-s", help="Override symbols, comma-separated")
@click.pass_context
def santiment_transform(
    ctx: click.Context,
    config_path: str,
    symbol: Optional[str],
) -> None:
    """Transform Santiment CSV data to Parquet format."""
    from nautilus_quants.data.config import (
        ConfigurationError,
        load_santiment_config,
    )

    overrides: dict[str, str] = {}
    if symbol:
        overrides["download.symbols"] = symbol

    try:
        config = load_santiment_config(config_path, overrides if overrides else None)
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        ctx.exit(EXIT_CONFIG_ERROR)
        return

    if ctx.obj.get("dry_run"):
        click.echo("DRY RUN: Would transform Santiment data:")
        click.echo(f"  Metrics: {list(config.download.metrics)}")
        click.echo(f"  Field mapping: {config.transform.field_mapping}")
        click.echo(f"  Catalog: {config.paths.catalog}")
        return

    from nautilus_quants.data.santiment.slug_map import AVAILABLE
    from nautilus_quants.data.transform.santiment import transform_santiment

    tickers = list(config.download.symbols) if config.download.symbols else sorted(AVAILABLE)

    click.echo("=" * 70)
    click.echo("SANTIMENT TRANSFORM")
    click.echo("=" * 70)

    has_errors = False

    for metric in config.download.metrics:
        field_name = config.transform.field_mapping.get(metric, f"san_{metric}")
        file_suffix = config.transform.file_suffix_mapping.get(metric, f"san_{metric}")

        click.echo(f"\nTransforming {metric} → {field_name} (suffix={file_suffix})...")

        results = transform_santiment(
            raw_dir=Path(config.paths.raw_data),
            catalog_path=Path(config.paths.catalog),
            tickers=tickers,
            metric=metric,
            field_name=field_name,
            file_suffix=file_suffix,
            venue=config.transform.venue,
            timeframe=config.transform.timeframe,
        )

        ok = sum(1 for r in results if r["success"])
        fail = sum(1 for r in results if not r["success"])
        total = sum(r["count"] for r in results if r["success"])
        click.echo(f"  OK: {ok}, Failed: {fail}, Total rows: {total}")

        for r in results:
            if not r["success"]:
                click.echo(f"  ! {r['ticker']}: {r.get('error', '')[:60]}", err=True)
                has_errors = True

    click.echo()
    click.echo("=" * 70)
    status = "SUCCESS" if not has_errors else "PARTIAL"
    click.echo(f"SANTIMENT TRANSFORM {status}")
    click.echo("=" * 70)

    if has_errors:
        ctx.exit(EXIT_TRANSFORM_ERROR)


def _load_config_with_overrides(
    ctx: click.Context,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    market_type: Optional[str] = None,
    funding_rate: Optional[bool] = None,
    open_interest: Optional[bool] = None,
    oi_period: Optional[str] = None,
) -> PipelineConfig:
    """Load configuration with CLI overrides applied."""
    overrides = {}
    if symbol:
        overrides["download.symbols"] = symbol
    if timeframe:
        overrides["download.timeframes"] = timeframe
    if start_date:
        overrides["download.start_date"] = start_date
    if end_date:
        overrides["download.end_date"] = end_date
    if market_type:
        overrides["download.market_type"] = market_type
    if funding_rate is not None:
        overrides["download.funding_rate"] = funding_rate
    if open_interest is not None:
        overrides["download.open_interest"] = open_interest
    if oi_period:
        overrides["download.oi_period"] = oi_period

    return load_config(ctx.obj["config_path"], overrides if overrides else None)


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
