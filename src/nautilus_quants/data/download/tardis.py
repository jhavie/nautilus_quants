"""
Tardis tick data downloader.

Downloads tick-level trade data from Tardis.dev using the official tardis-dev package.
Supports multi-symbol concurrent downloads with ThreadPoolExecutor.
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tardis_dev import datasets
from tqdm import tqdm

from nautilus_quants.data.config import TardisDownloadConfig, TardisPathsConfig


@dataclass(frozen=True)
class TardisDownloadResult:
    """Result of a single symbol download."""

    symbol: str
    success: bool
    error: str = ""


class TardisDownloader:
    """Multi-symbol concurrent downloader using tardis-dev.

    Each symbol runs in a separate thread. Within each thread,
    tardis-dev handles internal concurrency (async file downloads).

    Features provided by tardis-dev (no custom implementation needed):
    - File-level resume: existing .csv.gz files are skipped (zero requests)
    - Atomic writes: .unconfirmed temp file + os.replace()
    - Exponential backoff retry: up to 5 attempts, 429 handling
    """

    def __init__(
        self,
        config: TardisDownloadConfig,
        paths: TardisPathsConfig,
    ) -> None:
        self._config = config
        self._paths = paths

    def download_all(self) -> list[TardisDownloadResult]:
        """Download all configured symbols with tqdm progress bar.

        Returns:
            List of TardisDownloadResult, one per symbol.
        """
        symbols = self._config.symbols
        bar = tqdm(
            total=len(symbols),
            desc="Downloading",
            unit="symbol",
        )

        results: list[TardisDownloadResult] = []
        with ThreadPoolExecutor(max_workers=self._config.max_symbol_workers) as pool:
            futures = {
                pool.submit(self._download_symbol, sym): sym for sym in symbols
            }

            for future in as_completed(futures):
                sym = futures[future]
                result = future.result()
                results.append(result)
                status = "\u2713" if result.success else "\u2717"
                tqdm.write(
                    f"  {status} {sym}"
                    + (f": {result.error}" if result.error else "")
                )
                bar.update(1)

        bar.close()
        return results

    def download_symbol(self, symbol: str) -> TardisDownloadResult:
        """Public wrapper for downloading a single symbol."""
        return self._download_symbol(symbol)

    def _download_symbol(self, symbol: str) -> TardisDownloadResult:
        """Download a single symbol (runs in its own thread).

        tardis-dev internally creates an asyncio event loop and aiohttp session,
        so each thread is fully isolated and thread-safe.
        """
        api_key = os.environ.get(self._config.api_key_env, "")
        output_dir = str(Path(self._paths.raw_data) / self._config.exchange)

        def get_filename(
            exchange: str,
            data_type: str,
            date: object,
            symbol: str,
            fmt: str,
        ) -> str:
            return f"{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{fmt}.gz"

        try:
            datasets.download(
                exchange=self._config.exchange,
                data_types=list(self._config.data_types),
                symbols=[symbol],
                from_date=self._config.from_date,
                to_date=self._config.to_date,
                api_key=api_key,
                download_dir=output_dir,
                get_filename=get_filename,
                concurrency=self._config.concurrency,
            )
            return TardisDownloadResult(symbol=symbol, success=True)
        except Exception as e:
            return TardisDownloadResult(
                symbol=symbol, success=False, error=str(e)
            )

    def clean(self) -> None:
        """Remove all downloaded Tardis data for this exchange."""
        target = Path(self._paths.raw_data) / self._config.exchange
        if target.exists():
            shutil.rmtree(target)
