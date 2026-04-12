# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Santiment data downloader — fetches FR/OI metrics via sanpy.

Downloads per-ticker CSV files to ``{output_dir}/{metric}/{ticker}.csv``
with columns: ``timestamp_ms, value``.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nautilus_quants.data.config import SantimentDownloadConfig, SantimentPathsConfig
from nautilus_quants.data.santiment.slug_map import AVAILABLE, ticker_to_slug

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SantimentDownloadResult:
    """Result of a single ticker/metric download."""

    ticker: str
    metric: str
    rows: int
    success: bool
    error: str = ""


class SantimentDownloader:
    """Download metrics from SanAPI and save as CSV files.

    Uses ``san.get()`` with per-request rate limiting and checkpoint support
    (skip tickers whose CSV already exists and is non-empty).
    """

    def __init__(
        self,
        config: SantimentDownloadConfig,
        paths: SantimentPathsConfig,
    ) -> None:
        self._config = config
        self._paths = paths

        api_key = config.api_key or os.environ.get(config.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"No API key: set download.api_key in config or "
                f"export {config.api_key_env}=your_key"
            )

        import san

        san.ApiConfig.api_key = api_key
        self._san = san

    def _resolve_symbols(self) -> list[str]:
        """Resolve symbols from config. Raises if empty."""
        if not self._config.symbols:
            raise ValueError(
                "No symbols configured. List symbols explicitly in "
                "data_santiment.yaml → download.symbols"
            )
        return list(self._config.symbols)

    def download_all(self) -> list[SantimentDownloadResult]:
        """Download all configured metrics for all symbols.

        Returns:
            List of results, one per ticker/metric combination.
        """
        symbols = self._resolve_symbols()
        results: list[SantimentDownloadResult] = []

        for metric in self._config.metrics:
            logger.info(
                "Downloading %s for %d symbols (%s → %s, interval=%s)",
                metric,
                len(symbols),
                self._config.start_date,
                self._config.end_date,
                self._config.interval,
            )
            metric_results = self._download_metric(metric, symbols)
            results.extend(metric_results)

        return results

    def _download_metric(
        self,
        metric: str,
        symbols: list[str],
    ) -> list[SantimentDownloadResult]:
        """Download a single metric for all symbols."""
        output_dir = Path(self._paths.raw_data) / metric
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[SantimentDownloadResult] = []
        delay = self._config.rate_limit.delay_seconds

        for i, ticker in enumerate(symbols):
            csv_path = output_dir / f"{ticker}.csv"

            # Checkpoint: skip if file exists and non-empty
            if self._config.checkpoint_enabled and csv_path.exists():
                if csv_path.stat().st_size > 50:
                    logger.debug("Checkpoint skip: %s/%s", metric, ticker)
                    rows = _count_csv_rows(csv_path)
                    results.append(
                        SantimentDownloadResult(
                            ticker=ticker,
                            metric=metric,
                            rows=rows,
                            success=True,
                        )
                    )
                    continue

            slug = ticker_to_slug(ticker)
            if slug is None:
                results.append(
                    SantimentDownloadResult(
                        ticker=ticker,
                        metric=metric,
                        rows=0,
                        success=False,
                        error=f"No slug mapping for {ticker}",
                    )
                )
                continue

            result = self._download_one(metric, ticker, slug, csv_path)
            results.append(result)

            ok = "ok" if result.success else "FAIL"
            logger.info(
                "[%d/%d] %s %s/%s rows=%d %s",
                i + 1,
                len(symbols),
                ok,
                metric,
                ticker,
                result.rows,
                result.error[:60] if result.error else "",
            )

            if i < len(symbols) - 1:
                time.sleep(delay)

        return results

    def _download_one(
        self,
        metric: str,
        ticker: str,
        slug: str,
        csv_path: Path,
    ) -> SantimentDownloadResult:
        """Download a single ticker/metric with retry."""
        max_retries = self._config.rate_limit.max_retries
        backoff = self._config.rate_limit.backoff_multiplier
        wait = self._config.rate_limit.delay_seconds

        for attempt in range(1, max_retries + 1):
            try:
                df = self._san.get(
                    f"{metric}/{slug}",
                    from_date=self._config.start_date,
                    to_date=self._config.end_date,
                    interval=self._config.interval,
                )

                if df is None or df.empty:
                    return SantimentDownloadResult(
                        ticker=ticker,
                        metric=metric,
                        rows=0,
                        success=True,
                    )

                # Save as CSV: timestamp_ms, value
                ts_index = pd.DatetimeIndex(df.index)
                if ts_index.tz is None:
                    ts_index = ts_index.tz_localize("UTC")
                ts_ms = ts_index.astype("int64") // 1_000_000

                out = pd.DataFrame(
                    {
                        "timestamp_ms": ts_ms,
                        "value": df["value"].values,
                    }
                )
                out.to_csv(csv_path, index=False)

                return SantimentDownloadResult(
                    ticker=ticker,
                    metric=metric,
                    rows=len(out),
                    success=True,
                )

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Rate Limit" in err_msg:
                    if attempt < max_retries:
                        sleep_time = wait * (backoff ** (attempt - 1))
                        logger.warning(
                            "Rate limited on %s/%s, retry %d/%d in %.0fs",
                            metric,
                            ticker,
                            attempt,
                            max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                        continue

                return SantimentDownloadResult(
                    ticker=ticker,
                    metric=metric,
                    rows=0,
                    success=False,
                    error=err_msg[:200],
                )

        return SantimentDownloadResult(
            ticker=ticker,
            metric=metric,
            rows=0,
            success=False,
            error="Max retries exceeded",
        )


def _count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file (excluding header)."""
    try:
        with open(path) as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0
