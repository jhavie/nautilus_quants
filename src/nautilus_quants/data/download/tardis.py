"""
Tardis tick data downloader.

Downloads tick-level trade data from Tardis.dev using the official tardis-dev package.
Supports multi-symbol concurrent downloads with ThreadPoolExecutor.
"""

import logging
import os
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import socks
from tardis_dev import datasets
from tqdm import tqdm

from nautilus_quants.data.config import TardisDownloadConfig, TardisPathsConfig

logger = logging.getLogger(__name__)


def _resolve_api_key(raw: str) -> str:
    """Resolve API key: env var name → lookup, literal key → use directly.

    Heuristic: values starting with "TD." are Tardis API keys;
    otherwise treat as environment variable name.
    """
    if raw.startswith("TD."):
        return raw
    return os.environ.get(raw, "")


class SocksProxy:
    """SOCKS5 proxy context manager for transparent socket-level proxying.

    Supports two proxy URL schemes:
    - "ssh://hostname" → auto-starts `ssh -D <port> -N hostname` tunnel
    - "socks5://host:port" → uses an existing SOCKS5 proxy directly

    Patches socket.socket globally with PySocks so aiohttp (used by tardis-dev)
    routes all TCP connections through the proxy without any code changes.
    """

    _SSH_STARTUP_SECONDS = 2

    def __init__(self, proxy_url: str) -> None:
        self._proxy_url = proxy_url
        self._ssh_proc: subprocess.Popen | None = None
        self._original_socket: type | None = None
        self._proxy_host = "127.0.0.1"
        self._proxy_port = 1080
        self._we_started_ssh = False

    @staticmethod
    def _port_is_listening(host: str, port: int) -> bool:
        """Check if a port is already listening (e.g. existing SSH tunnel)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def __enter__(self) -> "SocksProxy":
        if self._proxy_url.startswith("ssh://"):
            host = self._proxy_url[6:]

            if self._port_is_listening(self._proxy_host, self._proxy_port):
                logger.info(
                    "SOCKS5 port %d already listening, reusing existing tunnel",
                    self._proxy_port,
                )
            else:
                logger.info("Starting SSH SOCKS5 tunnel → %s (port %d)", host, self._proxy_port)
                self._ssh_proc = subprocess.Popen(
                    [
                        "ssh", "-D", str(self._proxy_port), "-N",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ExitOnForwardFailure=yes",
                        host,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                time.sleep(self._SSH_STARTUP_SECONDS)
                if self._ssh_proc.poll() is not None:
                    stderr = self._ssh_proc.stderr.read().decode() if self._ssh_proc.stderr else ""
                    raise RuntimeError(f"SSH tunnel failed to start: {stderr}")
                self._we_started_ssh = True

        elif self._proxy_url.startswith("socks5://"):
            parsed = urlparse(self._proxy_url)
            self._proxy_host = parsed.hostname or "127.0.0.1"
            self._proxy_port = parsed.port or 1080
            logger.info("Using SOCKS5 proxy %s:%d", self._proxy_host, self._proxy_port)

        else:
            raise ValueError(f"Unsupported proxy scheme: {self._proxy_url!r} (use ssh:// or socks5://)")

        self._original_socket = socket.socket
        socks.set_default_proxy(socks.SOCKS5, self._proxy_host, self._proxy_port)
        socket.socket = socks.socksocket  # type: ignore[assignment]
        logger.info("Global socket patched → SOCKS5 %s:%d", self._proxy_host, self._proxy_port)
        return self

    def __exit__(self, *args: object) -> None:
        if self._original_socket is not None:
            socket.socket = self._original_socket  # type: ignore[assignment]
            logger.info("Global socket restored")

        if self._ssh_proc is not None and self._we_started_ssh:
            self._ssh_proc.terminate()
            self._ssh_proc.wait(timeout=5)
            logger.info("SSH tunnel process terminated")


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

        When proxy is configured, all TCP connections are transparently routed
        through a SOCKS5 proxy (socket-level patching via PySocks).

        Progress tracks each (symbol, data_type) pair for granular feedback.

        Returns:
            List of TardisDownloadResult, one per (symbol, data_type).
        """
        proxy_url = self._config.proxy
        ctx = SocksProxy(proxy_url) if proxy_url else nullcontext()

        with ctx:
            symbols = self._config.symbols
            data_types = self._config.data_types

            # Build work units: (symbol, data_type)
            work_units = [
                (sym, dt) for sym in symbols for dt in data_types
            ]

            bar = tqdm(
                total=len(work_units),
                desc="Downloading",
                unit="task",
            )

            results: list[TardisDownloadResult] = []
            with ThreadPoolExecutor(max_workers=self._config.max_symbol_workers) as pool:
                futures = {
                    pool.submit(self._download_symbol_datatype, sym, dt): (sym, dt)
                    for sym, dt in work_units
                }

                for future in as_completed(futures):
                    sym, dt = futures[future]
                    result = future.result()
                    results.append(result)
                    status = "\u2713" if result.success else "\u2717"
                    tqdm.write(
                        f"  {status} {sym}/{dt}"
                        + (f": {result.error}" if result.error else "")
                    )
                    bar.update(1)

            bar.close()
        return results

    def download_symbol(self, symbol: str) -> TardisDownloadResult:
        """Public wrapper for downloading a single symbol (all data types)."""
        return self._download_symbol_datatype(symbol)

    def _download_symbol_datatype(
        self, symbol: str, data_type: str | None = None,
    ) -> TardisDownloadResult:
        """Download a single (symbol, data_type) pair.

        Args:
            symbol: Trading symbol (e.g., "ETHUSDT")
            data_type: Single data type (e.g., "trades", "quotes").
                       If None, downloads all configured data_types.
        """
        api_key = _resolve_api_key(self._config.api_key_env)
        output_dir = str(Path(self._paths.raw_data) / self._config.exchange)
        dl_types = [data_type] if data_type else list(self._config.data_types)

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
                data_types=dl_types,
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
