# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Unified extra data framework — ExtraDataConfig, loaders, and ExtraDataManager.

Supplements OHLCV panels with external data sources (funding rate, open interest,
BinanceBar fields, broadcast market variables, etc.).

Two consumption modes:
- **Batch** (alpha analyze/mining): ``ExtraDataManager.inject_panels()`` loads all
  sources and injects into ``panel_fields``.
- **Streaming** (backtest/live): ``FactorEngineActor`` reads config, sets up
  per-bar enrichment; broadcast uses ``Buffer.inject_staged_field()`` at flush time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

logger = logging.getLogger(__name__)


# ── Data Model ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtraDataConfig:
    """Extra data field — supplements OHLCV with external data.

    Attributes:
        name: Field name in panel (e.g. "funding_rate", "btc_close").
        source: Loading strategy.
            "bar"       — extract from Bar objects (e.g. BinanceBar quote_volume)
            "catalog"   — load from NautilusTrader ParquetDataCatalog
            "parquet"   — load from external parquet directory
            "broadcast" — broadcast one instrument's column to all
        instruments: Instrument list.
            - broadcast: ["BTC"] — single pattern, prefix match against panel columns
            - catalog/parquet: ["BTCUSDT.BINANCE", ...] — filter to specific instruments
              (empty = all universe instruments)
            - bar: ignored (extracted from whatever bars are loaded)
        path: Data directory or catalog path for this source.
        timeframe: Data granularity (e.g. "4h") for sources with different
            frequency than bars.
        fill: Fill method for missing timestamps — "ffill" or "zero".
    """

    name: str
    source: str
    instruments: list[str] = field(default_factory=list)
    path: str = ""
    timeframe: str = ""
    fill: str = "ffill"


# ── Config Loading ───────────────────────────────────────────────────────


def load_extra_data_config(path: str | Path) -> list[ExtraDataConfig]:
    """Load extra data configuration from YAML file.

    Supports full dict syntax and shorthand (``funding_rate: catalog``).

    Args:
        path: Path to extra_data YAML file.

    Returns:
        List of ExtraDataConfig entries.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Extra data config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return parse_extra_data_raw(raw)


def parse_extra_data_raw(raw: dict | None) -> list[ExtraDataConfig]:
    """Parse raw YAML dict into ExtraDataConfig list.

    Supports two formats:
    - Full: ``funding_rate: {source: catalog, path: ...}``
    - Shorthand: ``funding_rate: catalog``
    """
    if not raw:
        return []

    configs: list[ExtraDataConfig] = []
    for name, value in raw.items():
        if isinstance(value, str):
            # Shorthand: "funding_rate: catalog"
            configs.append(ExtraDataConfig(name=name, source=value))
        elif isinstance(value, dict):
            instruments = value.get("instruments", [])
            if isinstance(instruments, str):
                instruments = [instruments]
            configs.append(ExtraDataConfig(
                name=name,
                source=value.get("source", ""),
                instruments=instruments,
                path=value.get("path", ""),
                timeframe=value.get("timeframe", ""),
                fill=value.get("fill", "ffill"),
            ))
        else:
            logger.warning("Skipping invalid extra_data entry: %s", name)

    return configs


# ── Loader Functions ─────────────────────────────────────────────────────


def _load_broadcast(
    cfg: ExtraDataConfig,
    close_panel: pd.DataFrame,
) -> pd.DataFrame | None:
    """Extract one instrument's column and broadcast to all columns.

    Matching: exact match first, then prefix match (case-insensitive).
    """
    if not cfg.instruments:
        logger.warning("Broadcast '%s' has no instruments configured", cfg.name)
        return None

    pattern = cfg.instruments[0]
    columns = list(close_panel.columns)

    # Exact match
    if pattern in columns:
        matched = pattern
    else:
        # Prefix match (case-insensitive)
        pattern_upper = pattern.upper()
        matched = next(
            (c for c in columns if c.upper().startswith(pattern_upper)),
            None,
        )

    if matched is None:
        logger.warning(
            "Broadcast '%s': instrument '%s' not found in panel columns",
            cfg.name, pattern,
        )
        return None

    col_data = close_panel[matched]
    result = pd.DataFrame(
        {c: col_data for c in columns},
        index=close_panel.index,
    )
    logger.info("Injected broadcast '%s' from %s", cfg.name, matched)
    return result


def _load_bar_field(
    cfg: ExtraDataConfig,
    bars_by_instrument: dict[str, list[Any]] | None,
    close_panel: pd.DataFrame,
    catalog_path: str = "",
) -> pd.DataFrame | None:
    """Extract a field from Bar objects and build a panel.

    Two loading strategies (tried in order):
    1. Extract from Bar/BinanceBar objects (live/backtest: BinanceBar has extras)
    2. Read from catalog's ``binance_bar`` parquet files (analyze: catalog returns
       plain Bar without extras, but parquet stores quote_volume etc.)

    Args:
        cfg: Extra data config with field name.
        bars_by_instrument: Raw Bar objects (may be plain Bar or BinanceBar).
        close_panel: Close panel for alignment.
        catalog_path: Catalog root path for parquet fallback.
    """
    field_name = cfg.name

    # Strategy 1: Extract from Bar objects (works for BinanceBar in live)
    if bars_by_instrument:
        series_dict: dict[str, pd.Series] = {}
        for inst_id, bars in bars_by_instrument.items():
            if not bars:
                continue
            timestamps: list[pd.Timestamp] = []
            values: list[float] = []
            for bar in bars:
                d = type(bar).to_dict(bar)
                ts = pd.Timestamp(d["ts_event"], unit="ns")
                val = d.get(field_name)
                if val is None:
                    val = getattr(bar, field_name, None)
                if val is not None:
                    timestamps.append(ts)
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        continue
            if timestamps:
                series_dict[inst_id] = pd.Series(values, index=timestamps)

        if series_dict:
            panel = pd.DataFrame(series_dict)
            fill_method = "ffill" if cfg.fill == "ffill" else None
            panel = panel.reindex(close_panel.index, method=fill_method)
            for col in close_panel.columns:
                if col not in panel.columns:
                    panel[col] = 0.0
            result = panel[close_panel.columns]
            logger.info("Injected bar field '%s' from objects: %s", field_name, result.shape)
            return result

    # Strategy 2: Read from catalog's binance_bar parquet (analyze fallback)
    path = cfg.path or catalog_path
    if path:
        result = _load_bar_field_from_parquet(field_name, path, close_panel)
        if result is not None:
            return result

    logger.warning("Bar field '%s' not found (no objects, no catalog parquet)", field_name)
    return None


def _load_bar_field_from_parquet(
    field_name: str,
    catalog_path: str,
    close_panel: pd.DataFrame,
) -> pd.DataFrame | None:
    """Read an extra field from catalog's binance_bar parquet files."""
    import pyarrow.parquet as pq

    binance_bar_dir = Path(catalog_path) / "data" / "binance_bar"
    if not binance_bar_dir.exists():
        return None

    series_dict: dict[str, pd.Series] = {}
    for inst_dir in sorted(binance_bar_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        # Extract instrument ID from dir name (e.g. "BTCUSDT.BINANCE-4-HOUR-LAST-EXTERNAL")
        inst_id = inst_dir.name.split("-")[0]
        if inst_id not in close_panel.columns:
            continue

        parquet_files = list(inst_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        try:
            table = pq.read_table(
                str(parquet_files[0]),
                columns=["ts_event", field_name],
            )
        except (KeyError, Exception):
            continue

        df = table.to_pandas()
        if field_name not in df.columns:
            continue

        df["ts"] = pd.to_datetime(df["ts_event"], unit="ns")
        df[field_name] = pd.to_numeric(df[field_name], errors="coerce")
        # Filter to close_panel date range
        ts_min, ts_max = close_panel.index.min(), close_panel.index.max()
        df = df[(df["ts"] >= ts_min) & (df["ts"] <= ts_max)]
        if len(df) > 0:
            series_dict[inst_id] = pd.Series(
                df[field_name].values, index=df["ts"].values,
            )

    if not series_dict:
        return None

    panel = pd.DataFrame(series_dict)
    panel = panel.reindex(close_panel.index, method="ffill")
    for col in close_panel.columns:
        if col not in panel.columns:
            panel[col] = 0.0
    result = panel[close_panel.columns]
    logger.info("Injected bar field '%s' from parquet: %s", field_name, result.shape)
    return result


def _load_catalog_field(
    cfg: ExtraDataConfig,
    catalog_path: str,
    instruments: list[str],
    close_panel: pd.DataFrame,
) -> pd.DataFrame | None:
    """Load FundingRateUpdate from catalog and build aligned panel.

    Refactored from ``FactorEvaluator._load_funding_rate_panel()``.
    """
    if not catalog_path:
        logger.warning("Catalog field '%s': no catalog_path provided", cfg.name)
        return None

    try:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        catalog = ParquetDataCatalog(catalog_path)
        fr_data: list = []
        target_instruments = cfg.instruments or instruments
        for inst in target_instruments:
            result = catalog.funding_rates(instrument_ids=[inst])
            if result:
                fr_data.extend(result)

        if not fr_data:
            logger.warning("Catalog field '%s': no data found", cfg.name)
            return None

        fr_dict: dict[str, dict[pd.Timestamp, float]] = {}
        for fru in fr_data:
            inst_id = str(fru.instrument_id)
            ts = pd.Timestamp(fru.ts_event, unit="ns")
            if inst_id not in fr_dict:
                fr_dict[inst_id] = {}
            fr_dict[inst_id][ts] = float(fru.rate)

        if not fr_dict:
            return None

        fr_panel = pd.DataFrame(
            {k: pd.Series(v) for k, v in fr_dict.items()}
        )
        fr_panel = fr_panel.reindex(close_panel.index, method="ffill")
        for col in close_panel.columns:
            if col not in fr_panel.columns:
                fr_panel[col] = 0.0
        result = fr_panel[close_panel.columns]
        logger.info("Injected catalog field '%s': %s", cfg.name, result.shape)
        return result
    except Exception as e:
        logger.warning("Failed to load catalog field '%s': %s", cfg.name, e)
        return None


def _load_parquet_field(
    cfg: ExtraDataConfig,
    instruments: list[str],
    close_panel: pd.DataFrame,
) -> pd.DataFrame | None:
    """Load data from parquet directory and build aligned panel.

    Refactored from ``FactorEvaluator._load_oi_panel()``.
    """
    if not cfg.path:
        logger.warning("Parquet field '%s': no path provided", cfg.name)
        return None

    try:
        from nautilus_quants.data.transform.open_interest import (
            load_oi_lookup,
        )

        target_instruments = cfg.instruments or instruments
        timeframe = cfg.timeframe or "4h"
        lookup = load_oi_lookup(cfg.path, target_instruments, timeframe)

        oi_series: dict[str, pd.Series] = {}
        for inst_id, ts_map in lookup.items():
            if not ts_map:
                continue
            timestamps = [pd.Timestamp(ts, unit="ns") for ts in ts_map]
            values = [d["open_interest"] for d in ts_map.values()]
            oi_series[inst_id] = pd.Series(values, index=timestamps)

        if not oi_series:
            return None

        oi_panel = pd.DataFrame(oi_series)
        oi_panel = oi_panel.reindex(close_panel.index, method="ffill")
        for col in close_panel.columns:
            if col not in oi_panel.columns:
                oi_panel[col] = 0.0
        result = oi_panel[close_panel.columns]
        logger.info("Injected parquet field '%s': %s", cfg.name, result.shape)
        return result
    except Exception as e:
        logger.warning("Failed to load parquet field '%s': %s", cfg.name, e)
        return None


# ── ExtraDataManager ─────────────────────────────────────────────────────


class ExtraDataManager:
    """Unified batch loader for all extra data sources.

    Used by ``FactorEvaluator`` (alpha analyze/mining/regime).
    All source types including broadcast are handled here in batch mode.

    For live/backtest, ``FactorEngineActor`` handles streaming data
    acquisition and uses ``Buffer.inject_staged_field()`` for broadcast
    — this class is NOT used in the live path.
    """

    def __init__(self, configs: list[ExtraDataConfig]) -> None:
        self._configs = configs

    def inject_panels(
        self,
        panel_fields: dict[str, pd.DataFrame | float],
        instruments: list[str],
        *,
        bars_by_instrument: dict[str, list[Any]] | None = None,
        catalog_path: str = "",
    ) -> None:
        """Load all configured extra data and inject into panel_fields.

        Args:
            panel_fields: Mutable dict to inject panels into.
            instruments: Sorted instrument IDs in the universe.
            bars_by_instrument: Raw Bar objects (for ``bar`` source).
            catalog_path: NT catalog path (for ``catalog`` source).
        """
        close_panel = panel_fields.get("close")
        if not isinstance(close_panel, pd.DataFrame) or close_panel.empty:
            return

        for cfg in self._configs:
            try:
                panel = self._load_one(
                    cfg, close_panel, instruments,
                    bars_by_instrument, catalog_path,
                )
                if panel is not None:
                    panel_fields[cfg.name] = panel
            except Exception:
                logger.warning(
                    "Extra data '%s' (%s) failed",
                    cfg.name, cfg.source, exc_info=True,
                )

    def _load_one(
        self,
        cfg: ExtraDataConfig,
        close_panel: pd.DataFrame,
        instruments: list[str],
        bars_by_instrument: dict[str, list[Any]] | None,
        catalog_path: str,
    ) -> pd.DataFrame | None:
        if cfg.source == "broadcast":
            return _load_broadcast(cfg, close_panel)
        if cfg.source == "bar":
            return _load_bar_field(cfg, bars_by_instrument, close_panel, catalog_path)
        if cfg.source == "catalog":
            return _load_catalog_field(cfg, catalog_path, instruments, close_panel)
        if cfg.source == "parquet":
            return _load_parquet_field(cfg, instruments, close_panel)
        logger.warning("Unknown extra_data source '%s' for '%s'", cfg.source, cfg.name)
        return None
