# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor cache — lazy pre-computation and persistence.

Provides save/load utilities for factor values in Parquet format.
Consumed by both ``alpha analyze`` (IC/ICIR analysis) and
``FactorEngineActor`` (backtest replay).

Storage layout::

    {cache_dir}/
        metadata.yaml      # config hash, creation time, stats
        factors.parquet     # MultiIndex[ts_event_ns, instrument_id] × factor columns
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from nautilus_quants.factors.config import FactorConfig

logger = logging.getLogger(__name__)

_PARQUET_FILE = "factors.parquet"
_METADATA_FILE = "metadata.yaml"


# ---------------------------------------------------------------------------
# Cache key / hash
# ---------------------------------------------------------------------------


def compute_config_hash(factor_config: FactorConfig) -> str:
    """Compute a deterministic hash of factor computation logic.

    Includes only fields that affect factor values: expressions,
    variables, and parameters.  Excludes cosmetic fields (description,
    category, performance) and runtime context (bar_spec, catalog_path).
    """
    canonical = {
        "variables": dict(factor_config.variables),
        "factors": {
            f.name: f.expression for f in factor_config.all_factors
        },
        "parameters": dict(factor_config.parameters),
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_cache_key(
    factor_config: FactorConfig,
    bar_spec: str,
    instrument_ids: list[str],
    catalog_path: str,
) -> str:
    """Compute a full SHA-256 cache key (alpha module use).

    Includes computation logic *and* data source context.
    """
    canonical = {
        "variables": dict(factor_config.variables),
        "factors": {
            f.name: f.expression for f in factor_config.all_factors
        },
        "parameters": dict(factor_config.parameters),
        "bar_spec": bar_spec,
        "instrument_ids": sorted(instrument_ids),
        "catalog_path": str(Path(catalog_path).resolve()),
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_cache(
    cache_dir: str | Path,
    current_config_hash: str,
    expected_instruments: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate cache against current configuration.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, warnings)``

        - ``is_valid=False`` when config_hash mismatches (cache stale).
        - ``is_valid=True`` with warnings when instruments are missing.
    """
    meta = load_cache_metadata(cache_dir)
    warnings: list[str] = []

    # Config hash check
    stored_hash = meta.get("config_hash", "")
    if stored_hash and stored_hash != current_config_hash:
        return False, [
            f"Config hash mismatch: cached={stored_hash}, "
            f"current={current_config_hash}"
        ]

    # Instruments coverage check
    if expected_instruments:
        cached_instruments = set(meta.get("instruments", []))
        if cached_instruments:
            missing = expected_instruments - cached_instruments
            if missing:
                warnings.append(
                    f"Cache missing {len(missing)} instruments: "
                    f"{sorted(missing)}"
                )

    return True, warnings


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _build_metadata(
    df: pd.DataFrame,
    *,
    factor_config_path: str = "",
    config_hash: str = "",
) -> dict:
    """Build metadata dict from a factor DataFrame."""
    unique_ts = df.index.get_level_values("ts_event_ns").unique()
    unique_insts = df.index.get_level_values("instrument_id").unique()
    return {
        "version": "1.0",
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "factor_config_path": factor_config_path,
        "factor_names": list(df.columns),
        "instruments": sorted(unique_insts.tolist()),
        "instrument_count": len(unique_insts),
        "timestamp_count": len(unique_ts),
        "timestamp_range": {
            "start_ns": int(unique_ts.min()),
            "end_ns": int(unique_ts.max()),
        },
    }


def save_factor_cache(
    factor_series: dict[str, pd.Series],
    cache_dir: str | Path,
    *,
    factor_config_path: str = "",
    config_hash: str = "",
) -> None:
    """Persist factor values to Parquet + metadata.

    Parameters
    ----------
    factor_series
        ``{factor_name: pd.Series(MultiIndex[date, asset])}``,
        the output format of ``FactorEvaluator.evaluate()``.
    cache_dir
        Target directory (created if missing).
    factor_config_path, config_hash
        Metadata written alongside the Parquet file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Combine all factor series into one DataFrame
    df = pd.DataFrame(factor_series)  # columns = factor names
    # Ensure index names are consistent
    df.index.names = ["date", "asset"]

    # Convert datetime index level to int64 nanoseconds for Nautilus compatibility
    dates = df.index.get_level_values("date")
    assets = df.index.get_level_values("asset")
    ts_ns = dates.astype("int64")  # nanoseconds since epoch
    new_index = pd.MultiIndex.from_arrays(
        [ts_ns, assets], names=["ts_event_ns", "instrument_id"],
    )
    df.index = new_index

    # Save Parquet
    df.to_parquet(cache_dir / _PARQUET_FILE, engine="pyarrow")

    # Save metadata
    metadata = _build_metadata(
        df, factor_config_path=factor_config_path, config_hash=config_hash,
    )
    with open(cache_dir / _METADATA_FILE, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    logger.info(
        "Factor cache saved: %s (%d instruments × %d timestamps × %d factors)",
        cache_dir,
        metadata["instrument_count"],
        metadata["timestamp_count"],
        len(df.columns),
    )


def save_snapshots_as_cache(
    snapshots: list[tuple[int, dict[str, dict[str, float]]]],
    cache_dir: str | Path,
    *,
    factor_config_path: str = "",
    config_hash: str = "",
) -> None:
    """Persist FactorEngineActor snapshots to cache.

    Converts ``[(ts_ns, {factor: {inst: value}}), ...]`` to the same
    Parquet format used by ``save_factor_cache``.
    """
    if not snapshots:
        return

    rows: list[dict] = []
    for ts_ns, factors in snapshots:
        # Collect all instruments across all factors for this timestamp
        all_insts: set[str] = set()
        for fvals in factors.values():
            all_insts.update(fvals.keys())
        for inst in sorted(all_insts):
            row: dict = {"ts_event_ns": ts_ns, "instrument_id": inst}
            for fname, fvals in factors.items():
                row[fname] = fvals.get(inst, float("nan"))
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index(["ts_event_ns", "instrument_id"])

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / _PARQUET_FILE, engine="pyarrow")

    metadata = _build_metadata(
        df, factor_config_path=factor_config_path, config_hash=config_hash,
    )
    with open(cache_dir / _METADATA_FILE, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    logger.info(
        "Factor cache saved (from snapshots): %s "
        "(%d instruments × %d timestamps × %d factors)",
        cache_dir,
        metadata["instrument_count"],
        metadata["timestamp_count"],
        len(df.columns),
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def has_cache(cache_dir: str | Path) -> bool:
    """Check whether a valid factor cache exists at *cache_dir*."""
    return (Path(cache_dir) / _PARQUET_FILE).is_file()


def load_as_factor_series(cache_dir: str | Path) -> dict[str, pd.Series]:
    """Load cache as ``{factor_name: Series(MultiIndex[date, asset])}``.

    This is the format consumed by ``alpha analyze`` / alphalens.
    """
    df = pd.read_parquet(Path(cache_dir) / _PARQUET_FILE, engine="pyarrow")

    # Convert int64-ns index back to DatetimeIndex
    ts_ns = df.index.get_level_values("ts_event_ns")
    dates = pd.to_datetime(ts_ns, unit="ns")
    assets = df.index.get_level_values("instrument_id")
    df.index = pd.MultiIndex.from_arrays([dates, assets], names=["date", "asset"])

    result: dict[str, pd.Series] = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) > 0:
            result[col] = series.astype(float)
    return result


def load_as_snapshots(
    cache_dir: str | Path,
) -> dict[int, dict[str, dict[str, float]]]:
    """Load cache as ``{ts_ns: {factor_name: {instrument_id: value}}}``.

    This is the format consumed by ``FactorEngineActor._flush_and_publish``.
    """
    df = pd.read_parquet(Path(cache_dir) / _PARQUET_FILE, engine="pyarrow")

    result: dict[int, dict[str, dict[str, float]]] = {}
    factor_names = list(df.columns)

    for (ts_ns, inst_id), row in df.iterrows():
        ts_int = int(ts_ns)
        if ts_int not in result:
            result[ts_int] = {fname: {} for fname in factor_names}
        for fname in factor_names:
            val = row[fname]
            if pd.notna(val):
                result[ts_int][fname][inst_id] = float(val)

    return result


def load_cache_metadata(cache_dir: str | Path) -> dict:
    """Load metadata.yaml from a cache directory."""
    meta_path = Path(cache_dir) / _METADATA_FILE
    if not meta_path.is_file():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
