# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FactorEngineActor - Nautilus Actor wrapper for FactorEngine.

This module provides a Nautilus-native Actor that wraps the FactorEngine,
enabling seamless integration with the Nautilus trading system.

The Panel architecture evaluates all factor expressions (including nested
CS/TS operators like ``correlation(rank(open), rank(volume), 10)``) correctly
via pd.DataFrame[T x N] intermediates.

Constitution Compliance:
    - Extends Nautilus Actor base class (Principle I)
    - Uses Nautilus Logger, Clock, MessageBus (Principle I)
    - Configuration-driven via ActorConfig (Principle II)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.common.events import TimeEvent
from nautilus_trader.model.data import Bar, BarType, DataType, FundingRateUpdate
from nautilus_trader.model.identifiers import InstrumentId

from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.factors.config import FactorConfig, load_factor_config, validate_factor_config
from nautilus_quants.factors.engine.extra_data import (
    ExtraDataConfig,
    load_extra_data_config,
    load_lookup,
)
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.types import FactorValues
from nautilus_quants.portfolio.monitor.factor import compute_factor_ic
from nautilus_quants.utils.cache_keys import FACTOR_IC_CACHE_KEY, FACTOR_VALUES_CACHE_KEY

# Timer name for the hot-reload check.
_HOT_RELOAD_TIMER = "factor_config_hot_reload"


def _try_reload_factors(
    config_path: str,
    last_mtime: float,
) -> tuple[FactorConfig, float] | None:
    """Check whether the factor config file has changed and reload if so.

    Returns ``(new_config, new_mtime)`` on success, or ``None`` if the file
    hasn't changed, doesn't exist, or fails validation.

    This function is intentionally extracted from the Actor class so that it
    can be unit-tested without Nautilus infrastructure.
    """
    path = Path(config_path)
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        return None

    if current_mtime == last_mtime:
        return None

    try:
        new_config = load_factor_config(config_path)
        errors = validate_factor_config(new_config)
        if errors:
            return None
    except Exception:
        return None

    return new_config, current_mtime


def _detect_extra_bar_fields(bar: Bar) -> list[str]:
    """Detect extra fields from BinanceBar via __dict__ introspection.

    BinanceBar stores its extra fields (quote_volume, count, etc.) in
    ``__dict__``, while standard OHLCV fields are Cython getset descriptors
    and therefore absent from ``__dict__``.

    Returns a sorted list of extra field names, or an empty list for
    non-BinanceBar instances or if the import fails.
    """
    try:
        from nautilus_trader.adapters.binance.common.types import BinanceBar
    except ImportError:
        return []

    if not isinstance(bar, BinanceBar):
        return []

    return sorted(k for k in bar.__dict__ if not k.startswith("_"))


def _extract_bar_data(bar: Bar) -> dict[str, float]:
    """Extract OHLCV data (and extra BinanceBar fields) from a bar into a plain dict."""
    data = {
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume),
    }
    # BinanceBar stores extra fields (quote_volume, count, etc.) in __dict__
    for k, v in getattr(bar, "__dict__", {}).items():
        if not k.startswith("_") and k not in data:
            try:
                data[k] = float(v)
            except (TypeError, ValueError):
                pass
    return data


class FactorEngineActorConfig(ActorConfig, frozen=True):
    """
    Configuration for FactorEngineActor.

    Parameters
    ----------
    factor_config_path : str
        Path to YAML factor configuration file (required).
    data_cls : str, default ""
        Full class path of the bar data type to subscribe to
        (e.g., "nautilus_trader.adapters.binance.common.types:BinanceBar").
        Used by CLI to precisely filter bar_types from the data: section.
        If empty, CLI injects all available bar_types.
    bar_spec : str, default ""
        Target timeframe for factor computation (e.g., "1h", "4h", "1d").
        Used with data_cls by CLI to filter matching bar_types.
        If empty, all bar_types matching data_cls are injected.
    max_history : int, default 500
        Maximum history to maintain per instrument.
    publish_signals : bool, default True
        Whether to publish factor values as CustomData.
    signal_prefix : str, default "factor"
        Prefix for signal names (e.g., "factor.breakout").
    bar_types : list[str], default []
        List of bar type strings to subscribe to (injected by CLI from data config).
        Filtered by data_cls + bar_spec when both are specified.
        If empty, actor will log an error and not start.
    flush_timeout_secs : int, default 5
        Seconds to wait for all instruments before force-flushing.
        Backtest: TestClock inserts a ts+5s alert within bar intervals (1h/4h),
        delay is negligible. Live: override to e.g. 180 for WebSocket feeds.
        Set to 0 to disable timeout (relies on set-complete + timestamp-advance).
    factor_cache_path : str, default ""
        Path to a pre-computed factor cache directory.  If set and the
        directory contains ``factors.parquet``, factor values are loaded
        from cache instead of being computed on each bar.  If set but the
        directory is empty, factor values are computed normally and saved
        to this path on actor stop for future runs.
    subscribe_funding_rates : bool, default False
        Whether to subscribe to FundingRateUpdate events and inject the
        latest funding rate into bar_data as ``funding_rate`` field.
    oi_data_path : str, default ""
        Path to open interest Parquet directory. If set, OI data is
        loaded at startup and injected into bar_data as ``open_interest``
        field on each bar.
    """

    factor_config_path: str
    data_cls: str = ""
    bar_spec: str = ""
    max_history: int = 500
    publish_signals: bool = True
    signal_prefix: str = "factor"
    bar_types: list[str] = []
    flush_timeout_secs: int = 5
    factor_cache_path: str = ""
    enable_hot_reload: bool = False
    hot_reload_interval_secs: int = 300
    extra_data_path: str = ""
    # Deprecated: kept for backward compat (use extra_data_path instead)
    subscribe_funding_rates: bool = False
    oi_data_path: str = ""


class FactorEngineActor(BarSubscriptionMixin, Actor):
    """
    Nautilus Actor that computes factors and publishes results.

    This actor wraps the FactorEngine and integrates it with the Nautilus
    trading system, providing:
    - Direct bar subscription (no aggregation - BinanceBar fields preserved)
    - Panel-based factor computation (CS + TS evaluated together)
    - CustomData publishing for factor values
    - Performance logging via Nautilus Logger

    BinanceBar usage pattern (analogous to OrderBook):
        FactorEngineActor subscribes directly to BinanceBar (1h),
        preserving extra fields like quote_volume and count for factor
        computation. This is the same pattern as using OrderBook for
        microstructure alpha - never aggregated, subscribed directly.

    Example
    -------
    ```python
    from nautilus_quants.actors import FactorEngineActor, FactorEngineActorConfig

    config = FactorEngineActorConfig(
        factor_config_path="config/factors.yaml",
        data_cls="nautilus_trader.adapters.binance.common.types:BinanceBar",
        bar_spec="1h",
    )
    actor = FactorEngineActor(config)
    # Add to TradingNode or BacktestEngine
    ```
    """

    def __init__(self, config: FactorEngineActorConfig) -> None:
        """
        Initialize the FactorEngineActor.

        Parameters
        ----------
        config : FactorEngineActorConfig
            The actor configuration.
        """
        super().__init__(config)

        self._config: FactorEngineActorConfig = config
        self._engine: FactorEngine | None = None

        # Flush synchronization:
        # Bars arrive instrument-by-instrument. We flush immediately when all
        # expected instruments have reported for a timestamp (set-complete),
        # with a configurable timeout fallback for partial arrivals.
        self._expected_instruments: set[str] = set()
        self._pending_instruments: set[str] = set()
        self._pending_ts: int = 0
        self._last_flushed_ts: int = 0
        self._extra_fields_detected: bool = False
        self._factor_snapshots: list[tuple[int, dict[str, dict[str, float]]]] = []
        self._cached_results: dict[int, dict[str, dict[str, float]]] | None = None
        self._flush_count: int = 0
        self._factors_mtime: float = 0.0

        # Previous factor results for realized IC computation (live only)
        self._prev_results: dict[str, dict[str, float]] | None = None

        # Funding rate: cache latest rate per instrument (forward-fill)
        self._latest_funding_rates: dict[str, float] = {}
        # Parquet lookups: {field_name: {inst_id: {ts_ns: {field: val}}}}
        # Supports multiple parquet extra_data sources (e.g. open_interest,
        # san_funding_rate, san_open_interest, san_volume_usd, ...).
        self._parquet_lookups: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
        # Bar field lookups: {field_name: {inst_id: {ts_ns: value}}}
        self._bar_field_lookups: dict[str, dict[str, dict[int, float]]] = {}
        # Broadcast configs for pre-flush injection (btc_close, eth_close, etc.)
        self._broadcast_configs: list[ExtraDataConfig] = []
        # Resolved broadcast instrument IDs (pattern → actual instrument_id)
        self._broadcast_resolved: dict[str, str] = {}

    def on_start(self) -> None:
        """
        Actions to perform on actor start.

        Initializes the FactorEngine and subscribes directly to bar types.
        Always uses direct subscription (no aggregation) to preserve
        BinanceBar extra fields like quote_volume and count.
        """
        self.log.info("Starting FactorEngineActor (Panel architecture)...")

        # Load factor config
        factor_config = None
        if self._config.factor_config_path:
            try:
                factor_config = load_factor_config(self._config.factor_config_path)
                self.log.info(
                    f"Loaded factor config: {factor_config.name} v{factor_config.version}"
                )
            except Exception as e:
                self.log.error(f"Failed to load factor config: {e}")
                return

        # Initialize FactorEngine
        self._engine = FactorEngine(
            config=factor_config,
            max_history=self._config.max_history,
        )
        self._pending_ts = 0
        self._last_flushed_ts = 0
        self._pending_instruments.clear()

        self.log.info(
            f"Registered {len(self._engine.factor_names)} factors: " f"{self._engine.factor_names}"
        )

        # Get bar types from injected config (required)
        if not self._config.bar_types:
            self.log.error(
                "bar_types not configured. Ensure backtest is run via CLI which injects "
                "bar_types from data config automatically (filtered by data_cls + bar_spec)."
            )
            return

        self.log.info(f"Using {len(self._config.bar_types)} bar types from config")

        # Subscribe directly to all injected bar types (no aggregation)
        self._subscribe_bar_types(self._config.bar_types)

        # Track expected instruments for set-complete flush detection
        self._expected_instruments = set(self._bar_type_to_inst_id.values())
        self.log.info(f"Tracking {len(self._expected_instruments)} instruments for flush sync")

        # ── Unified extra data setup ──
        extra_configs = self._load_extra_data_configs(factor_config)
        extra_buffer_fields: list[str] = []
        self._broadcast_configs = []

        for cfg in extra_configs:
            if cfg.source == "catalog":
                # FR: event-driven subscription (Actor-specific, not a loader)
                for inst_id_str in self._expected_instruments:
                    self.subscribe_funding_rates(InstrumentId.from_str(inst_id_str))
                extra_buffer_fields.append(cfg.name)
                self.log.info(
                    f"Extra data '{cfg.name}': subscribed to "
                    f"{len(self._expected_instruments)} instruments"
                )
            elif cfg.source == "broadcast":
                # Broadcast: flush 前 Buffer.inject_staged_field (Actor-specific)
                self._broadcast_configs.append(cfg)
                extra_buffer_fields.append(cfg.name)
                self.log.info(f"Extra data '{cfg.name}': broadcast from {cfg.instruments}")
            elif cfg.source in ("bar", "parquet"):
                # bar/parquet: preload lookup via extra_data.py unified entry
                extra_buffer_fields.append(cfg.name)
                try:
                    lookup = load_lookup(cfg, list(self._expected_instruments))
                    if lookup:
                        if cfg.source == "bar":
                            self._bar_field_lookups[cfg.name] = lookup
                        else:
                            # Store per-field-name to allow multiple parquet sources
                            self._parquet_lookups[cfg.name] = lookup
                        total = sum(len(v) for v in lookup.values())
                        self.log.info(
                            f"Extra data '{cfg.name}': preloaded "
                            f"{len(lookup)} instruments, {total} data points"
                        )
                    elif cfg.source == "bar" and not cfg.path:
                        self.log.info(f"Extra data '{cfg.name}': bar field auto-extract")
                    else:
                        self.log.warning(f"Extra data '{cfg.name}': no data found")
                except Exception as e:
                    self.log.warning(f"Extra data '{cfg.name}' load failed: {e}")

        # Register all extra fields so Buffer tracks them
        if extra_buffer_fields and self._engine is not None:
            existing = list(self._engine._buffer._extra_fields)
            new_fields = [f for f in extra_buffer_fields if f not in existing]
            if new_fields:
                all_extra = list(existing) + new_fields
                self._engine.set_extra_fields(all_extra)
                self.log.info(f"Registered extra data fields: {new_fields}")

        # Load pre-computed factor cache if configured
        # (after bar subscription so _expected_instruments is available for validation)
        if self._config.factor_cache_path:
            from nautilus_quants.factors.cache import (
                compute_config_hash,
                has_cache,
                load_as_snapshots,
                validate_cache,
            )

            cache_path = self._config.factor_cache_path
            if has_cache(cache_path):
                config_hash = compute_config_hash(factor_config)
                valid, warnings = validate_cache(
                    cache_path,
                    config_hash,
                    self._expected_instruments,
                )
                for w in warnings:
                    self.log.warning(w)
                if valid:
                    try:
                        self._cached_results = load_as_snapshots(cache_path)
                        self.log.info(
                            f"Loaded factor cache: "
                            f"{len(self._cached_results)} timestamps "
                            f"from {cache_path}"
                        )
                    except Exception as e:
                        self.log.warning(f"Failed to load factor cache: {e}")
                        self._cached_results = None
                else:
                    self.log.warning("Factor cache config mismatch, ignoring cache")
            else:
                self.log.info(
                    f"Factor cache path configured but empty, "
                    f"will save cache on stop: {cache_path}"
                )

        # Hot-reload: set up periodic config check via Nautilus Clock Timer.
        if self._config.enable_hot_reload:
            try:
                self._factors_mtime = Path(self._config.factor_config_path).stat().st_mtime
            except OSError:
                self.log.warning("Cannot stat factor config file, hot-reload disabled")
            else:
                self.clock.set_timer(
                    name=_HOT_RELOAD_TIMER,
                    interval=pd.Timedelta(
                        seconds=self._config.hot_reload_interval_secs,
                    ),
                    callback=self._on_hot_reload_check,
                )
                self.log.info(
                    f"Factor config hot-reload enabled, "
                    f"interval={self._config.hot_reload_interval_secs}s"
                )

        self.log.info("FactorEngineActor started successfully")

    def _on_hot_reload_check(self, event: TimeEvent) -> None:
        """Clock Timer callback: reload factor config if file changed."""
        if self._engine is None:
            return

        result = _try_reload_factors(
            self._config.factor_config_path,
            self._factors_mtime,
        )
        if result is None:
            return

        new_config, new_mtime = result

        # Sync parameters (Fix P1: was missing, expressions using
        # config parameters would keep stale values after reload).
        self._engine._parameters = new_config.parameters.copy()

        # Clean up factors/variables removed from the new config (Fix P1:
        # reload was additive-only, stale entries lingered indefinitely).
        new_factor_names = {f.name for f in new_config.all_factors}
        for name in set(self._engine.factor_names) - new_factor_names:
            del self._engine._factors[name]
            self._engine._factor_descriptions.pop(name, None)

        new_var_names = set(new_config.variables.keys())
        for name in set(self._engine.variable_names) - new_var_names:
            del self._engine._variables[name]
            if name in self._engine._variable_order:
                self._engine._variable_order.remove(name)

        # Register new/updated variables and factors.
        for var_name, var_expr in new_config.variables.items():
            self._engine.register_variable(var_name, var_expr)
        for factor in new_config.all_factors:
            self._engine.register_expression_factor(
                factor.name,
                factor.expression,
                factor.description,
            )
        self._factors_mtime = new_mtime
        self.log.info(
            f"Factor config hot-reloaded: {len(new_config.all_factors)} factors, "
            f"config={new_config.name} v{new_config.version}"
        )

    def on_stop(self) -> None:
        """Actions to perform on actor stop."""
        self.log.info("Stopping FactorEngineActor...")

        # Cancel hot-reload timer if active.
        if self._config.enable_hot_reload:
            try:
                self.clock.cancel_timer(_HOT_RELOAD_TIMER)
            except Exception:
                pass

        # Flush any pending data before shutdown
        if self._pending_ts > 0 and self._engine is not None:
            self._flush_and_publish(self._pending_ts)
            self._pending_ts = 0
            self._pending_instruments.clear()
        self._cancel_flush_alert()

        # Log performance stats
        if self._engine:
            stats = self._engine.get_performance_stats()
            self.log.info(
                f"Performance stats: "
                f"mean={stats['mean_ms']:.4f}ms, "
                f"max={stats['max_ms']:.4f}ms, "
                f"total_computes={stats['total_computes']}"
            )

        # Serialize factor data for report generation
        if self._cached_results is not None:
            # Cache replay: parquet already on disk, store path for reports
            from pathlib import Path as _Path

            parquet_path = _Path(self._config.factor_cache_path) / "factors.parquet"
            if parquet_path.is_file():
                self.cache.add(FACTOR_VALUES_CACHE_KEY, str(parquet_path).encode())
                self.log.info(f"Factor report will read from cache: {parquet_path}")
        elif self._factor_snapshots:
            # First run: serialize snapshots for reports + save cache
            self._serialize_factor_snapshots()

        self.log.info("FactorEngineActor stopped")

    def on_bar(self, bar: Bar) -> None:
        """
        Handle incoming bar data.

        Accumulates bar data and flushes immediately when all expected
        instruments have reported for a timestamp.  Falls back to a
        configurable timeout for partial arrivals, and force-flushes
        any residual data when the timestamp advances.

        Parameters
        ----------
        bar : Bar
            The received bar data.
        """
        instrument_id = self._resolve_bar(bar)
        if instrument_id is None or self._engine is None:
            return

        # Auto-detect extra bar fields on first BinanceBar
        if not self._extra_fields_detected:
            try:
                from nautilus_trader.adapters.binance.common.types import BinanceBar as _BinanceBar

                if isinstance(bar, _BinanceBar):
                    self._extra_fields_detected = True
                    extra = _detect_extra_bar_fields(bar)
                    if extra:
                        # Merge with already-registered fields (e.g. FR/OI
                        # from on_start) to avoid overwriting them.
                        existing = list(self._engine._buffer._extra_fields)
                        merged = list(dict.fromkeys(extra + existing))
                        self._engine.set_extra_fields(merged)
                        self.log.info(f"Auto-detected extra bar fields: {extra}")
            except ImportError:
                self._extra_fields_detected = True

        ts = bar.ts_event
        if self._last_flushed_ts > 0 and ts <= self._last_flushed_ts:
            self.log.debug(
                f"Ignoring late/duplicate bar ts={ts} (last_flushed={self._last_flushed_ts})"
            )
            return

        # Timestamp advanced → force-flush residual data from previous ts
        if self._pending_ts > 0 and ts > self._pending_ts:
            if self._pending_instruments:
                self.log.warning(
                    f"Timestamp advanced before flush: ts {self._pending_ts} -> {ts}, "
                    f"flushing {len(self._pending_instruments)}/"
                    f"{len(self._expected_instruments)} instruments"
                )
                self._cancel_flush_alert()
                self._flush_and_publish(self._pending_ts)
            else:
                self._cancel_flush_alert()
            self._pending_instruments.clear()
            self._pending_ts = 0

        # Accumulate bar into panel buffer
        bar_data = _extract_bar_data(bar)

        # Inject latest funding rate (forward-fill from on_funding_rate)
        if instrument_id in self._latest_funding_rates:
            bar_data["funding_rate"] = self._latest_funding_rates[instrument_id]

        # Inject all parquet-sourced fields from preloaded lookups
        # (open_interest, san_funding_rate, san_open_interest, ...)
        for field_name, inst_map in self._parquet_lookups.items():
            inst_data = inst_map.get(instrument_id)
            if inst_data is None:
                continue
            ts_data = inst_data.get(ts)
            if ts_data:
                bar_data.update(ts_data)

        # Inject bar fields from preloaded parquet lookups (e.g. quote_volume)
        for field_name, lookup in self._bar_field_lookups.items():
            if field_name not in bar_data and instrument_id in lookup:
                val = lookup[instrument_id].get(ts)
                if val is not None:
                    bar_data[field_name] = val

        self._engine.on_bar(instrument_id, bar_data, ts)
        self._pending_ts = ts
        self._pending_instruments.add(instrument_id)

        # All instruments reported → immediate flush
        if self._pending_instruments == self._expected_instruments:
            self._cancel_flush_alert()
            self._flush_and_publish(ts)
            self._pending_instruments.clear()
            self._pending_ts = 0
        elif self._config.flush_timeout_secs > 0 and len(self._pending_instruments) == 1:
            # First instrument arrived → start one-shot timeout alert
            self._set_flush_alert(ts)

    def _load_extra_data_configs(
        self,
        factor_config: FactorConfig | None,
    ) -> list[ExtraDataConfig]:
        """Load extra data configs from file or legacy fields."""
        if self._config.extra_data_path:
            try:
                configs = load_extra_data_config(self._config.extra_data_path)
                self.log.info(
                    f"Loaded {len(configs)} extra data configs from "
                    f"{self._config.extra_data_path}"
                )
                return configs
            except Exception as e:
                self.log.warning(f"Failed to load extra_data config: {e}")
                return []

        # Backward compat: convert legacy fields
        configs: list[ExtraDataConfig] = []
        if self._config.subscribe_funding_rates:
            configs.append(ExtraDataConfig(name="funding_rate", source="catalog"))
        if self._config.oi_data_path:
            configs.append(
                ExtraDataConfig(
                    name="open_interest",
                    source="parquet",
                    path=self._config.oi_data_path,
                    timeframe=self._config.bar_spec or "4h",
                )
            )
        return configs

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        """Cache latest funding rate for injection into bar_data.

        The cached value is injected on each subsequent ``on_bar()`` call,
        providing natural forward-fill: the 8h funding rate automatically
        persists across 1h/4h bars until the next settlement.
        """
        inst_id = str(funding_rate.instrument_id)
        self._latest_funding_rates[inst_id] = float(funding_rate.rate)

    def _flush_and_publish(self, ts: int) -> None:
        """Flush a timestamp, compute all factors, and publish results."""
        if self._engine is None:
            return

        # Inject broadcast fields BEFORE flush
        # (all bars received → BTC/ETH close is known and consistent)
        if self._broadcast_configs:
            self._inject_broadcast_staged(ts)

        # Use cached results if available, otherwise compute live
        if self._cached_results is not None and ts in self._cached_results:
            results = self._cached_results[ts]
            # Keep buffer consistent for potential cache-miss fallback
            self._engine.flush_timestamp(ts)
        else:
            results = self._engine.flush_and_compute(ts)
        self._last_flushed_ts = max(self._last_flushed_ts, ts)

        # Accumulate snapshot for factor export (skip when replaying from cache)
        if self._cached_results is None:
            self._factor_snapshots.append((ts, results))

        # Log factor values for warmup visibility (first 3 timestamps only)
        self._flush_count += 1
        if self._flush_count <= 3:
            for fname, fvals in results.items():
                if fvals:
                    sorted_vals = sorted(fvals.items(), key=lambda x: x[1])
                    vals_str = ", ".join(f"{inst}={v:.6f}" for inst, v in sorted_vals)
                    self.log.info(f"[{fname}] {vals_str}")

        # Create and publish FactorValues
        factor_values = FactorValues.create(
            ts_event=ts,
            factors=results,
        )

        if self._config.publish_signals:
            self.publish_data(data_type=DataType(FactorValues), data=factor_values)

            # 存入 Redis (JSON 格式)
            from nautilus_quants.utils.cache_keys import FACTOR_VALUES_CACHE_KEY

            self.cache.add(FACTOR_VALUES_CACHE_KEY, factor_values.to_json().encode())

        # Realized IC: live only (not during cache replay)
        if self._cached_results is None and self._prev_results is not None:
            self._publish_realized_ic(ts, results)
        self._prev_results = results

    def _publish_realized_ic(self, ts: int, results: dict[str, dict[str, float]]) -> None:
        """Compute realized IC and write to cache for Grafana monitoring."""
        if self._engine is None:
            return

        panel_fields = self._engine._buffer.to_panel()
        close = panel_fields.get("close")
        if close is None or not isinstance(close, pd.DataFrame) or len(close) < 2:
            return

        cols = close.columns.tolist()
        vals_cur = close.values[-1]
        vals_prev = close.values[-2]
        close_cur = {
            cols[i]: float(vals_cur[i]) for i in range(len(cols)) if np.isfinite(vals_cur[i])
        }
        close_prev = {
            cols[i]: float(vals_prev[i]) for i in range(len(cols)) if np.isfinite(vals_prev[i])
        }

        ic = compute_factor_ic(self._prev_results, close_cur, close_prev)
        if ic:
            payload = json.dumps({"ts": ts, "ic": ic})
            self.cache.add(FACTOR_IC_CACHE_KEY, payload.encode())

    def _inject_broadcast_staged(self, ts: int) -> None:
        """Inject broadcast fields into Buffer staging before flush.

        Called after all instruments have reported for ``ts``.
        Resolves broadcast instrument patterns on first call.
        """
        if self._engine is None:
            return

        # Lazy-resolve broadcast patterns → actual instrument IDs
        if not self._broadcast_resolved and self._broadcast_configs:
            for cfg in self._broadcast_configs:
                pattern = cfg.instruments[0] if cfg.instruments else ""
                if not pattern:
                    continue
                pattern_upper = pattern.upper()
                matched = next(
                    (i for i in self._expected_instruments if i.upper().startswith(pattern_upper)),
                    None,
                )
                if matched:
                    self._broadcast_resolved[cfg.name] = matched
                else:
                    self.log.warning(
                        f"Broadcast '{cfg.name}': instrument '{pattern}' " f"not found in universe"
                    )

        for cfg in self._broadcast_configs:
            matched = self._broadcast_resolved.get(cfg.name)
            if matched:
                self._engine._buffer.inject_staged_field(
                    ts,
                    cfg.name,
                    matched,
                )

    # -------------------------------------------------------------------------
    # Flush timeout (one-shot time alert)
    # -------------------------------------------------------------------------

    @property
    def _flush_alert_name(self) -> str:
        return f"factor_flush_timeout_{self.id}"

    def _set_flush_alert(self, ts: int) -> None:
        """Register a one-shot time alert to force-flush after timeout."""
        self._cancel_flush_alert()
        # Use clock.timestamp_ns() (not bar ts_event) so the timeout is
        # relative to wall-clock in live and simulated time in backtest.
        alert_time_ns = int(
            self.clock.timestamp_ns() + self._config.flush_timeout_secs * 1_000_000_000
        )
        self.clock.set_time_alert(
            name=self._flush_alert_name,
            alert_time=pd.Timestamp(alert_time_ns, unit="ns"),
            callback=self._on_flush_timeout,
        )

    def _cancel_flush_alert(self) -> None:
        """Cancel flush alert if active."""
        try:
            self.clock.cancel_timer(self._flush_alert_name)
        except Exception:
            pass  # Timer not active or already fired

    def _on_flush_timeout(self, event: object) -> None:
        """Time alert callback: flush pending data even if not all instruments reported."""
        if self._pending_ts > 0 and self._pending_instruments:
            missing = self._expected_instruments - self._pending_instruments
            if missing:
                self.log.info(
                    f"Flush timeout ({self._config.flush_timeout_secs}s) at ts={self._pending_ts}, "
                    f"proceeding with {len(self._pending_instruments)}/"
                    f"{len(self._expected_instruments)} instruments "
                    f"(missing: {missing})"
                )
            self._flush_and_publish(self._pending_ts)
            self._pending_instruments.clear()
            self._pending_ts = 0

    def _serialize_factor_snapshots(self) -> None:
        """Serialize accumulated snapshots to cache (first-run path only)."""
        import pickle

        try:
            self.cache.add(FACTOR_VALUES_CACHE_KEY, pickle.dumps(self._factor_snapshots))
            self.log.info(
                f"Cached {len(self._factor_snapshots)} factor snapshots "
                f"({FACTOR_VALUES_CACHE_KEY})"
            )
        except Exception as e:
            self.log.warning(f"Failed to cache factor snapshots: {e}")

        if self._config.factor_cache_path:
            from nautilus_quants.factors.cache import compute_config_hash, save_snapshots_as_cache

            try:
                factor_cfg = load_factor_config(
                    self._config.factor_config_path,
                )
                config_hash = compute_config_hash(factor_cfg)
                save_snapshots_as_cache(
                    self._factor_snapshots,
                    self._config.factor_cache_path,
                    factor_config_path=self._config.factor_config_path,
                    config_hash=config_hash,
                )
                self.log.info(
                    f"Factor cache saved: {self._config.factor_cache_path} "
                    f"({len(self._factor_snapshots)} timestamps)"
                )
            except Exception as e:
                self.log.warning(f"Failed to save factor cache: {e}")

    def on_reset(self) -> None:
        """Reset the actor state."""
        if self._engine:
            self._engine.reset()
        self._pending_ts = 0
        self._last_flushed_ts = 0
        self._prev_results = None
        self._pending_instruments.clear()
        self._expected_instruments.clear()
        self._cancel_flush_alert()
        self._factor_snapshots.clear()
        self._latest_funding_rates.clear()
        self._parquet_lookups.clear()
        self._bar_field_lookups.clear()
        self.log.info("FactorEngineActor reset")

    # -------------------------------------------------------------------------
    # Public API for programmatic factor registration
    # -------------------------------------------------------------------------

    @property
    def engine(self) -> FactorEngine | None:
        """Get the underlying FactorEngine instance."""
        return self._engine

    @property
    def factor_names(self) -> list[str]:
        """Get list of registered factor names."""
        if self._engine:
            return self._engine.factor_names
        return []

    def register_expression_factor(
        self,
        name: str,
        expression: str,
        description: str = "",
    ) -> None:
        """
        Register an expression-based factor.

        Parameters
        ----------
        name : str
            Factor name.
        expression : str
            Alpha101-style expression.
        description : str, optional
            Factor description.
        """
        if self._engine is None:
            self.log.warning("Cannot register factor: engine not initialized")
            return

        self._engine.register_expression_factor(
            name=name,
            expression=expression,
            description=description,
        )
        self.log.info(f"Registered factor: {name}")

    def register_variable(self, name: str, expression: str) -> None:
        """
        Register a reusable variable.

        Parameters
        ----------
        name : str
            Variable name.
        expression : str
            Expression defining the variable.
        """
        if self._engine is None:
            self.log.warning("Cannot register variable: engine not initialized")
            return

        self._engine.register_variable(name, expression)
        self.log.debug(f"Registered variable: {name}")

    def add_bar_type(self, bar_type: BarType) -> None:
        """
        Add a bar type to subscribe to at runtime.

        Must be called after ``on_start()`` has been invoked (i.e., after
        ``_subscribe_bar_types`` has initialised ``_bar_type_to_inst_id``).

        Parameters
        ----------
        bar_type : BarType
            The bar type to subscribe to.
        """
        if bar_type not in self._bar_type_to_inst_id:
            self._bar_type_to_inst_id[bar_type] = str(bar_type.instrument_id)
            if self.is_running():
                self.subscribe_bars(bar_type)
