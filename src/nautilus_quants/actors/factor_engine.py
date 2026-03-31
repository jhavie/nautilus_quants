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

import pandas as pd
from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType, DataType

from nautilus_quants.utils.cache_keys import FACTOR_VALUES_CACHE_KEY
from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.types import FactorValues


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
            f"Registered {len(self._engine.factor_names)} factors: "
            f"{self._engine.factor_names}"
        )

        # Load pre-computed factor cache if configured
        if self._config.factor_cache_path:
            from nautilus_quants.factors.cache import has_cache, load_as_snapshots

            cache_path = self._config.factor_cache_path
            if has_cache(cache_path):
                try:
                    self._cached_results = load_as_snapshots(cache_path)
                    self.log.info(
                        f"Loaded factor cache: {len(self._cached_results)} timestamps "
                        f"from {cache_path}"
                    )
                except Exception as e:
                    self.log.warning(f"Failed to load factor cache: {e}")
                    self._cached_results = None
            else:
                self.log.info(
                    f"Factor cache path configured but empty, "
                    f"will save cache on stop: {cache_path}"
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
        self.log.info(
            f"Tracking {len(self._expected_instruments)} instruments for flush sync"
        )

        self.log.info("FactorEngineActor started successfully")

    def on_stop(self) -> None:
        """Actions to perform on actor stop."""
        self.log.info("Stopping FactorEngineActor...")

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

        # Serialize factor snapshots to cache for report generation
        if self._factor_snapshots:
            import pickle

            try:
                self.cache.add(
                    FACTOR_VALUES_CACHE_KEY, pickle.dumps(self._factor_snapshots)
                )
                self.log.info(
                    f"Cached {len(self._factor_snapshots)} factor snapshots "
                    f"({FACTOR_VALUES_CACHE_KEY})"
                )
            except Exception as e:
                self.log.warning(f"Failed to cache factor snapshots: {e}")

        # Save factor cache if path configured and no cache was loaded (first run)
        if (
            self._config.factor_cache_path
            and self._cached_results is None
            and self._factor_snapshots
        ):
            from nautilus_quants.factors.cache import save_snapshots_as_cache

            try:
                save_snapshots_as_cache(
                    self._factor_snapshots,
                    self._config.factor_cache_path,
                    bar_spec=self._config.bar_spec,
                    factor_config_path=self._config.factor_config_path,
                )
                self.log.info(
                    f"Factor cache saved: {self._config.factor_cache_path} "
                    f"({len(self._factor_snapshots)} timestamps)"
                )
            except Exception as e:
                self.log.warning(f"Failed to save factor cache: {e}")

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
                from nautilus_trader.adapters.binance.common.types import (
                    BinanceBar as _BinanceBar,
                )

                if isinstance(bar, _BinanceBar):
                    self._extra_fields_detected = True
                    extra = _detect_extra_bar_fields(bar)
                    if extra:
                        self._engine.set_extra_fields(extra)
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
        self._engine.on_bar(instrument_id, bar_data, ts)
        self._pending_ts = ts
        self._pending_instruments.add(instrument_id)

        # All instruments reported → immediate flush
        if self._pending_instruments == self._expected_instruments:
            self._cancel_flush_alert()
            self._flush_and_publish(ts)
            self._pending_instruments.clear()
            self._pending_ts = 0
        elif (
            self._config.flush_timeout_secs > 0 and len(self._pending_instruments) == 1
        ):
            # First instrument arrived → start one-shot timeout alert
            self._set_flush_alert(ts)

    def _flush_and_publish(self, ts: int) -> None:
        """Flush a timestamp, compute all factors, and publish results."""
        if self._engine is None:
            return

        # Use cached results if available, otherwise compute live
        if self._cached_results is not None and ts in self._cached_results:
            results = self._cached_results[ts]
        else:
            results = self._engine.flush_and_compute(ts)
        self._last_flushed_ts = max(self._last_flushed_ts, ts)

        # Accumulate snapshot for factor export
        self._factor_snapshots.append((ts, results))

        # Diagnostic: log per-factor non-empty instrument count (first 5 + every 50th)
        compute_count = self._engine.get_performance_stats().get("total_computes", 0)
        if compute_count <= 5 or compute_count % 50 == 0:
            diag_parts = []
            for fname, fvals in results.items():
                diag_parts.append(f"{fname}={len(fvals)}")
            self.log.info(f"Factor compute #{compute_count} [{', '.join(diag_parts)}]")

        # Log actual factor values sorted by value (ascending) for ranking visibility
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

    def on_reset(self) -> None:
        """Reset the actor state."""
        if self._engine:
            self._engine.reset()
        self._pending_ts = 0
        self._last_flushed_ts = 0
        self._pending_instruments.clear()
        self._expected_instruments.clear()
        self._cancel_flush_alert()
        self._factor_snapshots.clear()
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
