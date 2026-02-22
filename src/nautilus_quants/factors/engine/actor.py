# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FactorEngineActor - Nautilus Actor wrapper for FactorEngine.

This module provides a Nautilus-native Actor that wraps the FactorEngine,
enabling seamless integration with the Nautilus trading system.

Constitution Compliance:
    - Extends Nautilus Actor base class (Principle I)
    - Uses Nautilus Logger, Clock, MessageBus (Principle I)
    - Configuration-driven via ActorConfig (Principle II)
"""

from __future__ import annotations

from collections import defaultdict

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType, DataType

from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.cs_factor_engine import CsFactorEngine
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
    """

    factor_config_path: str
    data_cls: str = ""
    bar_spec: str = ""
    max_history: int = 500
    publish_signals: bool = True
    signal_prefix: str = "factor"
    bar_types: list[str] = []


class FactorEngineActor(Actor):
    """
    Nautilus Actor that computes factors and publishes results.

    This actor wraps the FactorEngine and integrates it with the Nautilus
    trading system, providing:
    - Direct bar subscription (no aggregation - BinanceBar fields preserved)
    - Two-phase factor computation (time-series then cross-sectional)
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
    from nautilus_quants.factors.engine import FactorEngineActor, FactorEngineActorConfig

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
        self._cs_engine: CsFactorEngine | None = None
        self._bar_types: list[BarType] = []  # Target bar types to process

        # Synchronization state for cross-sectional factors
        self._ts_factor_values: dict[int, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._last_processed_ts: int = 0
        self._extra_fields_detected: bool = False

    def on_start(self) -> None:
        """
        Actions to perform on actor start.

        Initializes the FactorEngine and subscribes directly to bar types.
        Always uses direct subscription (no aggregation) to preserve
        BinanceBar extra fields like quote_volume and count.
        """
        self.log.info("Starting FactorEngineActor...")

        # Initialize FactorEngine
        factor_config = None
        if self._config.factor_config_path:
            try:
                factor_config = load_factor_config(self._config.factor_config_path)
                self.log.info(f"Loaded factor config: {factor_config.name} v{factor_config.version}")
            except Exception as e:
                self.log.error(f"Failed to load factor config: {e}")
                return

        self._engine = FactorEngine(
            config=factor_config,
            max_history=self._config.max_history,
        )

        # Initialize cross-sectional factor engine
        self._cs_engine = CsFactorEngine(config=factor_config)

        ts_factors = self._cs_engine.ts_factor_names
        cs_factors = self._cs_engine.cs_factor_names
        self.log.info(f"Time-series factors: {ts_factors}")
        self.log.info(f"Cross-sectional factors: {cs_factors}")

        self.log.info(f"Registered {len(self._engine.factor_names)} factors: {self._engine.factor_names}")

        # Get bar types from injected config (required)
        # bar_types is injected by CLI from data config, filtered by data_cls + bar_spec
        if not self._config.bar_types:
            self.log.error(
                "bar_types not configured. Ensure backtest is run via CLI which injects "
                "bar_types from data config automatically (filtered by data_cls + bar_spec)."
            )
            return

        self.log.info(f"Using {len(self._config.bar_types)} bar types from config")

        # Subscribe directly to all injected bar types (no aggregation)
        # This preserves BinanceBar extra fields (quote_volume, count, etc.)
        for bar_type_str in self._config.bar_types:
            bar_type = BarType.from_str(bar_type_str)
            self._bar_types.append(bar_type)
            self.subscribe_bars(bar_type)
            self.log.info(f"Subscribed to {bar_type} (direct)")

        self.log.info("FactorEngineActor started successfully")

    def on_stop(self) -> None:
        """Actions to perform on actor stop."""
        self.log.info("Stopping FactorEngineActor...")

        # Log performance stats
        if self._engine:
            stats = self._engine.get_performance_stats()
            self.log.info(
                f"Performance stats: "
                f"mean={stats['mean_ms']:.4f}ms, "
                f"max={stats['max_ms']:.4f}ms, "
                f"total_computes={stats['total_computes']}"
            )

        self.log.info("FactorEngineActor stopped")

    def on_bar(self, bar: Bar) -> None:
        """
        Handle incoming bar data.

        Implements two-phase factor computation:
        1. Compute time-series factors for this bar
        2. When timestamp changes, process previous batch

        Parameters
        ----------
        bar : Bar
            The received bar data.
        """
        # Only process target bar types
        if bar.bar_type not in self._bar_types:
            return

        if self._engine is None or self._cs_engine is None:
            return

        # Auto-detect extra bar fields on first BinanceBar
        # Only trigger detection for BinanceBar instances to avoid
        # standard Bar falsely marking detection as complete.
        if not self._extra_fields_detected:
            try:
                from nautilus_trader.adapters.binance.common.types import BinanceBar as _BinanceBar
                if isinstance(bar, _BinanceBar):
                    self._extra_fields_detected = True
                    extra = _detect_extra_bar_fields(bar)
                    if extra:
                        self._engine.set_extra_fields(extra)
                        self.log.info(f"Auto-detected extra bar fields: {extra}")
            except ImportError:
                self._extra_fields_detected = True

        instrument_id = str(bar.bar_type.instrument_id)
        ts = bar.ts_event

        # When timestamp changes, process the previous batch
        if self._last_processed_ts > 0 and ts > self._last_processed_ts:
            self._process_complete_batch(self._last_processed_ts)
            # Cleanup old timestamps
            old_ts = [t for t in self._ts_factor_values if t < ts]
            for t in old_ts:
                del self._ts_factor_values[t]

        # Compute time-series factors for this bar
        result = self._engine.on_bar(bar)

        if result is not None:
            # Store time-series factor values
            for factor_name, factor_values in result.factors.items():
                for inst_id, value in factor_values.items():
                    self._ts_factor_values[ts][factor_name][inst_id] = value

        self._last_processed_ts = ts

    def _process_complete_batch(self, ts: int) -> None:
        """
        Process a complete batch when all instruments have data.

        Computes cross-sectional factors and publishes combined results.
        """
        if self._cs_engine is None:
            return

        ts_values = dict(self._ts_factor_values[ts])

        # Compute cross-sectional factors
        cs_values = self._cs_engine.compute(ts_values)

        # Combine all factor values
        all_values = dict(ts_values)
        all_values.update(cs_values)

        # Create and publish FactorValues
        factor_values = FactorValues.create(
            ts_event=ts,
            factors=all_values,
        )

        if self._config.publish_signals:
            self.publish_data(data_type=DataType(FactorValues), data=factor_values)

    def on_reset(self) -> None:
        """Reset the actor state."""
        if self._engine:
            self._engine.reset()
        self._ts_factor_values.clear()
        self._last_processed_ts = 0
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
        warmup_period: int = 0,
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
        warmup_period : int, optional
            Warmup period before valid output.
        """
        if self._engine is None:
            self.log.warning("Cannot register factor: engine not initialized")
            return

        self._engine.register_expression_factor(
            name=name,
            expression=expression,
            description=description,
            warmup_period=warmup_period,
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
        Add a bar type to subscribe to.

        This should be called before on_start() or will subscribe immediately
        if the actor is already running.

        Parameters
        ----------
        bar_type : BarType
            The bar type to subscribe to.
        """
        if bar_type not in self._bar_types:
            self._bar_types.append(bar_type)

            if self.is_running():
                self.subscribe_bars(bar_type)
                self.log.info(f"Subscribed to {bar_type}")
