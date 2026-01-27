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

from typing import TYPE_CHECKING

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType, DataType

from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.types import FactorValues

if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId


class FactorEngineActorConfig(ActorConfig, frozen=True):
    """
    Configuration for FactorEngineActor.

    Parameters
    ----------
    factor_config_path : str
        Path to YAML factor configuration file (required).
    interval : str, default "1h"
        Target timeframe for factor computation (e.g., "1m", "1h", "4h", "1d").
        If not "1m", bars will be aggregated from data source to this interval.
    max_history : int, default 500
        Maximum history to maintain per instrument.
    publish_signals : bool, default True
        Whether to publish factor values as CustomData.
    signal_prefix : str, default "factor"
        Prefix for signal names (e.g., "factor.breakout").
    bar_types : list[str], default []
        List of bar type strings to subscribe to (injected by CLI from data config).
        If empty, will try to get from cache (legacy behavior).
    """

    factor_config_path: str
    interval: str = "1h"
    max_history: int = 500
    publish_signals: bool = True
    signal_prefix: str = "factor"
    bar_types: list[str] = []


class FactorEngineActor(Actor):
    """
    Nautilus Actor that computes factors and publishes results.

    This actor wraps the FactorEngine and integrates it with the Nautilus
    trading system, providing:
    - Automatic bar subscription with optional aggregation
    - Factor computation on each bar
    - CustomData publishing for factor values
    - Performance logging via Nautilus Logger

    Example
    -------
    ```python
    from nautilus_quants.factors.engine import FactorEngineActor, FactorEngineActorConfig

    config = FactorEngineActorConfig(
        factor_config_path="config/factors.yaml",
        interval="1h",  # Aggregate to 1-hour bars
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
        self._bar_types: list[BarType] = []  # Target bar types to process

    def on_start(self) -> None:
        """
        Actions to perform on actor start.

        Initializes the FactorEngine and subscribes to bar types.
        Uses interval config to determine bar aggregation.
        """
        self.log.info(f"Starting FactorEngineActor (interval={self._config.interval})...")

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

        self.log.info(f"Registered {len(self._engine.factor_names)} factors: {self._engine.factor_names}")

        # Get bar types from injected config (required)
        # bar_types must be injected by CLI from data config
        if not self._config.bar_types:
            self.log.error(
                "bar_types not configured. Ensure backtest is run via CLI which injects "
                "bar_types from data config automatically."
            )
            return
        
        source_bar_type_strs = list(self._config.bar_types)
        self.log.info(f"Using {len(source_bar_type_strs)} bar types from config")

        # Determine if aggregation is needed based on interval
        interval = self._config.interval.lower()
        
        if interval == "1m":
            # No aggregation needed - use source bars directly
            for bar_type_str in source_bar_type_strs:
                bar_type = BarType.from_str(bar_type_str)
                self._bar_types.append(bar_type)
                self.subscribe_bars(bar_type)
                self.log.info(f"Subscribed to {bar_type}")
        else:
            # Aggregation needed - create INTERNAL bar types
            from nautilus_quants.strategies.utils import create_aggregated_bar_type
            from nautilus_trader.model.enums import BarAggregation

            for bar_type_str in source_bar_type_strs:
                source_bar_type = BarType.from_str(bar_type_str)
                instrument_id = str(source_bar_type.instrument_id)
                # Build source spec from bar type string
                # spec.aggregation may be int or enum, handle both
                aggregation = source_bar_type.spec.aggregation
                if isinstance(aggregation, int):
                    aggregation_name = BarAggregation(aggregation).name
                else:
                    aggregation_name = aggregation.name
                source_spec = f"{source_bar_type.spec.step}-{aggregation_name}-EXTERNAL"

                target_str, subscribe_str = create_aggregated_bar_type(
                    instrument_id,
                    self._config.interval,
                    source_spec,
                )

                target_bar_type = BarType.from_str(target_str)
                subscribe_bar_type = BarType.from_str(subscribe_str)

                self._bar_types.append(target_bar_type)
                self.subscribe_bars(subscribe_bar_type)
                self.log.info(f"Subscribed to aggregated bars: {subscribe_str}")

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

        Only processes target bar types (aggregated INTERNAL bars if using
        interval aggregation). Computes all registered factors and publishes
        results as CustomData.

        Parameters
        ----------
        bar : Bar
            The received bar data.
        """
        # Only process target bar types (skip source bars when aggregating)
        if bar.bar_type not in self._bar_types:
            return

        if self._engine is None:
            return

        # Compute factors
        result = self._engine.on_bar(bar)

        if result is None:
            return

        # Publish FactorValues as Data (now a proper Nautilus Data subclass)
        if self._config.publish_signals:
            self.publish_data(data_type=DataType(FactorValues), data=result)

    def on_reset(self) -> None:
        """Reset the actor state."""
        if self._engine:
            self._engine.reset()
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
