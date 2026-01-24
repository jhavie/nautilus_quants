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

import json
from typing import TYPE_CHECKING

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.factor_engine import FactorEngine

if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId


class FactorEngineConfig(ActorConfig, frozen=True):
    """
    Configuration for FactorEngineActor.

    Parameters
    ----------
    factor_config_path : str, optional
        Path to YAML factor configuration file.
    bar_types : list[str], optional
        List of bar types to subscribe to (e.g., ["ETHUSDT.BINANCE-1-HOUR-LAST"]).
    max_history : int, default 500
        Maximum history to maintain per instrument.
    publish_signals : bool, default True
        Whether to publish factor values as signals.
    signal_prefix : str, default "factor"
        Prefix for signal names (e.g., "factor.breakout").
    """

    factor_config_path: str | None = None
    bar_types: list[str] | None = None
    max_history: int = 500
    publish_signals: bool = True
    signal_prefix: str = "factor"


class FactorEngineActor(Actor):
    """
    Nautilus Actor that computes factors and publishes results.

    This actor wraps the FactorEngine and integrates it with the Nautilus
    trading system, providing:
    - Automatic bar subscription
    - Factor computation on each bar
    - Signal publishing for factor values
    - Performance logging via Nautilus Logger

    Example
    -------
    ```python
    from nautilus_quants.factors.engine import FactorEngineActor, FactorEngineConfig

    config = FactorEngineConfig(
        factor_config_path="config/factors.yaml",
        bar_types=["ETHUSDT.BINANCE-1-HOUR-LAST"],
    )
    actor = FactorEngineActor(config)
    # Add to TradingNode or BacktestEngine
    ```
    """

    def __init__(self, config: FactorEngineConfig) -> None:
        """
        Initialize the FactorEngineActor.

        Parameters
        ----------
        config : FactorEngineConfig
            The actor configuration.
        """
        super().__init__(config)

        self._config: FactorEngineConfig = config
        self._engine: FactorEngine | None = None
        self._bar_types: list[BarType] = []

        # Parse bar types from config
        if config.bar_types:
            self._bar_types = [BarType.from_str(bt) for bt in config.bar_types]

    def on_start(self) -> None:
        """
        Actions to perform on actor start.

        Initializes the FactorEngine and subscribes to configured bar types.
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

        self.log.info(f"Registered {len(self._engine.factor_names)} factors: {self._engine.factor_names}")

        # Subscribe to bar types
        for bar_type in self._bar_types:
            self.subscribe_bars(bar_type)
            self.log.info(f"Subscribed to {bar_type}")

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

        # Unsubscribe from bar types
        for bar_type in self._bar_types:
            self.unsubscribe_bars(bar_type)

        self.log.info("FactorEngineActor stopped")

    def on_bar(self, bar: Bar) -> None:
        """
        Handle incoming bar data.

        Computes all registered factors and optionally publishes signals.

        Parameters
        ----------
        bar : Bar
            The received bar data.
        """
        if self._engine is None:
            return

        # Compute factors
        result = self._engine.on_bar(bar)

        if result is None:
            return

        # Publish signals for each factor
        if self._config.publish_signals:
            instrument_id = str(bar.bar_type.instrument_id)

            for factor_name, factor_values in result.factors.items():
                value = factor_values.get(instrument_id)
                if value is not None and not (isinstance(value, float) and value != value):  # Check for NaN
                    signal_name = f"{self._config.signal_prefix}.{factor_name}"
                    # Publish as signal with JSON value containing instrument and value
                    self.publish_signal(
                        name=signal_name,
                        value=json.dumps({"instrument": instrument_id, "value": value}),
                        ts_event=bar.ts_event,
                    )

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
