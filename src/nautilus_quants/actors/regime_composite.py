# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
RegimeCompositeActor - Regime-aware dynamic factor composite.

Sits between FactorEngineActor and DecisionEngineActor in the pipeline.
Subscribes to FactorValues (individual factors) and BTC bars, detects
market regime, reweights factors per regime, and publishes a new
FactorValues with the regime-aware composite.

Data flow:
    FactorEngineActor → FactorValues (8 individual + static composite)
        ↓
    RegimeCompositeActor → FactorValues (regime-aware composite) ← BTC bars
        ↓
    DecisionEngineActor → RebalanceOrders

Based on: Yu, Mulvey, Nie (JPM 2026)

Constitution Compliance:
    - Extends Nautilus Actor base class (Principle I)
    - Configuration-driven via ActorConfig (Principle II)
    - Pure reweighting, no order submission (Principle V)
"""

from __future__ import annotations

import json

import numpy as np
from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType, DataType

from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.factors.factor_values import FactorValues


class RegimeCompositeActorConfig(ActorConfig, frozen=True):
    """Configuration for RegimeCompositeActor.

    Parameters
    ----------
    regime_instrument : str
        Instrument ID for regime detection (e.g., "BTCUSDT.BINANCE").
    ema_span : int
        EMA span for regime proxy (BTC return EMA).
    regime_threshold : float
        Threshold for bull/bear classification (0.0 = EMA sign).
    composite_name : str
        Name of the output composite factor.
    factor_names : list[str]
        Names of individual factors to consume from FactorValues.
    weight_map : dict[str, dict[str, float]]
        Per-regime weights: {"bull": {factor: weight}, "bear": {factor: weight}}.
    bar_types : list[str]
        Bar types to subscribe for regime instrument (injected by CLI).
    """

    regime_instrument: str = "BTCUSDT.BINANCE"
    ema_span: int = 20
    regime_threshold: float = 0.0
    composite_name: str = "composite"
    factor_names: list[str] = []
    weight_map: dict[str, dict[str, float]] = {}
    bar_types: list[str] = []


class RegimeCompositeActor(BarSubscriptionMixin, Actor):
    """Regime-aware dynamic factor composite actor.

    Detects market regime from BTC price data using EMA of returns,
    then reweights factor values according to the per-regime weight map.
    Publishes a new FactorValues containing only the regime-aware composite.
    """

    def __init__(self, config: RegimeCompositeActorConfig) -> None:
        super().__init__(config)
        self._regime_instrument = config.regime_instrument
        self._ema_span = config.ema_span
        self._threshold = config.regime_threshold
        self._composite_name = config.composite_name
        self._factor_names = list(config.factor_names)
        self._weight_map = dict(config.weight_map)

        # EMA state
        self._ema_value: float = 0.0
        self._ema_initialized: bool = False
        self._ema_alpha: float = 2.0 / (config.ema_span + 1)
        self._prev_close: float = 0.0
        self._current_regime: str = "neutral"

        # Pending factor values (wait for bar + factors to align)
        self._pending_factors: dict[str, dict[str, float]] | None = None
        self._bar_count: int = 0

    def on_start(self) -> None:
        self._subscribe_bar_types(self.config.bar_types)
        self.subscribe_data(DataType(FactorValues))
        self.log.info(
            f"RegimeCompositeActor started: "
            f"regime={self._regime_instrument}, "
            f"ema_span={self._ema_span}, "
            f"factors={len(self._factor_names)}, "
            f"regimes={list(self._weight_map.keys())}",
        )

    def on_bar(self, bar: Bar) -> None:
        instrument_id = str(bar.bar_type.instrument_id)
        if instrument_id != self._regime_instrument:
            return

        close = float(bar.close)
        if self._prev_close <= 0:
            self._prev_close = close
            return

        # Update EMA of returns
        ret = (close - self._prev_close) / self._prev_close
        self._prev_close = close

        if not self._ema_initialized:
            self._ema_value = ret
            self._ema_initialized = True
        else:
            self._ema_value = (
                self._ema_alpha * ret + (1 - self._ema_alpha) * self._ema_value
            )

        # Classify regime
        if self._ema_value > self._threshold:
            self._current_regime = "bull"
        elif "neutral" in self._weight_map and abs(self._ema_value) < self._threshold:
            self._current_regime = "neutral"
        else:
            self._current_regime = "bear"

        self._bar_count += 1
        self._try_publish()

    def on_data(self, data: FactorValues) -> None:
        factors = data.to_dict()
        self._pending_factors = factors
        self._try_publish()

    def _try_publish(self) -> None:
        """Publish regime-aware composite if both bar and factors are available."""
        if self._pending_factors is None or not self._ema_initialized:
            return

        factors = self._pending_factors
        self._pending_factors = None

        # Select weights for current regime
        weights = self._weight_map.get(self._current_regime)
        if weights is None:
            # Fallback to equal weight
            weights = {n: 1.0 / len(self._factor_names) for n in self._factor_names}

        # Cross-sectional normalize each factor, then weighted sum
        composite: dict[str, float] = {}
        factor_arrays: dict[str, dict[str, float]] = {}

        for name in self._factor_names:
            if name not in factors:
                continue
            factor_arrays[name] = factors[name]

        if not factor_arrays:
            return

        # Get all instruments
        all_instruments: set[str] = set()
        for vals in factor_arrays.values():
            all_instruments.update(vals.keys())

        # Cross-sectional normalize per factor
        normalized: dict[str, dict[str, float]] = {}
        for name, vals in factor_arrays.items():
            values = np.array([vals.get(inst, np.nan) for inst in all_instruments])
            valid = ~np.isnan(values)
            if valid.sum() < 2:
                continue
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std < 1e-12:
                continue
            norm_values = (values - mean) / std
            normalized[name] = {
                inst: float(nv)
                for inst, nv, v in zip(all_instruments, norm_values, valid)
                if v
            }

        # Weighted composite
        for inst in all_instruments:
            total = 0.0
            total_weight = 0.0
            for name in self._factor_names:
                if name in normalized and inst in normalized[name]:
                    w = weights.get(name, 0.0)
                    total += w * normalized[name][inst]
                    total_weight += w
            if total_weight > 0:
                composite[inst] = total / total_weight

        if not composite:
            return

        # Build new FactorValues with only the composite
        # Include original factors + regime-aware composite
        output_factors = dict(factors)
        output_factors[self._composite_name] = composite

        fv = FactorValues.create(
            ts_event=self.clock.timestamp_ns(),
            factors=output_factors,
        )
        self.publish_data(data_type=DataType(FactorValues), data=fv)

        if self._bar_count % 100 == 0:
            self.log.info(
                f"Regime={self._current_regime}, "
                f"EMA={self._ema_value:.6f}, "
                f"composite_instruments={len(composite)}, "
                f"bar_count={self._bar_count}",
            )

    def on_stop(self) -> None:
        self.log.info(
            f"RegimeCompositeActor stopped. "
            f"Total bars={self._bar_count}, "
            f"final_regime={self._current_regime}",
        )
