# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Core data types for the Factor Framework.

This module defines the fundamental data containers used throughout
the factor computation pipeline.

Constitution Compliance:
    - Uses Nautilus-native types (InstrumentId, Bar) where available (Principle I)
    - FactorValues extends Nautilus Data for MessageBus publishing (Principle I)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# Re-export FactorValues from its own module (avoids __future__ annotations conflict)
from nautilus_quants.factors.factor_values import FactorValues

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.identifiers import InstrumentId

# Make FactorValues available via this module
__all__ = ["FactorInput", "FactorValues"]


@dataclass(frozen=True)
class FactorInput:
    """
    Input data container for factor computation.

    Holds the current bar data and historical values needed for
    time-series and cross-sectional factor calculations.

    Attributes
    ----------
    instrument_id : InstrumentId
        The instrument identifier.
    timestamp_ns : int
        Event timestamp in nanoseconds.
    open : float
        Current bar open price.
    high : float
        Current bar high price.
    low : float
        Current bar low price.
    close : float
        Current bar close price.
    volume : float
        Current bar volume.
    history : dict[str, np.ndarray]
        Historical data arrays keyed by field name.
    """

    instrument_id: InstrumentId
    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    history: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_bar(
        cls,
        bar: Bar,
        history: dict[str, np.ndarray] | None = None,
    ) -> FactorInput:
        """
        Create FactorInput from a Nautilus Bar.

        Parameters
        ----------
        bar : Bar
            The Nautilus Bar object.
        history : dict[str, np.ndarray], optional
            Historical data arrays.

        Returns
        -------
        FactorInput
            New FactorInput instance.
        """
        return cls(
            instrument_id=bar.bar_type.instrument_id,
            timestamp_ns=bar.ts_event,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            history=history or {},
        )
