# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Core data types for the Factor Framework.

This module defines the fundamental data containers used throughout
the factor computation pipeline.

Constitution Compliance:
    - Uses Nautilus-native types (InstrumentId, Bar) where available (Principle I)
    - FactorValues can be converted to CustomData for MessageBus (Principle I)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.identifiers import InstrumentId


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


@dataclass(frozen=True)
class FactorValues:
    """
    Factor computation results - transmitted via MessageBus.

    This is the output of FactorEngine, containing computed factor values
    for all instruments at a given timestamp. Can be converted to Nautilus
    CustomData for MessageBus publishing.

    Attributes
    ----------
    ts_event : int
        Event timestamp in nanoseconds.
    factors : dict[str, dict[str, float]]
        Nested dict of {factor_name: {instrument_id_str: value}}.

    Example
    -------
    ```python
    values = FactorValues(
        ts_event=1234567890000000000,
        factors={
            "alpha001": {"ETHUSDT.BINANCE": 0.5, "BTCUSDT.BINANCE": -0.3},
            "alpha002": {"ETHUSDT.BINANCE": 0.8, "BTCUSDT.BINANCE": 0.2},
        }
    )

    # Convert to CustomData for MessageBus
    custom_data = values.to_custom_data()
    ```
    """

    ts_event: int
    factors: dict[str, dict[str, float]] = field(default_factory=dict)

    def get(self, factor_name: str, instrument_id: str) -> float | None:
        """
        Get a specific factor value.

        Parameters
        ----------
        factor_name : str
            Name of the factor.
        instrument_id : str
            String representation of instrument ID.

        Returns
        -------
        float | None
            Factor value or None if not found.
        """
        factor_values = self.factors.get(factor_name)
        if factor_values is None:
            return None
        return factor_values.get(instrument_id)

    def get_factor(self, factor_name: str) -> dict[str, float]:
        """
        Get all values for a specific factor.

        Parameters
        ----------
        factor_name : str
            Name of the factor.

        Returns
        -------
        dict[str, float]
            Dict of {instrument_id: value} or empty dict.
        """
        return self.factors.get(factor_name, {})

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            "ts_event": self.ts_event,
            "factors": self.factors,
        }

    def to_json(self) -> str:
        """
        Convert to JSON string.

        Returns
        -------
        str
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    def to_bytes(self) -> bytes:
        """
        Convert to bytes for CustomData.

        Returns
        -------
        bytes
            UTF-8 encoded JSON bytes.
        """
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> FactorValues:
        """
        Create FactorValues from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with ts_event and factors keys.

        Returns
        -------
        FactorValues
            New FactorValues instance.
        """
        return cls(
            ts_event=data["ts_event"],
            factors=data.get("factors", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> FactorValues:
        """
        Create FactorValues from JSON string.

        Parameters
        ----------
        json_str : str
            JSON string.

        Returns
        -------
        FactorValues
            New FactorValues instance.
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_bytes(cls, data: bytes) -> FactorValues:
        """
        Create FactorValues from bytes.

        Parameters
        ----------
        data : bytes
            UTF-8 encoded JSON bytes.

        Returns
        -------
        FactorValues
            New FactorValues instance.
        """
        return cls.from_json(data.decode("utf-8"))

    def to_custom_data(self, ts_init: int | None = None):
        """
        Convert to Nautilus CustomData for MessageBus publishing.

        Parameters
        ----------
        ts_init : int, optional
            Initialization timestamp in nanoseconds. Defaults to ts_event.

        Returns
        -------
        CustomData
            Nautilus CustomData instance.

        Note
        ----
        Requires nautilus_trader to be installed.
        """
        from nautilus_trader.core.nautilus_pyo3 import CustomData, DataType

        return CustomData(
            data_type=DataType(type(self).__name__),
            value=self.to_bytes(),
            ts_event=self.ts_event,
            ts_init=ts_init if ts_init is not None else self.ts_event,
        )

    @classmethod
    def from_custom_data(cls, custom_data) -> FactorValues:
        """
        Create FactorValues from Nautilus CustomData.

        Parameters
        ----------
        custom_data : CustomData
            Nautilus CustomData instance.

        Returns
        -------
        FactorValues
            New FactorValues instance.
        """
        # CustomData.value returns string, need to encode back to bytes
        if isinstance(custom_data.value, str):
            return cls.from_json(custom_data.value)
        return cls.from_bytes(custom_data.value)
