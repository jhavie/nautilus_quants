# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FactorValues - Nautilus Data type for factor computation results.

NOTE: This file intentionally does NOT use `from __future__ import annotations`
because @customdataclass requires actual type objects, not string annotations.
"""

import json
from typing import Any, Dict, Optional

from nautilus_trader.core.data import Data
from nautilus_trader.model.custom import customdataclass


@customdataclass
class FactorValues(Data):
    """
    Factor computation results - transmitted via MessageBus as Nautilus Data.

    This is the output of FactorEngine, containing computed factor values
    for all instruments at a given timestamp. Extends Nautilus Data for
    direct MessageBus publishing via publish_data().

    Attributes
    ----------
    ts_event : int
        Event timestamp in nanoseconds (inherited from Data).
    ts_init : int
        Initialization timestamp in nanoseconds (inherited from Data).
    factors_bytes : bytes
        JSON-encoded factors dict stored as bytes for Arrow compatibility.

    The factors structure is: {factor_name: {instrument_id_str: value}}

    Example
    -------
    ```python
    values = FactorValues.create(
        ts_event=1234567890000000000,
        factors={
            "alpha001": {"ETHUSDT.BINANCE": 0.5, "BTCUSDT.BINANCE": -0.3},
            "alpha002": {"ETHUSDT.BINANCE": 0.8, "BTCUSDT.BINANCE": 0.2},
        }
    )

    # Publish directly via MessageBus
    actor.publish_data(data_type=DataType(FactorValues), data=values)
    ```
    """

    # Store factors as bytes for Arrow schema compatibility
    factors_bytes: bytes = b"{}"

    @classmethod
    def create(
        cls,
        ts_event: int,
        factors: Dict[str, Dict[str, float]],
        ts_init: Optional[int] = None,
    ) -> "FactorValues":
        """
        Create FactorValues from a factors dictionary.

        Parameters
        ----------
        ts_event : int
            Event timestamp in nanoseconds.
        factors : dict[str, dict[str, float]]
            Nested dict of {factor_name: {instrument_id_str: value}}.
        ts_init : int, optional
            Initialization timestamp. Defaults to ts_event.

        Returns
        -------
        FactorValues
            New FactorValues instance.
        """
        factors_bytes = json.dumps(factors).encode("utf-8")
        return cls(
            ts_event=ts_event,
            ts_init=ts_init if ts_init is not None else ts_event,
            factors_bytes=factors_bytes,
        )

    @property
    def factors(self) -> Dict[str, Dict[str, float]]:
        """
        Get the factors dictionary.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict of {factor_name: {instrument_id_str: value}}.
        """
        return json.loads(self.factors_bytes.decode("utf-8"))

    def get(self, factor_name: str, instrument_id: str) -> Optional[float]:
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
        factors = self.factors
        factor_values = factors.get(factor_name)
        if factor_values is None:
            return None
        return factor_values.get(instrument_id)

    def get_factor(self, factor_name: str) -> Dict[str, float]:
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            "type": "FactorValues",
            "ts_event": self._ts_event,
            "ts_init": self._ts_init,
            "factors_bytes": self.factors_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorValues":
        """
        Create FactorValues from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with ts_event, ts_init, and factors_bytes keys.

        Returns
        -------
        FactorValues
            New FactorValues instance.
        """
        data.pop("type", None)
        return cls(**data)

    def to_json(self) -> str:
        """
        Convert to JSON string (convenience method).

        Returns
        -------
        str
            JSON string representation with factors.
        """
        return json.dumps({
            "ts_event": self.ts_event,
            "factors": self.factors,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "FactorValues":
        """
        Create FactorValues from JSON string.

        Parameters
        ----------
        json_str : str
            JSON string with ts_event and factors.

        Returns
        -------
        FactorValues
            New FactorValues instance.
        """
        data = json.loads(json_str)
        return cls.create(
            ts_event=data["ts_event"],
            factors=data.get("factors", {}),
        )

    def to_custom_data(self, ts_init: Optional[int] = None):
        """
        Convert to Nautilus CustomData for legacy compatibility.

        Note: Since FactorValues now inherits from Data, you can publish
        it directly via publish_data(). This method is for backward compatibility.

        Parameters
        ----------
        ts_init : int, optional
            Initialization timestamp. Defaults to ts_event.

        Returns
        -------
        CustomData
            Nautilus CustomData instance.
        """
        from nautilus_trader.core.nautilus_pyo3 import CustomData, DataType

        return CustomData(
            data_type=DataType("FactorValues"),
            value=self.to_bytes(),
            ts_event=self.ts_event,
            ts_init=ts_init if ts_init is not None else self.ts_event,
        )

    @classmethod
    def from_custom_data(cls, custom_data) -> "FactorValues":
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
        if isinstance(custom_data.value, str):
            return cls.from_json(custom_data.value)
        return cls.from_bytes(custom_data.value)
