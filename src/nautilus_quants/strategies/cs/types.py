# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
RebalanceOrders - Nautilus Data type for cross-sectional rebalance decisions.

NOTE: This file intentionally does NOT use `from __future__ import annotations`
because @customdataclass requires actual type objects, not string annotations.
"""

import json
from typing import Any, Dict, List, Optional

from nautilus_trader.core.data import Data
from nautilus_trader.model.custom import customdataclass


@customdataclass
class RebalanceOrders(Data):
    """
    DecisionEngineActor output: rebalance trade instructions for a single period.

    Published via MessageBus ``publish_data()``, consumed by CSStrategy via
    ``subscribe_data(DataType(RebalanceOrders))``.

    Field naming aligns with Nautilus ``MarketOrder`` model:

    - ``instrument_id`` — matches ``Order.instrument_id``
    - ``order_side`` — matches ``Order.order_side`` (BUY/SELL)
    - ``reduce_only`` — matches ``Order.reduce_only``
    - ``quote_quantity`` — matches ``Order.quote_quantity`` (USDT amount)
    - ``tags`` — matches ``Order.tags``
    - ``action`` — "CLOSE", "OPEN", or "FLIP"

    Attributes
    ----------
    ts_event : int
        Event timestamp in nanoseconds (inherited from Data).
    ts_init : int
        Initialization timestamp in nanoseconds (inherited from Data).
    orders_bytes : bytes
        JSON-encoded orders list stored as bytes for Arrow compatibility.
    """

    orders_bytes: bytes = b"[]"

    @classmethod
    def create(
        cls,
        ts_event: int,
        orders: List[Dict[str, Any]],
        ts_init: Optional[int] = None,
    ) -> "RebalanceOrders":
        """
        Create RebalanceOrders from an orders list.

        Each order dict contains fields aligned with Nautilus MarketOrder:

        - ``instrument_id`` : str — e.g. "BTCUSDT.BINANCE"
        - ``order_side`` : str — "BUY" or "SELL"
        - ``action`` : str — "CLOSE", "OPEN", or "FLIP"
        - ``reduce_only`` : bool — True for close orders
        - ``quote_quantity`` : float — USDT amount (0 for closes, position_value for opens/flips)
        - ``tags`` : list[str] — e.g. ["FLIP_TO_LONG", "rank:15"]
        - ``rank`` : int — factor rank (metadata)
        - ``composite`` : float | None — composite factor value (metadata)

        Parameters
        ----------
        ts_event : int
            Event timestamp in nanoseconds.
        orders : list[dict]
            List of order instruction dicts.
        ts_init : int, optional
            Initialization timestamp. Defaults to ts_event.

        Returns
        -------
        RebalanceOrders
        """
        return cls(
            ts_event=ts_event,
            ts_init=ts_init if ts_init is not None else ts_event,
            orders_bytes=json.dumps(orders).encode("utf-8"),
        )

    @property
    def orders(self) -> List[Dict[str, Any]]:
        """Get the orders list."""
        return json.loads(self.orders_bytes.decode("utf-8"))

    @property
    def closes(self) -> List[Dict[str, Any]]:
        """Get close orders only."""
        return [o for o in self.orders if o["action"] == "CLOSE"]

    @property
    def opens(self) -> List[Dict[str, Any]]:
        """Get open orders only."""
        return [o for o in self.orders if o["action"] == "OPEN"]

    @property
    def flips(self) -> List[Dict[str, Any]]:
        """Get flip orders only (one-shot direction reversals)."""
        return [o for o in self.orders if o["action"] == "FLIP"]
