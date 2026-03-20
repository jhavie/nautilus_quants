# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
DecisionEngineActor - Transforms FactorValues into RebalanceOrders.

Subscribes to FactorValues from FactorEngineActor, reads current positions
from cache, computes FMZ-style rebalance decisions (sort + bottom N long /
top N short), and publishes RebalanceOrders via MessageBus.

Does not submit orders. Pure decision-making.

Constitution Compliance:
    - Extends Nautilus Actor base class (Principle I)
    - Configuration-driven via ActorConfig (Principle II)
    - Manages state/filtering, does not place orders (Principle V)
"""

from __future__ import annotations

import math
from typing import Any

from nautilus_trader.common.actor import Actor
from nautilus_trader.model.data import DataType

from nautilus_quants.factors.types import FactorValues
from nautilus_quants.strategies.cs.config import DecisionEngineActorConfig
from nautilus_quants.strategies.cs.types import RebalanceOrders


class DecisionEngineActor(Actor):
    """
    Decision engine: FactorValues → RebalanceOrders.

    FMZ core logic:
    1. Sort instruments by composite factor value (ascending)
    2. Long bottom N (lowest values)
    3. Short top N (highest values)
    4. Only change positions when direction flips
    5. Close positions with missing factor data (delisting protection)
    """

    def __init__(self, config: DecisionEngineActorConfig) -> None:
        super().__init__(config)
        self._signal_count: int = 0
        self._bars_until_rebalance: int = 0

    def on_start(self) -> None:
        """Subscribe to FactorValues on start."""
        self.subscribe_data(DataType(FactorValues))
        self.log.info(
            f"DecisionEngineActor started: "
            f"n_long={self.config.n_long}, n_short={self.config.n_short}, "
            f"position_value={self.config.position_value}, "
            f"rebalance_interval={self.config.rebalance_interval}"
        )

    def on_data(self, data: object) -> None:
        """Process FactorValues and publish RebalanceOrders."""
        if not isinstance(data, FactorValues):
            return

        composite = data.factors.get(self.config.composite_factor, {})
        if not composite:
            return

        # Filter NaN values
        composite = {k: v for k, v in composite.items() if not math.isnan(v)}
        if not composite:
            return

        # Rebalance gate
        self._signal_count += 1
        if self._bars_until_rebalance > 0:
            self._bars_until_rebalance -= 1
            return
        self._bars_until_rebalance = self.config.rebalance_interval - 1

        # Validate sufficient instruments
        required = self.config.n_long + self.config.n_short
        if len(composite) < required:
            self.log.warning(
                f"Skip rebalance: {len(composite)} instruments < {required} required"
            )
            return

        # Read current positions from cache
        current_long, current_short = self._get_current_positions()

        # Compute rebalance orders
        orders = self._compute_orders(composite, current_long, current_short)

        if orders:
            rebalance = RebalanceOrders.create(ts_event=data.ts_event, orders=orders)
            self.publish_data(DataType(RebalanceOrders), rebalance)
            self.log.info(
                f"Published RebalanceOrders: {len(orders)} orders "
                f"(signal #{self._signal_count})"
            )

    def _get_current_positions(self) -> tuple[set[str], set[str]]:
        """Read current open positions from cache."""
        long_positions: set[str] = set()
        short_positions: set[str] = set()
        for position in self.cache.positions_open():
            inst_id = str(position.instrument_id)
            if position.is_long:
                long_positions.add(inst_id)
            elif position.is_short:
                short_positions.add(inst_id)
        return long_positions, short_positions

    def _compute_orders(
        self,
        composite: dict[str, float],
        current_long: set[str],
        current_short: set[str],
    ) -> list[dict[str, Any]]:
        """
        FMZ core logic: sort + target selection + diff with current positions.

        Returns a list of order instruction dicts aligned with Nautilus MarketOrder fields.
        """
        sorted_symbols = sorted(composite.items(), key=lambda x: (x[1], x[0]))
        rank_lookup = {s: i for i, (s, _) in enumerate(sorted_symbols)}
        long_targets = set(s for s, _ in sorted_symbols[: self.config.n_long])
        short_targets = set(s for s, _ in sorted_symbols[-self.config.n_short :])

        orders: list[dict[str, Any]] = []
        instruments_with_data = set(composite.keys())

        # Delisting protection: close positions with no factor data
        for inst_id in sorted(current_long - instruments_with_data):
            orders.append(
                self._close_order(inst_id, tags=["NO_FACTOR_DATA"])
            )
        for inst_id in sorted(current_short - instruments_with_data):
            orders.append(
                self._close_order(inst_id, order_side="BUY", tags=["NO_FACTOR_DATA"])
            )

        # FMZ core: only act when direction changes
        for inst_id in sorted(composite.keys()):
            is_long_target = inst_id in long_targets
            is_short_target = inst_id in short_targets
            currently_long = inst_id in current_long
            currently_short = inst_id in current_short
            rank = rank_lookup.get(inst_id, -1)
            comp = composite.get(inst_id)

            if is_long_target and not currently_long:
                if currently_short:
                    orders.append(
                        self._close_order(
                            inst_id,
                            order_side="BUY",
                            tags=["FLIP_TO_LONG"],
                            rank=rank,
                            composite=comp,
                        )
                    )
                orders.append(
                    self._open_order(
                        inst_id,
                        order_side="BUY",
                        tags=["NEW_LONG", f"rank:{rank}"],
                        rank=rank,
                        composite=comp,
                    )
                )

            elif is_short_target and not currently_short:
                if currently_long:
                    orders.append(
                        self._close_order(
                            inst_id,
                            order_side="SELL",
                            tags=["FLIP_TO_SHORT"],
                            rank=rank,
                            composite=comp,
                        )
                    )
                orders.append(
                    self._open_order(
                        inst_id,
                        order_side="SELL",
                        tags=["NEW_SHORT", f"rank:{rank}"],
                        rank=rank,
                        composite=comp,
                    )
                )

        return orders

    def _close_order(
        self,
        instrument_id: str,
        order_side: str = "SELL",
        tags: list[str] | None = None,
        rank: int = -1,
        composite: float | None = None,
    ) -> dict[str, Any]:
        return {
            "instrument_id": instrument_id,
            "order_side": order_side,
            "action": "CLOSE",
            "reduce_only": True,
            "quote_quantity": 0,
            "tags": tags or [],
            "rank": rank,
            "composite": composite,
        }

    def _open_order(
        self,
        instrument_id: str,
        order_side: str = "BUY",
        tags: list[str] | None = None,
        rank: int = -1,
        composite: float | None = None,
    ) -> dict[str, Any]:
        return {
            "instrument_id": instrument_id,
            "order_side": order_side,
            "action": "OPEN",
            "reduce_only": False,
            "quote_quantity": self.config.position_value,
            "tags": tags or [],
            "rank": rank,
            "composite": composite,
        }
