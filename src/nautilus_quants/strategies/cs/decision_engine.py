# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
DecisionEngineActor - Transforms FactorValues into RebalanceOrders.

Subscribes to FactorValues from FactorEngineActor, reads current positions
from cache, computes rebalance decisions, and publishes RebalanceOrders
via MessageBus.

Does not submit orders. Pure decision-making.

Selection algorithm is pluggable via SelectionPolicy (Strategy Pattern):
- FMZSelectionPolicy: sort + bottom-N long / top-N short, sticky hold
- TopKDropoutSelectionPolicy: active rotation with n_drop per leg

NETTING mode: direction flips emit a single FLIP action (instead of
separate CLOSE + OPEN). The execution layer resolves actual quantities
from cache positions.

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
from nautilus_quants.strategies.cs.selection_policy import (
    FMZSelectionPolicy,
    SelectionPolicy,
    TopKDropoutSelectionPolicy,
)
from nautilus_quants.strategies.cs.types import RebalanceOrders

_SELECTION_POLICIES: dict[str, type] = {
    "FMZSelectionPolicy": FMZSelectionPolicy,
    "TopKDropoutSelectionPolicy": TopKDropoutSelectionPolicy,
}


class DecisionEngineActor(Actor):
    """
    Decision engine: FactorValues → RebalanceOrders.

    Uses a pluggable SelectionPolicy to determine target long/short sets,
    then diffs against current positions to produce CLOSE/OPEN/FLIP intents.
    """

    def __init__(self, config: DecisionEngineActorConfig) -> None:
        super().__init__(config)
        self._signal_count: int = 0
        self._bars_until_rebalance: int = 0
        self._selection_policy: SelectionPolicy = self._build_selection_policy(config)

    @staticmethod
    def _build_selection_policy(config: DecisionEngineActorConfig) -> SelectionPolicy:
        policy_name = config.selection_policy
        if policy_name == "TopKDropoutSelectionPolicy":
            for field in ("topk_long", "topk_short", "n_drop_long", "n_drop_short"):
                if getattr(config, field) is None:
                    raise ValueError(f"{field} required for TopKDropoutSelectionPolicy")
            return TopKDropoutSelectionPolicy(
                config.topk_long,  # type: ignore[arg-type]
                config.topk_short,  # type: ignore[arg-type]
                config.n_drop_long,  # type: ignore[arg-type]
                config.n_drop_short,  # type: ignore[arg-type]
            )
        elif policy_name == "FMZSelectionPolicy":
            return FMZSelectionPolicy(config.n_long, config.n_short)
        else:
            raise ValueError(
                f"Unknown selection_policy: {policy_name}. "
                f"Available: {list(_SELECTION_POLICIES)}"
            )

    def on_start(self) -> None:
        """Subscribe to FactorValues on start."""
        self.subscribe_data(DataType(FactorValues))
        policy_name = self.config.selection_policy
        self.log.info(
            f"DecisionEngineActor started: "
            f"selection_policy={policy_name}, "
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
        Compute rebalance orders by diffing current positions vs policy targets.

        Intents:
        - CLOSE: close existing position (dropped from pool or delisted)
        - OPEN: open new position
        - FLIP: one-shot direction reversal (NETTING mode)
        """
        orders: list[dict[str, Any]] = []
        instruments_with_data = set(composite.keys())

        # Delisting protection: close positions with no factor data
        for inst_id in sorted(current_long - instruments_with_data):
            orders.append(self._close_order(inst_id, tags=["NO_FACTOR_DATA"]))
        for inst_id in sorted(current_short - instruments_with_data):
            orders.append(self._close_order(inst_id, order_side="BUY", tags=["NO_FACTOR_DATA"]))

        # Policy selection: compute target final sets
        final_long, final_short = self._selection_policy.select(
            composite,
            current_long & instruments_with_data,
            current_short & instruments_with_data,
        )

        # Build rank lookup for tags
        sorted_symbols = sorted(composite.items(), key=lambda x: (x[1], x[0]))
        rank_lookup = {s: i for i, (s, _) in enumerate(sorted_symbols)}

        # Unified diff: compare current vs final → CLOSE/OPEN/FLIP
        for inst_id in sorted(instruments_with_data):
            in_fl = inst_id in final_long
            in_fs = inst_id in final_short
            was_long = inst_id in current_long
            was_short = inst_id in current_short
            rank = rank_lookup.get(inst_id, -1)
            comp = composite.get(inst_id)

            if was_long and not in_fl:
                if in_fs:
                    orders.append(
                        self._flip_order(
                            inst_id,
                            "SELL",
                            ["FLIP_TO_SHORT", f"rank:{rank}"],
                            rank,
                            comp,
                        )
                    )
                else:
                    orders.append(
                        self._close_order(
                            inst_id,
                            "SELL",
                            ["DROPPED_LONG", f"rank:{rank}"],
                            rank,
                            comp,
                        )
                    )
            elif was_short and not in_fs:
                if in_fl:
                    orders.append(
                        self._flip_order(
                            inst_id,
                            "BUY",
                            ["FLIP_TO_LONG", f"rank:{rank}"],
                            rank,
                            comp,
                        )
                    )
                else:
                    orders.append(
                        self._close_order(
                            inst_id,
                            "BUY",
                            ["DROPPED_SHORT", f"rank:{rank}"],
                            rank,
                            comp,
                        )
                    )
            elif not was_long and not was_short:
                if in_fl:
                    orders.append(
                        self._open_order(
                            inst_id,
                            "BUY",
                            ["NEW_LONG", f"rank:{rank}"],
                            rank,
                            comp,
                        )
                    )
                elif in_fs:
                    orders.append(
                        self._open_order(
                            inst_id,
                            "SELL",
                            ["NEW_SHORT", f"rank:{rank}"],
                            rank,
                            comp,
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
            "intent": "CLOSE",
            "target_quote_quantity": 0,
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
            "intent": "OPEN",
            "target_quote_quantity": self.config.position_value,
            "tags": tags or [],
            "rank": rank,
            "composite": composite,
        }

    def _flip_order(
        self,
        instrument_id: str,
        order_side: str = "BUY",
        tags: list[str] | None = None,
        rank: int = -1,
        composite: float | None = None,
    ) -> dict[str, Any]:
        """One-shot flip intent. target_quote_quantity = target position value (1x).

        Execution layer resolves actual flip quantity:
        flip_qty = current_position_qty + target_qty
        """
        return {
            "instrument_id": instrument_id,
            "order_side": order_side,
            "intent": "FLIP",
            "target_quote_quantity": self.config.position_value,
            "tags": tags or [],
            "rank": rank,
            "composite": composite,
        }
