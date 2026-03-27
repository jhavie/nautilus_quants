# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
DecisionEngineActor - Transforms FactorValues into RebalanceOrders.

Subscribes to FactorValues from FactorEngineActor, reads current positions
from cache, computes rebalance decisions, and publishes RebalanceOrders
via MessageBus.

Does not submit orders. Pure decision-making. Each order carries
(instrument_id, order_side, target_quote_quantity) — the execution layer
derives the actual operation (open/close/flip/resize) from cache state.

Selection algorithm is pluggable via SelectionPolicy (Strategy Pattern):
- FMZSelectionPolicy: sort + bottom-N long / top-N short, sticky hold
- TopKDropoutSelectionPolicy: active rotation with n_drop per leg

When rebalance_to_weights is True, target_quote_quantity is computed from
normalized factor scores × NAV (qlib-style). Otherwise, fixed position_value.

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
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency

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
        """Compute rebalance orders by diffing current vs target positions.

        Each order carries (instrument_id, order_side, target_quote_quantity).
        target_quote_quantity > 0 means "target position at this value";
        target_quote_quantity = 0 means "close position".
        The execution layer derives the operation from cache state.
        """
        orders: list[dict[str, Any]] = []
        instruments_with_data = set(composite.keys())

        # Delisting protection: close positions with no factor data
        for inst_id in sorted(current_long - instruments_with_data):
            orders.append(self._rebalance_order(inst_id, "SELL", 0, ["NO_FACTOR_DATA"]))
        for inst_id in sorted(current_short - instruments_with_data):
            orders.append(self._rebalance_order(inst_id, "BUY", 0, ["NO_FACTOR_DATA"]))

        # Policy selection: compute target final sets
        final_long, final_short = self._selection_policy.select(
            composite,
            current_long & instruments_with_data,
            current_short & instruments_with_data,
        )

        n_positions = max(len(final_long) + len(final_short), 1)

        # Build rank lookup for tags
        sorted_symbols = sorted(composite.items(), key=lambda x: (x[1], x[0]))
        rank_lookup = {s: i for i, (s, _) in enumerate(sorted_symbols)}

        # Emit rebalance orders for every instrument with a target state change
        for inst_id in sorted(instruments_with_data):
            in_fl = inst_id in final_long
            in_fs = inst_id in final_short
            was_long = inst_id in current_long
            was_short = inst_id in current_short
            rank = rank_lookup.get(inst_id, -1)
            comp = composite.get(inst_id)

            if in_fl:
                target = self._compute_target(
                    inst_id, final_long, final_short, composite, n_positions,
                )
                if was_short:
                    tag = "FLIP_TO_LONG"
                elif was_long:
                    tag = "HOLD_LONG"
                else:
                    tag = "NEW_LONG"
                orders.append(
                    self._rebalance_order(
                        inst_id, "BUY", target,
                        [tag, f"rank:{rank}"], rank, comp,
                    )
                )
            elif in_fs:
                target = self._compute_target(
                    inst_id, final_long, final_short, composite, n_positions,
                )
                if was_long:
                    tag = "FLIP_TO_SHORT"
                elif was_short:
                    tag = "HOLD_SHORT"
                else:
                    tag = "NEW_SHORT"
                orders.append(
                    self._rebalance_order(
                        inst_id, "SELL", target,
                        [tag, f"rank:{rank}"], rank, comp,
                    )
                )
            else:
                # Not in any target set → close if currently held
                if was_long:
                    orders.append(
                        self._rebalance_order(
                            inst_id, "SELL", 0,
                            ["DROPPED_LONG", f"rank:{rank}"], rank, comp,
                        )
                    )
                elif was_short:
                    orders.append(
                        self._rebalance_order(
                            inst_id, "BUY", 0,
                            ["DROPPED_SHORT", f"rank:{rank}"], rank, comp,
                        )
                    )

        return orders

    # ------------------------------------------------------------------
    # Target value computation
    # ------------------------------------------------------------------

    def _compute_target(
        self,
        inst_id: str,
        final_long: set[str],
        final_short: set[str],
        composite: dict[str, float],
        n_positions: int,
    ) -> float:
        """Compute target_quote_quantity for an instrument."""
        if not self.config.rebalance_to_weights:
            return self.config.position_value

        nav = self._get_nav()
        if nav is None or nav <= 0:
            self.log.warning("NAV unavailable, falling back to position_value")
            return self.config.position_value

        # Score-proportional weight within the instrument's leg
        if inst_id in final_long:
            leg_scores = {s: -composite[s] for s in final_long}
        else:
            leg_scores = {s: composite[s] for s in final_short}

        total = sum(abs(v) for v in leg_scores.values()) or 1.0
        weight = abs(leg_scores.get(inst_id, 0)) / total * 0.5
        return weight * nav

    def _get_nav(self) -> float | None:
        """Read portfolio NAV from cache via compute_mtm_equity."""
        if self.config.venue_name is None:
            return None
        try:
            from nautilus_quants.utils.equity import compute_mtm_equity

            venue = Venue(self.config.venue_name)
            currency = Currency.from_str("USDT")
            return compute_mtm_equity(self.portfolio, venue, currency)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Order dict construction
    # ------------------------------------------------------------------

    @staticmethod
    def _rebalance_order(
        instrument_id: str,
        order_side: str,
        target_quote_quantity: float,
        tags: list[str] | None = None,
        rank: int = -1,
        composite: float | None = None,
    ) -> dict[str, Any]:
        return {
            "instrument_id": instrument_id,
            "order_side": order_side,
            "target_quote_quantity": target_quote_quantity,
            "tags": tags or [],
            "rank": rank,
            "composite": composite,
        }
