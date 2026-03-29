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
- WorldQuantSelectionPolicy: BRAIN 7-step pipeline with heterogeneous weights

Position sizing is controlled by position_mode:
- "fixed": position_value per instrument, HOLD without resize
- "equal_weight": NAV × long_share / N, continuous resize
- "weighted": capital × |weight| from policy, continuous resize

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
from nautilus_trader.model.identifiers import ClientId, Venue
from nautilus_trader.model.objects import Currency

from nautilus_quants.factors.types import FactorValues
from nautilus_quants.strategies.cs.config import DecisionEngineActorConfig
from nautilus_quants.strategies.cs.selection_policy import (
    FMZSelectionPolicy,
    SelectionPolicy,
    TargetPosition,
    TopKDropoutSelectionPolicy,
)
from nautilus_quants.strategies.cs.types import RebalanceOrders
from nautilus_quants.strategies.cs.worldquant_selection_policy import (
    WorldQuantSelectionPolicy,
)

_SELECTION_POLICIES: dict[str, type] = {
    "FMZSelectionPolicy": FMZSelectionPolicy,
    "TopKDropoutSelectionPolicy": TopKDropoutSelectionPolicy,
    "WorldQuantSelectionPolicy": WorldQuantSelectionPolicy,
}


class DecisionEngineActor(Actor):
    """
    Decision engine: FactorValues → RebalanceOrders.

    Uses a pluggable SelectionPolicy to determine target portfolio
    (list[TargetPosition]), then diffs against current positions to
    produce rebalance orders.
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
        elif policy_name == "WorldQuantSelectionPolicy":
            return WorldQuantSelectionPolicy(
                delay=config.delay,
                decay=config.decay,
                neutralization=config.neutralization,
                truncation=config.truncation,
                enable_long=config.enable_long,
                enable_short=config.enable_short,
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
        self.subscribe_data(DataType(FactorValues), client_id=ClientId(self.id.value))
        policy_name = self.config.selection_policy
        self.log.info(
            f"DecisionEngineActor started: "
            f"selection_policy={policy_name}, "
            f"position_mode={self.config.position_mode}, "
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
        """Compute rebalance orders by diffing current vs target positions."""
        orders: list[dict[str, Any]] = []
        instruments_with_data = set(composite.keys())

        # Delisting protection: close positions with no factor data
        for inst_id in sorted(current_long - instruments_with_data):
            orders.append(self._rebalance_order(inst_id, "SELL", 0, ["NO_FACTOR_DATA"]))
        for inst_id in sorted(current_short - instruments_with_data):
            orders.append(self._rebalance_order(inst_id, "BUY", 0, ["NO_FACTOR_DATA"]))

        # Policy selection: get target portfolio
        targets = self._selection_policy.select(
            composite,
            current_long & instruments_with_data,
            current_short & instruments_with_data,
        )

        target_longs = {t.symbol for t in targets if t.weight > 0}
        target_shorts = {t.symbol for t in targets if t.weight < 0}
        target_map = {t.symbol: t for t in targets}

        # Build rank from sorted factor order
        rank_counter = 0
        rank_lookup: dict[str, int] = {}
        for t in targets:
            rank_lookup[t.symbol] = rank_counter
            rank_counter += 1

        is_fixed = self.config.position_mode == "fixed"

        # Emit orders for target positions
        for t in targets:
            is_long = t.weight > 0
            was_long = t.symbol in current_long
            was_short = t.symbol in current_short
            same_dir = (is_long and was_long) or (not is_long and was_short)

            # HOLD: fixed mode skips same-direction positions (no resize)
            if same_dir and is_fixed:
                continue

            target_value = self._compute_target(t, target_longs, target_shorts)
            side = "BUY" if is_long else "SELL"
            rank = rank_lookup.get(t.symbol, -1)

            if is_long:
                tag = (
                    "FLIP_TO_LONG" if was_short
                    else "HOLD_LONG" if was_long
                    else "NEW_LONG"
                )
            else:
                tag = (
                    "FLIP_TO_SHORT" if was_long
                    else "HOLD_SHORT" if was_short
                    else "NEW_SHORT"
                )

            orders.append(
                self._rebalance_order(
                    t.symbol, side, target_value,
                    [tag, f"rank:{rank}"], rank, t.factor,
                )
            )

        # Close dropped positions (currently held but not in targets)
        for inst_id in sorted(
            (current_long | current_short) & instruments_with_data
            - target_longs - target_shorts
        ):
            side = "SELL" if inst_id in current_long else "BUY"
            rank = -1
            comp = composite.get(inst_id)
            tag = "DROPPED_LONG" if inst_id in current_long else "DROPPED_SHORT"
            orders.append(
                self._rebalance_order(
                    inst_id, side, 0,
                    [tag, f"rank:{rank}"], rank, comp,
                )
            )

        return orders

    # ------------------------------------------------------------------
    # Target value computation
    # ------------------------------------------------------------------

    def _compute_target(
        self,
        t: TargetPosition,
        target_longs: set[str],
        target_shorts: set[str],
    ) -> float:
        """Compute target_quote_quantity based on position_mode.

        - fixed: position_value (equal notional, no resize)
        - equal_weight: NAV × long_share / N (qlib parity)
        - weighted: position_value × |weight| × N (heterogeneous weights)
        """
        mode = self.config.position_mode
        if mode == "fixed":
            return self.config.position_value

        if mode == "equal_weight":
            nav = self._get_nav()
            if nav is None or nav <= 0:
                self.log.warning("NAV unavailable, falling back to position_value")
                return self.config.position_value
            is_long = t.weight > 0
            n = len(target_longs) if is_long else len(target_shorts)
            share = (
                self.config.long_share if is_long
                else (1.0 - self.config.long_share)
            )
            return nav * share / max(n, 1)

        if mode == "weighted":
            n_total = len(target_longs) + len(target_shorts)
            return self.config.position_value * abs(t.weight) * max(n_total, 1)

        return self.config.position_value

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
