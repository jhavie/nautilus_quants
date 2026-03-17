# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""OrderState enum and OrderExecutionState for PostLimitExecAlgorithm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity


class OrderState(Enum):
    """State machine states for a single order execution sequence."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CHASING = "CHASING"
    MARKET_FALLBACK = "MARKET_FALLBACK"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Legal state transitions (current_state -> set of allowed next states)
VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.PENDING: {OrderState.ACTIVE, OrderState.FAILED},
    OrderState.ACTIVE: {
        OrderState.COMPLETED,
        OrderState.CHASING,
        OrderState.MARKET_FALLBACK,
    },
    OrderState.CHASING: {
        OrderState.ACTIVE,
        OrderState.COMPLETED,
        OrderState.FAILED,
        OrderState.MARKET_FALLBACK,
    },
    OrderState.MARKET_FALLBACK: {
        OrderState.COMPLETED,
        OrderState.FAILED,
    },
    OrderState.COMPLETED: set(),
    OrderState.FAILED: set(),
}

TERMINAL_STATES = {OrderState.COMPLETED, OrderState.FAILED}


@dataclass
class OrderExecutionState:
    """Tracks the lifecycle of a single order execution sequence.

    Each incoming MarketOrder with ``exec_algorithm_id="PostLimit"`` gets one
    ``OrderExecutionState`` that persists until the sequence reaches a terminal
    state (COMPLETED or FAILED).
    """

    primary_order_id: ClientOrderId
    instrument_id: InstrumentId
    side: OrderSide
    total_quantity: Quantity
    anchor_px: float
    reduce_only: bool = False

    state: OrderState = OrderState.PENDING
    current_limit_order_id: ClientOrderId | None = None
    chase_count: int = 0
    timer_name: str = ""
    created_ns: int = 0
    completed_ns: int = 0
    used_market_fallback: bool = False

    # Per-order overrides from exec_algorithm_params (None = use config default)
    timeout_secs: float | None = None
    max_chase_attempts: int | None = None
    chase_step_ticks: int | None = None
    post_only: bool | None = None

    # POST_ONLY rejection retry tracking
    post_only_retreat_ticks: int = 0

    # Metrics
    limit_orders_submitted: int = 0
    last_limit_price: float = 0.0
    filled_quantity: Quantity | None = None
    fill_cost: float = 0.0  # Cumulative sum(fill_px * fill_qty) for VWAP calculation

    def __post_init__(self) -> None:
        if self.filled_quantity is None:
            self.filled_quantity = Quantity.zero(self.total_quantity.precision)

    def transition_to(self, new_state: OrderState) -> None:
        """Validate and apply a state transition.

        Raises
        ------
        ValueError
            If the transition is not legal.
        """
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition: {self.state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        self.state = new_state

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES
