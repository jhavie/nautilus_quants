# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""State, snapshots, and runtime bookkeeping for PostLimitExecAlgorithm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import msgspec
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity


class OrderState(Enum):
    """Runtime phases for a single execution session."""

    PENDING_LIMIT = "PENDING_LIMIT"
    WORKING_LIMIT = "WORKING_LIMIT"
    CANCEL_PENDING_REPRICE = "CANCEL_PENDING_REPRICE"
    CANCEL_PENDING_MARKET = "CANCEL_PENDING_MARKET"
    PENDING_MARKET = "PENDING_MARKET"
    WORKING_MARKET = "WORKING_MARKET"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SpawnKind(Enum):
    """Type of child order created by the algorithm."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"
    SWEEP = "SWEEP"


VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.PENDING_LIMIT: {
        OrderState.WORKING_LIMIT,
        OrderState.PENDING_LIMIT,
        OrderState.PENDING_MARKET,
        OrderState.FAILED,
        OrderState.COMPLETED,
    },
    OrderState.WORKING_LIMIT: {
        OrderState.CANCEL_PENDING_REPRICE,
        OrderState.CANCEL_PENDING_MARKET,
        OrderState.PENDING_MARKET,
        OrderState.COMPLETED,
        OrderState.FAILED,
    },
    OrderState.CANCEL_PENDING_REPRICE: {
        OrderState.PENDING_LIMIT,
        OrderState.WORKING_LIMIT,
        OrderState.COMPLETED,
        OrderState.FAILED,
    },
    OrderState.CANCEL_PENDING_MARKET: {
        OrderState.PENDING_MARKET,
        OrderState.WORKING_LIMIT,
        OrderState.COMPLETED,
        OrderState.FAILED,
    },
    OrderState.PENDING_MARKET: {
        OrderState.WORKING_MARKET,
        OrderState.COMPLETED,
        OrderState.FAILED,
    },
    OrderState.WORKING_MARKET: {
        OrderState.COMPLETED,
        OrderState.FAILED,
    },
    OrderState.COMPLETED: set(),
    OrderState.FAILED: set(),
}

TERMINAL_STATES = {OrderState.COMPLETED, OrderState.FAILED}
WORKING_STATES = {OrderState.WORKING_LIMIT, OrderState.WORKING_MARKET}


@dataclass
class SpawnLedgerEntry:
    """Runtime bookkeeping for the current active child order."""

    client_order_id: ClientOrderId
    sequence: int
    kind: SpawnKind
    reserved_quantity: Quantity
    accepted: bool = False
    terminal: bool = False


@dataclass
class OrderExecutionState:
    """Tracks the full execution lifecycle of a primary PostLimit request."""

    primary_order_id: ClientOrderId
    instrument_id: InstrumentId
    side: OrderSide
    total_quantity: Quantity
    anchor_px: float
    reduce_only: bool = False

    state: OrderState = OrderState.PENDING_LIMIT
    active_order_id: ClientOrderId | None = None
    active_order_kind: SpawnKind | None = None
    active_reserved_quantity: Quantity | None = None
    active_order_accepted: bool = False
    chase_count: int = 0
    spawn_sequence: int = 0
    timer_name: str = ""
    created_ns: int = 0
    completed_ns: int = 0
    used_market_fallback: bool = False
    residual_sweep_pending: bool = False

    timeout_secs: float | None = None
    max_chase_attempts: int | None = None
    chase_step_ticks: int | None = None
    post_only: bool | None = None

    post_only_retreat_ticks: int = 0
    target_quote_quantity: float | None = None
    filled_quote_quantity: float = 0.0
    contract_multiplier: float = 1.0
    intent: str = "UNKNOWN"

    limit_orders_submitted: int = 0
    last_limit_price: float = 0.0
    filled_quantity: Quantity | None = None
    fill_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.filled_quantity is None:
            self.filled_quantity = Quantity.zero(self.total_quantity.precision)

    def transition_to(self, new_state: OrderState) -> None:
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition: {self.state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        self.state = new_state

    def activate_order(
        self,
        *,
        client_order_id: ClientOrderId,
        kind: SpawnKind,
        reserved_quantity: Quantity,
        accepted: bool = False,
    ) -> None:
        self.active_order_id = client_order_id
        self.active_order_kind = kind
        self.active_reserved_quantity = reserved_quantity
        self.active_order_accepted = accepted

    def clear_active_order(self) -> None:
        self.active_order_id = None
        self.active_order_kind = None
        self.active_reserved_quantity = None
        self.active_order_accepted = False

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_working(self) -> bool:
        return self.state in WORKING_STATES


class OrderExecutionStateSnapshot(msgspec.Struct, frozen=True):
    """Serializable session state persisted across restarts."""

    primary_order_id: str
    instrument_id: str
    side: str
    total_quantity: str
    anchor_px: float
    reduce_only: bool
    state: str
    active_order_id: str | None
    active_order_kind: str | None
    active_reserved_quantity: str | None
    active_order_accepted: bool
    chase_count: int
    spawn_sequence: int
    timer_name: str
    created_ns: int
    completed_ns: int
    used_market_fallback: bool
    residual_sweep_pending: bool
    timeout_secs: float | None
    max_chase_attempts: int | None
    chase_step_ticks: int | None
    post_only: bool | None
    post_only_retreat_ticks: int
    target_quote_quantity: float | None
    filled_quote_quantity: float
    contract_multiplier: float
    intent: str
    limit_orders_submitted: int
    last_limit_price: float
    filled_quantity: str
    fill_cost: float


class OrderExecutionStateStore(msgspec.Struct, frozen=True):
    """Versioned container for on-save state."""

    version: int
    sessions: list[OrderExecutionStateSnapshot]


def encode_execution_states(states: dict[ClientOrderId, OrderExecutionState]) -> bytes:
    """Serialize runtime state to msgpack bytes."""
    store = OrderExecutionStateStore(
        version=1,
        sessions=[_snapshot_from_state(state) for state in states.values()],
    )
    return msgspec.msgpack.encode(store)


def decode_execution_states(data: bytes) -> dict[ClientOrderId, OrderExecutionState]:
    """Decode msgpack bytes into runtime state keyed by primary order ID."""
    store = msgspec.msgpack.decode(data, type=OrderExecutionStateStore)
    result: dict[ClientOrderId, OrderExecutionState] = {}
    for snapshot in store.sessions:
        state = _state_from_snapshot(snapshot)
        result[state.primary_order_id] = state
    return result


def _snapshot_from_state(state: OrderExecutionState) -> OrderExecutionStateSnapshot:
    return OrderExecutionStateSnapshot(
        primary_order_id=state.primary_order_id.value,
        instrument_id=str(state.instrument_id),
        side=state.side.name,
        total_quantity=str(state.total_quantity),
        anchor_px=state.anchor_px,
        reduce_only=state.reduce_only,
        state=state.state.value,
        active_order_id=None if state.active_order_id is None else state.active_order_id.value,
        active_order_kind=None if state.active_order_kind is None else state.active_order_kind.value,
        active_reserved_quantity=(
            None if state.active_reserved_quantity is None else str(state.active_reserved_quantity)
        ),
        active_order_accepted=state.active_order_accepted,
        chase_count=state.chase_count,
        spawn_sequence=state.spawn_sequence,
        timer_name=state.timer_name,
        created_ns=state.created_ns,
        completed_ns=state.completed_ns,
        used_market_fallback=state.used_market_fallback,
        residual_sweep_pending=state.residual_sweep_pending,
        timeout_secs=state.timeout_secs,
        max_chase_attempts=state.max_chase_attempts,
        chase_step_ticks=state.chase_step_ticks,
        post_only=state.post_only,
        post_only_retreat_ticks=state.post_only_retreat_ticks,
        target_quote_quantity=state.target_quote_quantity,
        filled_quote_quantity=state.filled_quote_quantity,
        contract_multiplier=state.contract_multiplier,
        intent=state.intent,
        limit_orders_submitted=state.limit_orders_submitted,
        last_limit_price=state.last_limit_price,
        filled_quantity=str(state.filled_quantity),
        fill_cost=state.fill_cost,
    )


def _state_from_snapshot(snapshot: OrderExecutionStateSnapshot) -> OrderExecutionState:
    state = OrderExecutionState(
        primary_order_id=ClientOrderId(snapshot.primary_order_id),
        instrument_id=InstrumentId.from_str(snapshot.instrument_id),
        side=OrderSide[snapshot.side],
        total_quantity=Quantity.from_str(snapshot.total_quantity),
        anchor_px=snapshot.anchor_px,
        reduce_only=snapshot.reduce_only,
        state=OrderState(snapshot.state),
        chase_count=snapshot.chase_count,
        spawn_sequence=snapshot.spawn_sequence,
        timer_name=snapshot.timer_name,
        created_ns=snapshot.created_ns,
        completed_ns=snapshot.completed_ns,
        used_market_fallback=snapshot.used_market_fallback,
        residual_sweep_pending=snapshot.residual_sweep_pending,
        timeout_secs=snapshot.timeout_secs,
        max_chase_attempts=snapshot.max_chase_attempts,
        chase_step_ticks=snapshot.chase_step_ticks,
        post_only=snapshot.post_only,
        post_only_retreat_ticks=snapshot.post_only_retreat_ticks,
        target_quote_quantity=snapshot.target_quote_quantity,
        filled_quote_quantity=snapshot.filled_quote_quantity,
        contract_multiplier=snapshot.contract_multiplier,
        intent=snapshot.intent,
        limit_orders_submitted=snapshot.limit_orders_submitted,
        last_limit_price=snapshot.last_limit_price,
        filled_quantity=Quantity.from_str(snapshot.filled_quantity),
        fill_cost=snapshot.fill_cost,
    )
    if snapshot.active_order_id is not None and snapshot.active_reserved_quantity is not None:
        state.activate_order(
            client_order_id=ClientOrderId(snapshot.active_order_id),
            kind=SpawnKind(snapshot.active_order_kind),
            reserved_quantity=Quantity.from_str(snapshot.active_reserved_quantity),
            accepted=snapshot.active_order_accepted,
        )
    return state
