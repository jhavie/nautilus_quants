# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Pricing and quantity helpers for PostLimitExecAlgorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Price, Quantity

if TYPE_CHECKING:
    from nautilus_trader.execution.config import ExecAlgorithmConfig
    from nautilus_trader.model.instruments import Instrument

    from nautilus_quants.execution.post_limit.state import OrderExecutionState


def compute_limit_price(
    tick: float,
    side: OrderSide,
    anchor_px: float,
    offset_ticks: int,
    chase_count: int,
    chase_step_ticks: int,
    post_only: bool,
    best_bid: float | None,
    best_ask: float | None,
) -> float:
    """Pure-function limit price calculation used by the algorithm and tests."""
    total_offset = offset_ticks + chase_count * chase_step_ticks

    if side == OrderSide.BUY:
        base = best_bid if best_bid is not None else anchor_px
        price = base + total_offset * tick
        if post_only and best_ask is not None:
            price = min(price, best_ask - tick)
    else:
        base = best_ask if best_ask is not None else anchor_px
        price = base - total_offset * tick
        if post_only and best_bid is not None:
            price = max(price, best_bid + tick)

    return max(price, tick)


def normalize_qty_or_zero(
    *,
    instrument: Instrument,
    raw_qty: float,
    precision: int,
    instrument_id: InstrumentId,
    primary_order_id: ClientOrderId,
    logger,
) -> Quantity:
    """Normalize quantity to instrument increment, returning zero for dust."""
    if raw_qty <= 0:
        return Quantity.zero(precision)

    try:
        return instrument.make_qty(raw_qty, round_down=True)
    except ValueError as exc:
        if "rounded to zero" not in str(exc):
            raise

        logger.warning(
            "PostLimit residual below increment, treating as zero: "
            f"instrument_id={instrument_id} "
            f"primary_order_id={primary_order_id} "
            f"raw_qty={raw_qty} "
            f"size_increment={instrument.size_increment} "
            f"size_precision={instrument.size_precision}"
        )
        return Quantity.zero(precision)


def get_execution_price(cache, state: OrderExecutionState) -> float | None:
    """Get the best execution price for quote-target quantity recalculation."""
    quote = cache.quote_tick(state.instrument_id)
    if quote is not None:
        if state.side == OrderSide.BUY:
            return float(quote.ask_price) if quote.ask_price else None
        return float(quote.bid_price) if quote.bid_price else None
    return state.anchor_px if state.anchor_px > 0 else None


def compute_remaining_quantity(cache, state: OrderExecutionState, logger) -> Quantity:
    """Compute remaining base quantity from fills or quote-value target."""
    instrument = cache.instrument(state.instrument_id)
    if instrument is None:
        return state.total_quantity

    remaining: Quantity | None = None
    if state.target_quote_quantity is not None:
        remaining_value = state.target_quote_quantity - state.filled_quote_quantity
        if remaining_value <= 0:
            return Quantity.zero(state.total_quantity.precision)

        exec_price = get_execution_price(cache, state)
        if exec_price is not None and exec_price > 0:
            raw_qty = remaining_value / (exec_price * state.contract_multiplier)
            remaining = normalize_qty_or_zero(
                instrument=instrument,
                raw_qty=raw_qty,
                precision=state.total_quantity.precision,
                instrument_id=state.instrument_id,
                primary_order_id=state.primary_order_id,
                logger=logger,
            )

    if remaining is None:
        remaining_raw = float(state.total_quantity - state.filled_quantity)
        remaining = normalize_qty_or_zero(
            instrument=instrument,
            raw_qty=remaining_raw,
            precision=state.total_quantity.precision,
            instrument_id=state.instrument_id,
            primary_order_id=state.primary_order_id,
            logger=logger,
        )

    return _cap_remaining_to_primary_leaves(
        cache=cache,
        state=state,
        remaining=remaining,
        logger=logger,
    )


def _cap_remaining_to_primary_leaves(
    *,
    cache,
    state: OrderExecutionState,
    remaining: Quantity,
    logger,
) -> Quantity:
    primary = cache.order(state.primary_order_id)
    if primary is None:
        return remaining

    leaves_qty = getattr(primary, "leaves_qty", None)
    if not isinstance(leaves_qty, Quantity):
        return remaining

    if remaining <= leaves_qty:
        return remaining

    capped = Quantity(float(leaves_qty), remaining.precision)
    if capped < Quantity.zero(capped.precision):
        capped = Quantity.zero(capped.precision)

    mode = "target_quote" if state.target_quote_quantity is not None else "fixed_quantity"
    logger.warning(
        "PostLimit remaining capped by primary leaves: "
        f"primary={state.primary_order_id} "
        f"mode={mode} "
        f"requested={remaining} "
        f"capped={capped} "
        f"leaves={leaves_qty}"
    )
    return capped


def determine_limit_price(
    *,
    cache,
    instrument: Instrument,
    state: OrderExecutionState,
    config: ExecAlgorithmConfig,
    post_only_default: bool,
    chase_step_ticks_default: int,
) -> Price:
    """Resolve the next limit price from cache BBO and session state."""
    tick = float(instrument.price_increment)
    best_bid, best_ask = resolve_best_prices(cache, state.instrument_id)

    post_only = state.post_only if state.post_only is not None else post_only_default
    chase_step_ticks = (
        state.chase_step_ticks if state.chase_step_ticks is not None else chase_step_ticks_default
    )
    effective_offset = config.offset_ticks - state.post_only_retreat_ticks
    raw_price = compute_limit_price(
        tick=tick,
        side=state.side,
        anchor_px=state.anchor_px,
        offset_ticks=effective_offset,
        chase_count=0 if post_only else state.chase_count,
        chase_step_ticks=chase_step_ticks,
        post_only=post_only,
        best_bid=best_bid,
        best_ask=best_ask,
    )
    return instrument.make_price(raw_price)


def resolve_best_prices(cache, instrument_id: InstrumentId) -> tuple[float | None, float | None]:
    """Resolve best bid/ask from QuoteTick first, then OrderBook."""
    best_bid: float | None = None
    best_ask: float | None = None

    quote = cache.quote_tick(instrument_id)
    if quote is not None:
        if quote.bid_price:
            best_bid = float(quote.bid_price)
        if quote.ask_price:
            best_ask = float(quote.ask_price)

    if best_bid is None or best_ask is None:
        book = cache.order_book(instrument_id)
        if book is not None:
            if best_bid is None:
                bid_price = book.best_bid_price()
                if bid_price is not None:
                    best_bid = float(bid_price)
            if best_ask is None:
                ask_price = book.best_ask_price()
                if ask_price is not None:
                    best_ask = float(ask_price)

    return best_bid, best_ask


def compute_residual_notional(
    *,
    cache,
    instrument_id: InstrumentId,
    anchor_px: float,
    quantity: Quantity,
) -> float | None:
    """Estimate residual position notional from cache or anchor fallback."""
    quote = cache.quote_tick(instrument_id)
    if quote is not None and quote.bid_price and quote.ask_price:
        mid_price = (float(quote.bid_price) + float(quote.ask_price)) / 2
    else:
        mid_price = anchor_px

    if mid_price <= 0:
        return None
    return float(quantity) * mid_price
