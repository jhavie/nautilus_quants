# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PostLimit pricing and remaining-quantity helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.pricing import (
    compute_limit_price,
    compute_remaining_quantity,
    compute_residual_notional,
    normalize_qty_or_zero,
)
from nautilus_quants.execution.post_limit.state import OrderExecutionState


class _FakeInstrument:
    price_increment = 0.01
    size_increment = Quantity.from_str("0.01")
    size_precision = 2
    min_notional = 5.0

    def make_qty(self, value: float, round_down: bool = False) -> Quantity:
        if value <= 0:
            return Quantity.zero(2)
        if value < 0.01:
            raise ValueError("Invalid `value` for quantity: rounded to zero")
        if round_down:
            floored = int(value * 100) / 100
            return Quantity.from_str(f"{floored:.2f}")
        return Quantity.from_str(f"{value:.2f}")

    def make_price(self, value: float) -> float:
        return value


class TestComputeLimitPrice:
    def test_buy_post_only_clamps_inside_spread(self) -> None:
        price = compute_limit_price(
            tick=1.0,
            side=OrderSide.BUY,
            anchor_px=90.0,
            offset_ticks=5,
            chase_count=0,
            chase_step_ticks=1,
            post_only=True,
            best_bid=100.0,
            best_ask=102.0,
        )

        assert price == 101.0

    def test_sell_chase_steps_down_from_offer(self) -> None:
        price = compute_limit_price(
            tick=1.0,
            side=OrderSide.SELL,
            anchor_px=110.0,
            offset_ticks=0,
            chase_count=2,
            chase_step_ticks=2,
            post_only=False,
            best_bid=90.0,
            best_ask=100.0,
        )

        assert price == 96.0

    def test_falls_back_to_anchor_when_no_book(self) -> None:
        price = compute_limit_price(
            tick=0.01,
            side=OrderSide.BUY,
            anchor_px=50000.0,
            offset_ticks=0,
            chase_count=0,
            chase_step_ticks=1,
            post_only=False,
            best_bid=None,
            best_ask=None,
        )

        assert price == 50000.0


class TestQuantities:
    def test_normalize_qty_or_zero_returns_zero_for_dust(self) -> None:
        logger = MagicMock()

        quantity = normalize_qty_or_zero(
            instrument=_FakeInstrument(),
            raw_qty=0.005,
            precision=2,
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            primary_order_id=ClientOrderId("primary001"),
            logger=logger,
        )

        assert quantity == Quantity.zero(2)
        logger.warning.assert_called_once()

    def test_compute_remaining_quantity_uses_quote_target(self) -> None:
        state = OrderExecutionState(
            primary_order_id=ClientOrderId("primary001"),
            instrument_id=InstrumentId.from_str("SOL-USDT-SWAP.OKX"),
            side=OrderSide.BUY,
            total_quantity=Quantity.from_str("22.16"),
            anchor_px=90.20,
            filled_quantity=Quantity.from_str("11.08"),
            target_quote_quantity=2000.0,
            filled_quote_quantity=999.3052,
            contract_multiplier=1.0,
        )
        cache = MagicMock()
        cache.instrument.return_value = _FakeInstrument()
        cache.quote_tick.return_value = SimpleNamespace(ask_price=90.19, bid_price=90.18)

        remaining = compute_remaining_quantity(cache, state, MagicMock())

        assert remaining == Quantity.from_str("11.09")

    def test_compute_remaining_quantity_caps_to_primary_leaves(self) -> None:
        state = OrderExecutionState(
            primary_order_id=ClientOrderId("primary001"),
            instrument_id=InstrumentId.from_str("SOL-USDT-SWAP.OKX"),
            side=OrderSide.BUY,
            total_quantity=Quantity.from_str("12.33"),
            anchor_px=81.10,
            filled_quantity=Quantity.from_str("0.00"),
            target_quote_quantity=1000.0,
            filled_quote_quantity=0.0,
            contract_multiplier=1.0,
        )
        cache = MagicMock()
        cache.instrument.return_value = _FakeInstrument()
        cache.quote_tick.return_value = SimpleNamespace(ask_price=81.10, bid_price=81.00)
        cache.order.return_value = SimpleNamespace(
            leaves_qty=Quantity.from_str("12.32"),
            is_closed=False,
        )
        logger = MagicMock()

        remaining = compute_remaining_quantity(cache, state, logger)

        assert remaining == Quantity.from_str("12.32")
        logger.warning.assert_called_once()
        assert "remaining capped by primary leaves" in logger.warning.call_args.args[0]

    def test_compute_residual_notional_uses_mid_price_then_anchor(self) -> None:
        cache = MagicMock()
        cache.quote_tick.return_value = SimpleNamespace(bid_price=99.0, ask_price=101.0)

        mid_value = compute_residual_notional(
            cache=cache,
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            anchor_px=95.0,
            quantity=Quantity.from_str("0.10"),
        )

        assert mid_value == pytest.approx(10.0)

        cache.quote_tick.return_value = None
        anchor_value = compute_residual_notional(
            cache=cache,
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            anchor_px=95.0,
            quantity=Quantity.from_str("0.10"),
        )

        assert anchor_value == pytest.approx(9.5)
