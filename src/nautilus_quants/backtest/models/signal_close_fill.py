# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
SignalCloseFillModel - Deterministic fill model using signal-time anchor prices.

Solves Layer 2 of backtest PnL drift: when a strategy computes an execution price
at signal time (via EventTimePriceBook), this fill model injects that price into the
matching engine so that fills are deterministic regardless of OrderBook callback order.

Usage:
    Orders must carry exec_algorithm_params={"anchor_px": "<price>"} to activate
    anchor-based fills. Orders without this parameter fall back to default matching.
"""

from __future__ import annotations

from decimal import Decimal

from nautilus_trader.backtest.config import FillModelConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import BookOrder
from nautilus_trader.model.enums import BookType, OrderSide
from nautilus_trader.model.objects import Quantity


class SignalCloseFillModelConfig(FillModelConfig, frozen=True):
    """Configuration for SignalCloseFillModel (no extra parameters needed)."""


class SignalCloseFillModel(FillModel):
    """
    Fill model that uses a strategy-supplied anchor price for deterministic fills.

    When an order carries exec_algorithm_params["anchor_px"], this model constructs
    a synthetic OrderBook with a single contra-side level at the anchor price,
    ensuring the fill occurs exactly at that price.

    When no anchor_px is present, returns None to fall back to default matching.
    """

    def __init__(self, config: SignalCloseFillModelConfig | None = None) -> None:
        super().__init__(config=config)

    def is_slipped(self) -> bool:
        """Anchor fills never slip."""
        return False

    def get_orderbook_for_fill_simulation(
        self, instrument, order, best_bid, best_ask,
    ):
        """
        Build a synthetic OrderBook at the anchor price if available.

        Parameters
        ----------
        instrument : Instrument
            The instrument being traded.
        order : Order
            The order to simulate fills for.
        best_bid : Price
            Current best bid (unused when anchor is present).
        best_ask : Price
            Current best ask (unused when anchor is present).

        Returns
        -------
        OrderBook | None
            Synthetic book at anchor price, or None for default matching.
        """
        params = order.exec_algorithm_params
        if not params or "anchor_px" not in params:
            return None

        anchor_price = instrument.make_price(Decimal(params["anchor_px"]))

        book = OrderBook(
            instrument_id=instrument.id,
            book_type=BookType.L2_MBP,
        )

        # BUY order needs SELL-side liquidity; SELL order needs BUY-side liquidity
        contra_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY

        book.add(
            BookOrder(
                side=contra_side,
                price=anchor_price,
                size=Quantity(1_000_000, instrument.size_precision),
                order_id=1,
            ),
            ts_event=0,
        )

        return book
