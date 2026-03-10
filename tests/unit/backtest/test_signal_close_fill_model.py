# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for SignalCloseFillModel."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nautilus_trader.model.enums import BookType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from nautilus_quants.backtest.models.signal_close_fill import (
    SignalCloseFillModel,
    SignalCloseFillModelConfig,
)


def _make_instrument(instrument_id_str: str = "BTCUSDT.BINANCE") -> MagicMock:
    """Create a mock instrument."""
    instrument = MagicMock()
    instrument.id = InstrumentId.from_str(instrument_id_str)
    instrument.size_precision = 0
    instrument.make_price = lambda d: Price.from_str(str(d))
    return instrument


def _make_order(
    side: OrderSide,
    exec_algorithm_params: dict | None = None,
) -> MagicMock:
    """Create a mock order."""
    order = MagicMock()
    order.side = side
    order.exec_algorithm_params = exec_algorithm_params
    return order


class TestSignalCloseFillModel:
    """Tests for SignalCloseFillModel."""

    def test_no_exec_algorithm_params_returns_none(self) -> None:
        """Order without exec_algorithm_params falls back to default matching."""
        model = SignalCloseFillModel()
        instrument = _make_instrument()
        order = _make_order(OrderSide.BUY, exec_algorithm_params=None)

        result = model.get_orderbook_for_fill_simulation(
            instrument, order, Price.from_str("100.0"), Price.from_str("101.0"),
        )

        assert result is None

    def test_no_anchor_px_key_returns_none(self) -> None:
        """Order with params but no anchor_px key falls back to default matching."""
        model = SignalCloseFillModel()
        instrument = _make_instrument()
        order = _make_order(OrderSide.BUY, exec_algorithm_params={"other_key": "value"})

        result = model.get_orderbook_for_fill_simulation(
            instrument, order, Price.from_str("100.0"), Price.from_str("101.0"),
        )

        assert result is None

    def test_buy_order_creates_sell_side_liquidity(self) -> None:
        """BUY order with anchor_px creates SELL-side liquidity at anchor price."""
        model = SignalCloseFillModel()
        instrument = _make_instrument()
        order = _make_order(OrderSide.BUY, exec_algorithm_params={"anchor_px": "100.5"})

        book = model.get_orderbook_for_fill_simulation(
            instrument, order, Price.from_str("99.0"), Price.from_str("101.0"),
        )

        assert book is not None
        assert book.best_ask_price() == Price.from_str("100.5")
        assert book.best_bid_price() is None

    def test_sell_order_creates_buy_side_liquidity(self) -> None:
        """SELL order with anchor_px creates BUY-side liquidity at anchor price."""
        model = SignalCloseFillModel()
        instrument = _make_instrument()
        order = _make_order(OrderSide.SELL, exec_algorithm_params={"anchor_px": "100.5"})

        book = model.get_orderbook_for_fill_simulation(
            instrument, order, Price.from_str("99.0"), Price.from_str("101.0"),
        )

        assert book is not None
        assert book.best_bid_price() == Price.from_str("100.5")
        assert book.best_ask_price() is None

    def test_is_slipped_returns_false(self) -> None:
        """Anchor fills never slip."""
        model = SignalCloseFillModel()
        assert model.is_slipped() is False

    def test_config_creates_model(self) -> None:
        """SignalCloseFillModelConfig can be used to create model."""
        config = SignalCloseFillModelConfig()
        model = SignalCloseFillModel(config=config)
        assert model is not None

    def test_empty_params_dict_returns_none(self) -> None:
        """Order with empty params dict falls back to default matching."""
        model = SignalCloseFillModel()
        instrument = _make_instrument()
        order = _make_order(OrderSide.BUY, exec_algorithm_params={})

        result = model.get_orderbook_for_fill_simulation(
            instrument, order, Price.from_str("100.0"), Price.from_str("101.0"),
        )

        assert result is None

    def test_book_has_correct_instrument_id(self) -> None:
        """Synthetic book has the correct instrument_id."""
        model = SignalCloseFillModel()
        instrument = _make_instrument("ETHUSDT.BINANCE")
        order = _make_order(OrderSide.BUY, exec_algorithm_params={"anchor_px": "2000.0"})

        book = model.get_orderbook_for_fill_simulation(
            instrument, order, Price.from_str("1999.0"), Price.from_str("2001.0"),
        )

        assert book is not None
        assert book.instrument_id == InstrumentId.from_str("ETHUSDT.BINANCE")
