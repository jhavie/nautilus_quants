# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for BracketExecutionPolicyWrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from nautilus_trader.model.enums import ContingencyType, OrderSide, OrderType
from nautilus_trader.model.identifiers import (
    ExecAlgorithmId,
    InstrumentId,
    StrategyId,
)
from nautilus_trader.model.objects import Price, Quantity

from nautilus_quants.strategies.cs.config import BracketConfig
from nautilus_quants.strategies.cs.execution_policy import (
    BracketExecutionPolicyWrapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_instrument(price_precision: int = 1, qty_precision: int = 2):
    inst = MagicMock()
    inst.make_price = lambda v: Price.from_str(f"{float(v):.{price_precision}f}")
    inst.make_qty = lambda v: Quantity.from_str(f"{float(v):.{qty_precision}f}")
    return inst


def _make_strategy(*, quote_ask: float = 100.0, quote_bid: float = 99.9):
    strategy = MagicMock()
    strategy.id = StrategyId("CSStrategy-001")

    quote = SimpleNamespace(ask_price=quote_ask, bid_price=quote_bid)
    strategy.cache.quote_tick.return_value = quote

    inst_id = InstrumentId.from_str("BTC-USDT-SWAP.OKX")
    strategy.cache.instrument.return_value = _make_instrument()
    strategy.cache.orders_open.return_value = []

    bracket_list = MagicMock()
    bracket_list.id = "OL-001"
    strategy.order_factory.bracket.return_value = bracket_list

    return strategy, inst_id


def _make_wrapper(
    strategy,
    *,
    tp_pct: float | None = 0.03,
    sl_pct: float | None = 0.02,
    entry_algo: str | None = None,
) -> BracketExecutionPolicyWrapper:
    inner = MagicMock()
    config = BracketConfig(
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        entry_exec_algorithm_id=entry_algo,
    )
    return BracketExecutionPolicyWrapper(inner, strategy, config)


# ---------------------------------------------------------------------------
# TP/SL price calculation
# ---------------------------------------------------------------------------


class TestTPSLPriceComputation:
    def test_long_tp_price(self) -> None:
        strategy, _ = _make_strategy()
        wrapper = _make_wrapper(strategy, tp_pct=0.03)
        assert wrapper._compute_tp_price(100.0, OrderSide.BUY) == pytest.approx(103.0)

    def test_long_sl_price(self) -> None:
        strategy, _ = _make_strategy()
        wrapper = _make_wrapper(strategy, sl_pct=0.02)
        assert wrapper._compute_sl_price(100.0, OrderSide.BUY) == pytest.approx(98.0)

    def test_short_tp_price(self) -> None:
        strategy, _ = _make_strategy()
        wrapper = _make_wrapper(strategy, tp_pct=0.03)
        assert wrapper._compute_tp_price(100.0, OrderSide.SELL) == pytest.approx(97.0)

    def test_short_sl_price(self) -> None:
        strategy, _ = _make_strategy()
        wrapper = _make_wrapper(strategy, sl_pct=0.02)
        assert wrapper._compute_sl_price(100.0, OrderSide.SELL) == pytest.approx(102.0)


# ---------------------------------------------------------------------------
# submit_open — bracket creation
# ---------------------------------------------------------------------------


class TestSubmitOpenBracket:
    def test_creates_bracket_order_list(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy)
        qty = Quantity.from_str("1.00")

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=qty,
            intent="OPEN",
        )

        strategy.order_factory.bracket.assert_called_once()
        strategy.submit_order_list.assert_called_once()

    def test_bracket_kwargs_contain_tp_and_sl(self) -> None:
        strategy, inst_id = _make_strategy(quote_ask=100.0)
        wrapper = _make_wrapper(strategy, tp_pct=0.03, sl_pct=0.02)

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        kwargs = strategy.order_factory.bracket.call_args.kwargs
        # TP: 100 * 1.03 = 103.0
        assert kwargs["tp_order_type"] == OrderType.LIMIT
        assert kwargs["tp_price"] == Price.from_str("103.0")
        # SL: 100 * 0.98 = 98.0
        assert kwargs["sl_order_type"] == OrderType.STOP_MARKET
        assert kwargs["sl_trigger_price"] == Price.from_str("98.0")

    def test_short_bracket_reverses_tp_sl_direction(self) -> None:
        strategy, inst_id = _make_strategy(quote_bid=100.0)
        wrapper = _make_wrapper(strategy, tp_pct=0.03, sl_pct=0.02)

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        kwargs = strategy.order_factory.bracket.call_args.kwargs
        # Short TP: 100 * 0.97 = 97.0
        assert kwargs["tp_price"] == Price.from_str("97.0")
        # Short SL: 100 * 1.02 = 102.0
        assert kwargs["sl_trigger_price"] == Price.from_str("102.0")

    def test_tp_only_no_sl(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy, tp_pct=0.05, sl_pct=None)

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        kwargs = strategy.order_factory.bracket.call_args.kwargs
        assert "tp_price" in kwargs
        assert "sl_trigger_price" not in kwargs

    def test_sl_only_no_tp(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy, tp_pct=None, sl_pct=0.02)

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        kwargs = strategy.order_factory.bracket.call_args.kwargs
        assert "tp_price" not in kwargs
        assert "sl_trigger_price" in kwargs


# ---------------------------------------------------------------------------
# submit_open — PostLimit exec algorithm
# ---------------------------------------------------------------------------


class TestSubmitOpenWithPostLimit:
    def test_passes_entry_exec_algorithm_id(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy, entry_algo="PostLimit")

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
            target_quote_quantity=1000.0,
            contract_multiplier=1.0,
        )

        kwargs = strategy.order_factory.bracket.call_args.kwargs
        assert kwargs["entry_exec_algorithm_id"] == ExecAlgorithmId("PostLimit")
        params = kwargs["entry_exec_algorithm_params"]
        assert "anchor_px" in params
        assert params["target_quote_quantity"] == "1000.0"
        assert params["intent"] == "OPEN"

    def test_no_entry_algo_when_not_configured(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy, entry_algo=None)

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        kwargs = strategy.order_factory.bracket.call_args.kwargs
        assert "entry_exec_algorithm_id" not in kwargs


# ---------------------------------------------------------------------------
# submit_open — no bracket config → delegate to inner
# ---------------------------------------------------------------------------


class TestSubmitOpenNoBracket:
    def test_delegates_to_inner_when_no_tp_sl(self) -> None:
        strategy, inst_id = _make_strategy()
        inner = MagicMock()
        config = BracketConfig()  # both pcts are None
        wrapper = BracketExecutionPolicyWrapper(inner, strategy, config)

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        inner.submit_open.assert_called_once()
        strategy.order_factory.bracket.assert_not_called()

    def test_delegates_when_no_price_available(self) -> None:
        strategy, inst_id = _make_strategy()
        strategy.cache.quote_tick.return_value = None
        inner = MagicMock()
        config = BracketConfig(take_profit_pct=0.03, stop_loss_pct=0.02)
        wrapper = BracketExecutionPolicyWrapper(inner, strategy, config)

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        inner.submit_open.assert_called_once()
        strategy.order_factory.bracket.assert_not_called()


# ---------------------------------------------------------------------------
# submit_open — FLIP cancels existing contingent orders
# ---------------------------------------------------------------------------


class TestSubmitOpenFlip:
    def test_flip_cancels_contingent_orders_before_bracket(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy)

        oco_order = MagicMock()
        oco_order.contingency_type = ContingencyType.OCO
        oco_order.is_closed = False
        strategy.cache.orders_open.return_value = [oco_order]

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="FLIP",
        )

        strategy.cancel_order.assert_called_once_with(oco_order)
        strategy.submit_order_list.assert_called_once()

    def test_flip_skips_already_closed_orders(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy)

        closed_order = MagicMock()
        closed_order.contingency_type = ContingencyType.OCO
        closed_order.is_closed = True
        strategy.cache.orders_open.return_value = [closed_order]

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="FLIP",
        )

        strategy.cancel_order.assert_not_called()

    def test_open_intent_does_not_cancel(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy)

        oco_order = MagicMock()
        oco_order.contingency_type = ContingencyType.OCO
        oco_order.is_closed = False
        strategy.cache.orders_open.return_value = [oco_order]

        wrapper.submit_open(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.00"),
            intent="OPEN",
        )

        strategy.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# submit_close — cancel contingent + delegate
# ---------------------------------------------------------------------------


class TestSubmitClose:
    def test_cancels_contingent_then_delegates(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy)

        position = MagicMock()
        position.instrument_id = inst_id
        position.quantity = Quantity.from_str("5.00")

        oco_order = MagicMock()
        oco_order.contingency_type = ContingencyType.OCO
        oco_order.is_closed = False
        strategy.cache.orders_open.return_value = [oco_order]

        wrapper.submit_close(position, tags=["CLOSE"])

        strategy.cancel_order.assert_called_once_with(oco_order)
        wrapper._inner.submit_close.assert_called_once_with(
            position, tags=["CLOSE"],
        )

    def test_no_contingent_orders_still_delegates(self) -> None:
        strategy, inst_id = _make_strategy()
        wrapper = _make_wrapper(strategy)

        position = MagicMock()
        position.instrument_id = inst_id
        strategy.cache.orders_open.return_value = []

        wrapper.submit_close(position)

        strategy.cancel_order.assert_not_called()
        wrapper._inner.submit_close.assert_called_once()


# ---------------------------------------------------------------------------
# CSStrategy factory wiring
# ---------------------------------------------------------------------------


class TestCSStrategyFactoryWiring:
    def test_bracket_config_creates_wrapper(self) -> None:
        from nautilus_quants.strategies.cs.config import CSStrategyConfig
        from nautilus_quants.strategies.cs.strategy import CSStrategy

        config = CSStrategyConfig(
            instrument_ids=["BTC-USDT-SWAP.OKX"],
            execution_policy="MarketExecutionPolicy",
            bracket=BracketConfig(take_profit_pct=0.03, stop_loss_pct=0.02),
        )
        strategy = CSStrategy(config)
        assert isinstance(strategy._execution_policy, BracketExecutionPolicyWrapper)

    def test_no_bracket_config_uses_inner_directly(self) -> None:
        from nautilus_quants.strategies.cs.config import CSStrategyConfig
        from nautilus_quants.strategies.cs.execution_policy import (
            MarketExecutionPolicy,
        )
        from nautilus_quants.strategies.cs.strategy import CSStrategy

        config = CSStrategyConfig(
            instrument_ids=["BTC-USDT-SWAP.OKX"],
            execution_policy="MarketExecutionPolicy",
        )
        strategy = CSStrategy(config)
        assert isinstance(strategy._execution_policy, MarketExecutionPolicy)

    def test_postlimit_bracket_auto_sets_entry_algo(self) -> None:
        from nautilus_quants.strategies.cs.config import CSStrategyConfig

        config = CSStrategyConfig(
            instrument_ids=["BTC-USDT-SWAP.OKX"],
            execution_policy="PostLimitExecutionPolicy",
            bracket=BracketConfig(take_profit_pct=0.03, stop_loss_pct=0.02),
        )
        from nautilus_quants.strategies.cs.strategy import CSStrategy

        strategy = CSStrategy(config)
        wrapper = strategy._execution_policy
        assert isinstance(wrapper, BracketExecutionPolicyWrapper)
        assert wrapper._entry_algo_id == ExecAlgorithmId("PostLimit")
