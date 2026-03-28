# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for CSStrategy._execute_rebalance unified execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.strategies.cs.config import CSStrategyConfig
from nautilus_quants.strategies.cs.strategy import CSStrategy


class _FakeInstrument:
    def __init__(self, multiplier: float = 1.0, qty_precision: int = 2) -> None:
        self.multiplier = multiplier
        self._qty_precision = qty_precision

    def make_qty(self, value) -> Quantity:
        return Quantity.from_str(f"{float(value):.{self._qty_precision}f}")


def _make_strategy() -> tuple[CSStrategy, InstrumentId]:
    strategy = CSStrategy(
        CSStrategyConfig(
            instrument_ids=["LINK-USDT-SWAP.OKX"],
            execution_policy="PostLimitExecutionPolicy",
        ),
    )
    inst_id = InstrumentId.from_str("LINK-USDT-SWAP.OKX")
    strategy._instruments = {inst_id: _FakeInstrument(multiplier=1.0)}
    strategy._execution_policy = MagicMock()
    return strategy, inst_id


def _no_position_cache():
    cache = MagicMock()
    cache.positions_open.return_value = []
    return cache


def _long_position(qty: float = 10.0):
    return SimpleNamespace(
        quantity=qty,
        side=PositionSide.LONG,
        is_closed=False,
        instrument_id=InstrumentId.from_str("LINK-USDT-SWAP.OKX"),
    )


def _short_position(qty: float = 10.0):
    return SimpleNamespace(
        quantity=qty,
        side=PositionSide.SHORT,
        is_closed=False,
        instrument_id=InstrumentId.from_str("LINK-USDT-SWAP.OKX"),
    )


class TestExecuteRebalance:
    """Test unified _execute_rebalance method."""

    def test_open_new_position(self) -> None:
        strategy, _ = _make_strategy()
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "target_quote_quantity": 1000.0,
            "tags": ["NEW_LONG"],
        }

        cache = _no_position_cache()
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
                strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_open.assert_called_once()
        kwargs = strategy._execution_policy.submit_open.call_args.kwargs
        assert kwargs["quantity"] == Quantity.from_str("10.00")
        assert kwargs["target_quote_quantity"] == 1000.0

    def test_close_position(self) -> None:
        strategy, _ = _make_strategy()
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "SELL",
            "target_quote_quantity": 0,
            "tags": ["DROPPED_LONG"],
        }

        pos = _long_position()
        cache = MagicMock()
        cache.positions_open.return_value = [pos]
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_close.assert_called_once()

    def test_flip_uses_current_plus_target(self) -> None:
        strategy, _ = _make_strategy()
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "target_quote_quantity": 1000.0,
            "tags": ["FLIP_TO_LONG"],
        }

        pos = _short_position(qty=10.0)
        cache = MagicMock()
        cache.positions_open.return_value = [pos]
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
                strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_open.assert_called_once()
        kwargs = strategy._execution_policy.submit_open.call_args.kwargs
        # flip_qty = current(10) + target(10) = 20
        assert kwargs["quantity"] == Quantity.from_str("20.00")
        assert kwargs["order_side"] == OrderSide.BUY

    def test_skip_when_no_position_and_target_zero(self) -> None:
        strategy, _ = _make_strategy()
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "SELL",
            "target_quote_quantity": 0,
        }

        cache = _no_position_cache()
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_open.assert_not_called()
        strategy._execution_policy.submit_close.assert_not_called()

    def test_skip_small_delta(self) -> None:
        """Same direction, delta < min_rebalance_pct → skip."""
        strategy, _ = _make_strategy()
        # min_rebalance_pct default is 0.05 (5%)
        # current value = 10 * 100 * 1 = 1000, target = 1020 → 2% delta → skip
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "target_quote_quantity": 1020.0,
            "tags": ["HOLD_LONG"],
        }

        pos = _long_position(qty=10.0)
        cache = MagicMock()
        cache.positions_open.return_value = [pos]
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
                strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_open.assert_not_called()

    def test_resize_up(self) -> None:
        """Same direction, delta > min_rebalance_pct → resize up."""
        strategy, _ = _make_strategy()
        # current = 1000, target = 1500 → 50% delta → resize
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "target_quote_quantity": 1500.0,
            "tags": ["HOLD_LONG"],
        }

        pos = _long_position(qty=10.0)
        cache = MagicMock()
        cache.positions_open.return_value = [pos]
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
                strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_open.assert_called_once()
        kwargs = strategy._execution_policy.submit_open.call_args.kwargs
        assert kwargs["order_side"] == OrderSide.BUY
        assert kwargs["quantity"] == Quantity.from_str("5.00")  # 500/100

    def test_resize_down(self) -> None:
        """Same direction, negative delta → resize down via submit_reduce."""
        strategy, _ = _make_strategy()
        # current = 1000, target = 500 → -50% delta → resize down
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "target_quote_quantity": 500.0,
            "tags": ["HOLD_LONG"],
        }

        pos = _long_position(qty=10.0)
        cache = MagicMock()
        cache.positions_open.return_value = [pos]
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
                strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_open.assert_not_called()
        strategy._execution_policy.submit_reduce.assert_called_once()
        kwargs = strategy._execution_policy.submit_reduce.call_args.kwargs
        assert kwargs["order_side"] == OrderSide.SELL  # closing side for long
        assert kwargs["quantity"] == Quantity.from_str("5.00")  # 500/100

    def test_resize_skips_when_make_qty_raises(self) -> None:
        """Resize skips gracefully when qty rounds to zero."""
        strategy, _ = _make_strategy()
        inst_id = InstrumentId.from_str("LINK-USDT-SWAP.OKX")
        bad_instrument = MagicMock()
        bad_instrument.multiplier = 1.0
        bad_instrument.make_qty.side_effect = ValueError("below size increment")
        strategy._instruments = {inst_id: bad_instrument}

        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "target_quote_quantity": 1500.0,
            "tags": ["HOLD_LONG"],
        }

        pos = _long_position(qty=10.0)
        cache = MagicMock()
        cache.positions_open.return_value = [pos]
        with patch.object(CSStrategy, "cache", new_callable=PropertyMock, return_value=cache):
            with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
                strategy._execute_rebalance(order_dict)

        strategy._execution_policy.submit_open.assert_not_called()
        strategy._execution_policy.submit_reduce.assert_not_called()


class TestCSStrategyExternalOrderClaims:
    """external_order_claims auto-population from instrument_ids."""

    def test_auto_populated_from_instrument_ids(self) -> None:
        strategy = CSStrategy(
            CSStrategyConfig(
                instrument_ids=["LINK-USDT-SWAP.OKX", "BTC-USDT-SWAP.OKX"],
                execution_policy="PostLimitExecutionPolicy",
            ),
        )
        expected = [
            InstrumentId.from_str("LINK-USDT-SWAP.OKX"),
            InstrumentId.from_str("BTC-USDT-SWAP.OKX"),
        ]
        assert strategy.external_order_claims == expected

    def test_explicit_claims_not_overridden(self) -> None:
        strategy = CSStrategy(
            CSStrategyConfig(
                instrument_ids=["LINK-USDT-SWAP.OKX", "BTC-USDT-SWAP.OKX"],
                execution_policy="PostLimitExecutionPolicy",
                external_order_claims=["LINK-USDT-SWAP.OKX"],
            ),
        )
        assert strategy.external_order_claims == [
            InstrumentId.from_str("LINK-USDT-SWAP.OKX"),
        ]

    def test_empty_instrument_ids(self) -> None:
        strategy = CSStrategy(
            CSStrategyConfig(
                instrument_ids=[],
                execution_policy="PostLimitExecutionPolicy",
            ),
        )
        assert strategy.external_order_claims == []
