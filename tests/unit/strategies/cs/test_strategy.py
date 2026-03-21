# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for CSStrategy order translation semantics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

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


class TestCSStrategyOrderTranslation:
    def test_open_uses_target_quote_quantity_directly(self) -> None:
        strategy, _ = _make_strategy()
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "intent": "OPEN",
            "target_quote_quantity": 1000.0,
            "tags": ["NEW_LONG"],
        }

        with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
            strategy._execute_open(order_dict)

        strategy._execution_policy.submit_open.assert_called_once()
        kwargs = strategy._execution_policy.submit_open.call_args.kwargs
        assert kwargs["quantity"] == Quantity.from_str("10.00")
        assert kwargs["target_quote_quantity"] == 1000.0
        assert kwargs["intent"] == "OPEN"

    def test_flip_uses_current_plus_target_and_passes_total_target_quote_quantity(self) -> None:
        strategy, _ = _make_strategy()
        order_dict = {
            "instrument_id": "LINK-USDT-SWAP.OKX",
            "order_side": "BUY",
            "intent": "FLIP",
            "target_quote_quantity": 1000.0,
            "tags": ["FLIP_TO_LONG"],
        }
        cache = MagicMock()
        cache.positions_open.return_value = [SimpleNamespace(quantity=10.0)]

        with patch.object(CSStrategy, "cache", new_callable=PropertyMock) as cache_prop:
            cache_prop.return_value = cache
            with patch.object(CSStrategy, "_get_exec_price", return_value=100.0):
                strategy._execute_flip(order_dict)

        strategy._execution_policy.submit_open.assert_called_once()
        kwargs = strategy._execution_policy.submit_open.call_args.kwargs
        assert kwargs["quantity"] == Quantity.from_str("20.00")
        assert kwargs["target_quote_quantity"] == 2000.0
        assert kwargs["intent"] == "FLIP"

