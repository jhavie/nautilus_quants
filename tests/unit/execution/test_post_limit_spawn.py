# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for child order spawning and primary quantity mirroring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.spawn import (
    ChildOrderFactory,
    PrimaryMirror,
    spawn_linkage_fields,
)
from nautilus_quants.execution.post_limit.state import OrderExecutionState, SpawnKind


def _make_state() -> OrderExecutionState:
    return OrderExecutionState(
        primary_order_id=ClientOrderId("primary001"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        side=OrderSide.BUY,
        total_quantity=Quantity.from_str("2.00"),
        anchor_px=50000.0,
        timer_name="PostLimit-primary001",
        created_ns=123,
    )


def _make_primary() -> SimpleNamespace:
    return SimpleNamespace(
        trader_id="TRADER",
        strategy_id="STRAT",
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        client_order_id=ClientOrderId("primary001"),
        side=OrderSide.BUY,
        quantity=Quantity.from_str("2.00"),
        leaves_qty=Quantity.from_str("2.00"),
        is_quote_quantity=False,
        contingency_type="OCO",
        order_list_id="LIST001",
        linked_order_ids=["LINK1", "LINK2"],
        parent_order_id="PARENT001",
        tags=["OPEN", "POST_LIMIT"],
        venue_order_id=None,
        account_id=None,
        apply=MagicMock(),
    )


class TestSpawnLinkageFields:
    def test_extracts_linkage_and_tags(self) -> None:
        fields = spawn_linkage_fields(_make_primary())

        assert fields == {
            "contingency_type": "OCO",
            "order_list_id": "LIST001",
            "linked_order_ids": ["LINK1", "LINK2"],
            "parent_order_id": "PARENT001",
            "tags": ["OPEN", "POST_LIMIT"],
        }


class TestChildOrderFactory:
    def test_create_limit_preserves_exec_and_linkage_fields(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _fake_limit_order(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(
            "nautilus_quants.execution.post_limit.spawn.LimitOrder",
            _fake_limit_order,
        )

        state = _make_state()
        primary = _make_primary()
        factory = ChildOrderFactory(
            clock=SimpleNamespace(timestamp_ns=lambda: 123456), exec_algorithm_id="PostLimit"
        )

        order = factory.create_limit(
            primary=primary,
            state=state,
            quantity=Quantity.from_str("1.25"),
            price=49999.5,
            time_in_force=TimeInForce.GTC,
            post_only=True,
            reduce_only=False,
        )

        assert order.client_order_id == ClientOrderId("primary001E1")
        assert "-" not in order.client_order_id.value
        assert captured["exec_algorithm_id"] == "PostLimit"
        assert captured["exec_spawn_id"] == primary.client_order_id
        assert captured["contingency_type"] == "OCO"
        assert captured["order_list_id"] == "LIST001"
        assert captured["linked_order_ids"] == ["LINK1", "LINK2"]
        assert captured["parent_order_id"] == "PARENT001"
        assert captured["tags"] == ["OPEN", "POST_LIMIT"]
        assert captured["quote_quantity"] is False
        assert captured["post_only"] is True

    def test_register_child_sets_active_order_and_sequence(self) -> None:
        state = _make_state()
        factory = ChildOrderFactory(
            clock=SimpleNamespace(timestamp_ns=lambda: 1), exec_algorithm_id="PostLimit"
        )
        order = SimpleNamespace(
            client_order_id=ClientOrderId("primary001E2"),
            quantity=Quantity.from_str("1.0"),
        )
        state.spawn_sequence = 2

        factory.register_child(state, order, SpawnKind.LIMIT)

        assert state.active_order_id == ClientOrderId("primary001E2")
        assert state.active_order_kind == SpawnKind.LIMIT
        assert state.active_reserved_quantity == Quantity.from_str("1.0")
        assert state.limit_orders_submitted == 1


class TestPrimaryMirror:
    def test_reduce_and_restore_primary_quantity(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "nautilus_quants.execution.post_limit.spawn.OrderUpdated",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        cache = MagicMock()
        clock = SimpleNamespace(timestamp_ns=lambda: 999)
        logger = MagicMock()
        mirror = PrimaryMirror(cache=cache, clock=clock, logger=logger)

        primary = _make_primary()
        primary.apply.side_effect = lambda event: setattr(primary, "quantity", event.quantity)
        child = SimpleNamespace(leaves_qty=Quantity.from_str("0.40"))

        mirrored_reserved = mirror.reduce_primary(primary, Quantity.from_str("1.25"))

        reduced_event = primary.apply.call_args.args[0]
        assert mirrored_reserved == Quantity.from_str("1.25")
        assert reduced_event.quantity == Quantity.from_str("0.75")
        assert cache.update_order.call_count == 1

        mirror.restore_primary(
            primary,
            child,
            reserved_quantity=Quantity.from_str("0.60"),
        )

        restored_event = primary.apply.call_args.args[0]
        assert restored_event.quantity == Quantity.from_str("1.15")
        assert cache.update_order.call_count == 2

    def test_full_delegation_skips_zero_quantity_update(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "nautilus_quants.execution.post_limit.spawn.OrderUpdated",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        cache = MagicMock()
        clock = SimpleNamespace(timestamp_ns=lambda: 999)
        logger = MagicMock()
        mirror = PrimaryMirror(cache=cache, clock=clock, logger=logger)

        primary = _make_primary()
        child = SimpleNamespace(leaves_qty=Quantity.from_str("2.00"))

        mirrored_reserved = mirror.reduce_primary(primary, Quantity.from_str("2.00"))

        assert mirrored_reserved == Quantity.from_str("0.00")
        primary.apply.assert_not_called()
        cache.update_order.assert_not_called()
        logger.debug.assert_called_once()

        mirror.restore_primary(primary, child, reserved_quantity=mirrored_reserved)

        primary.apply.assert_not_called()
        cache.update_order.assert_not_called()

    def test_reduce_above_leaves_warns_and_reserves_best_effort(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "nautilus_quants.execution.post_limit.spawn.OrderUpdated",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        cache = MagicMock()
        clock = SimpleNamespace(timestamp_ns=lambda: 999)
        logger = MagicMock()
        mirror = PrimaryMirror(cache=cache, clock=clock, logger=logger)

        primary = _make_primary()
        primary.leaves_qty = Quantity.from_str("0.80")
        primary.apply.side_effect = lambda event: setattr(primary, "quantity", event.quantity)

        mirrored_reserved = mirror.reduce_primary(primary, Quantity.from_str("1.25"))

        assert mirrored_reserved == Quantity.from_str("0.80")
        reduced_event = primary.apply.call_args.args[0]
        assert reduced_event.quantity == Quantity.from_str("1.20")
        logger.warning.assert_called_once()
        assert "above leaves, continuing best-effort" in logger.warning.call_args.args[0]
