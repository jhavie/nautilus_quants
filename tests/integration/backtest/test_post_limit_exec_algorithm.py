# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Backtest-oriented integration tests for PostLimitExecAlgorithm state recovery."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.algorithm import PostLimitExecAlgorithm
from nautilus_quants.execution.post_limit.state import (
    OrderExecutionState,
    OrderState,
    SpawnKind,
)


class _FakeClock:
    def __init__(self) -> None:
        self._timer_names: set[str] = set()
        self.set_timer = MagicMock(side_effect=self._set_timer)
        self.cancel_timer = MagicMock(side_effect=self._cancel_timer)
        self.cancel_timers = MagicMock(side_effect=self._cancel_all)
        self._ts = 1_000

    @property
    def timer_names(self) -> set[str]:
        return self._timer_names

    def timestamp_ns(self) -> int:
        self._ts += 1
        return self._ts

    def _set_timer(self, *, name, interval, callback) -> None:
        self._timer_names.add(name)

    def _cancel_timer(self, name) -> None:
        self._timer_names.discard(name)

    def _cancel_all(self) -> None:
        self._timer_names.clear()


def _patch_algo_environment(algo: PostLimitExecAlgorithm):
    stack = ExitStack()
    clock = _FakeClock()
    cache = MagicMock()
    cache.add = MagicMock()
    cache.order = MagicMock(return_value=None)
    log = MagicMock()
    stack.enter_context(
        patch.object(PostLimitExecAlgorithm, "clock", new_callable=PropertyMock, return_value=clock),
    )
    stack.enter_context(
        patch.object(PostLimitExecAlgorithm, "cache", new_callable=PropertyMock, return_value=cache),
    )
    stack.enter_context(
        patch.object(PostLimitExecAlgorithm, "log", new_callable=PropertyMock, return_value=log),
    )
    return stack, cache, clock


def _make_working_state() -> OrderExecutionState:
    state = OrderExecutionState(
        primary_order_id=ClientOrderId("primary001"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        side=OrderSide.BUY,
        total_quantity=Quantity.from_str("1.0"),
        anchor_px=50000.0,
        state=OrderState.WORKING_LIMIT,
        timer_name="PostLimit-primary001",
        created_ns=123,
        chase_count=1,
        spawn_sequence=3,
    )
    state.activate_order(
        client_order_id=ClientOrderId("primary001E3"),
        kind=SpawnKind.LIMIT,
        reserved_quantity=Quantity.from_str("0.4"),
        accepted=True,
    )
    return state


class TestPostLimitExecAlgorithmRecovery:
    def test_save_load_rebuilds_runtime_indexes_and_rearms_working_timer(self) -> None:
        writer = PostLimitExecAlgorithm()
        writer_stack, _, _, = _patch_algo_environment(writer)
        with writer_stack:
            state = _make_working_state()
            writer._states[state.primary_order_id] = state

            saved = writer.on_save()

        reader = PostLimitExecAlgorithm()
        reader_stack, _, reader_clock = _patch_algo_environment(reader)
        with reader_stack:
            reader.subscribe_quote_ticks = MagicMock()  # type: ignore[method-assign]

            reader.on_load(saved)

            assert reader._states[ClientOrderId("primary001")].spawn_sequence == 3
            assert reader._active_child_to_primary == {
                ClientOrderId("primary001E3"): ClientOrderId("primary001"),
            }
            assert reader._timer_to_primary == {}

            reader.on_start()

            assert reader._timer_to_primary == {
                "PostLimit-primary001": ClientOrderId("primary001"),
            }
            reader_clock.set_timer.assert_called_once()
            reader.subscribe_quote_ticks.assert_called_once_with(
                InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            )

    def test_reduce_only_completed_session_requests_single_residual_sweep(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _ = _patch_algo_environment(algo)
        with stack:
            completed = OrderExecutionState(
                primary_order_id=ClientOrderId("primary001"),
                instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                side=OrderSide.SELL,
                total_quantity=Quantity.from_str("1.0"),
                anchor_px=50000.0,
                reduce_only=True,
                state=OrderState.COMPLETED,
                timer_name="PostLimit-primary001",
                created_ns=123,
                residual_sweep_pending=True,
            )
            another = OrderExecutionState(
                primary_order_id=ClientOrderId("primary002"),
                instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                side=OrderSide.SELL,
                total_quantity=Quantity.from_str("1.0"),
                anchor_px=50000.0,
                reduce_only=True,
                state=OrderState.COMPLETED,
                timer_name="PostLimit-primary002",
                created_ns=124,
                residual_sweep_pending=True,
            )
            algo._states = {
                completed.primary_order_id: completed,
                another.primary_order_id: another,
            }
            cache.instrument.return_value = SimpleNamespace(min_notional=5.0)
            cache.positions_open.return_value = [SimpleNamespace(is_closed=False, quantity=Quantity.from_str("0.0001"))]
            cache.quote_tick.return_value = SimpleNamespace(bid_price=10000.0, ask_price=10000.0)
            algo._submit_market_child = MagicMock(  # type: ignore[method-assign]
                side_effect=lambda state, **kwargs: state.activate_order(
                    client_order_id=ClientOrderId(f"{state.primary_order_id.value}E1"),
                    kind=SpawnKind.SWEEP,
                    reserved_quantity=Quantity.from_str("0.0001"),
                ),
            )

            algo.on_position_changed(
                SimpleNamespace(instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE")),
            )

            assert algo._submit_market_child.call_count == 1
