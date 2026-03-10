# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for EventTimePendingExecutionMixin."""

from __future__ import annotations

from nautilus_quants.common.event_time_pending_execution import (
    EventTimePendingExecutionMixin,
)


class _LogStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)

    def debug(self, msg: str) -> None:
        self.messages.append(msg)

    def warning(self, msg: str) -> None:
        self.messages.append(msg)


class _DummyPendingExecution(EventTimePendingExecutionMixin[dict[str, float]]):
    def __init__(self, max_pending_wait_timestamps: int = 2) -> None:
        self.log = _LogStub()
        self.ready_calls: list[tuple[int, dict[str, float], dict[str, float]]] = []
        self._init_event_time_pending(
            max_timestamps=6,
            max_pending_wait_timestamps=max_pending_wait_timestamps,
        )

    def _required_instruments(self, payload: dict[str, float]) -> list[str]:
        return list(payload.keys())

    def _on_pending_ready(
        self,
        signal_ts: int,
        payload: dict[str, float],
        execution_prices: dict[str, float],
    ) -> None:
        self.ready_calls.append((signal_ts, dict(payload), dict(execution_prices)))


def test_pending_executes_in_ascending_signal_ts_order() -> None:
    strategy = _DummyPendingExecution()
    strategy._enqueue_pending(200, {"A.BINANCE": 1.0})
    strategy._enqueue_pending(100, {"B.BINANCE": 2.0})

    strategy._record_close_and_try_execute(100, "B.BINANCE", 10.0)
    strategy._record_close_and_try_execute(200, "A.BINANCE", 20.0)

    assert [call[0] for call in strategy.ready_calls] == [100, 200]


def test_no_cross_timestamp_fallback_for_missing_prices() -> None:
    strategy = _DummyPendingExecution()
    strategy._enqueue_pending(100, {"A.BINANCE": 1.0, "B.BINANCE": 2.0})

    strategy._record_close_and_try_execute(100, "A.BINANCE", 10.0)
    strategy._record_close_and_try_execute(200, "B.BINANCE", 20.0)

    assert strategy.ready_calls == []
    assert 100 in strategy._pending_by_ts


def test_missing_prices_timeout_drops_pending() -> None:
    strategy = _DummyPendingExecution(max_pending_wait_timestamps=2)
    strategy._enqueue_pending(100, {"A.BINANCE": 1.0, "B.BINANCE": 2.0})

    strategy._record_close_and_try_execute(100, "A.BINANCE", 10.0)
    strategy._record_close_and_try_execute(101, "X.BINANCE", 1.0)
    strategy._record_close_and_try_execute(102, "X.BINANCE", 1.0)

    assert strategy.ready_calls == []
    assert 100 not in strategy._pending_by_ts


def test_ready_pending_is_consumed_only_once() -> None:
    strategy = _DummyPendingExecution()
    strategy._enqueue_pending(100, {"A.BINANCE": 1.0})

    strategy._record_close_and_try_execute(100, "A.BINANCE", 10.0)
    strategy._try_execute_pending(100)

    assert len(strategy.ready_calls) == 1
    assert strategy._pending_by_ts == {}
