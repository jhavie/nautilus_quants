# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Reusable event-time pending execution flow for signal-close semantics."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Generic, TypeVar

from nautilus_quants.common.event_time_price_book import EventTimePriceBook

TPayload = TypeVar("TPayload")


class EventTimePendingExecutionMixin(Generic[TPayload]):
    """
    Shared pending queue + event-time snapshot execution.

    The mixin guarantees:
    - execution uses only ``signal_ts`` close snapshots
    - no cross-timestamp fallback
    - deterministic pending traversal by ascending ``signal_ts``
    """

    _price_book: EventTimePriceBook
    _pending_by_ts: dict[int, TPayload]
    _max_pending_wait_timestamps: int

    def _init_event_time_pending(
        self,
        max_timestamps: int = 6,
        max_pending_wait_timestamps: int = 2,
    ) -> None:
        self._price_book = EventTimePriceBook(max_timestamps=max_timestamps)
        self._pending_by_ts = {}
        self._max_pending_wait_timestamps = max_pending_wait_timestamps

    def _record_close_and_try_execute(
        self,
        ts_event: int,
        instrument_id: str,
        close: float,
    ) -> None:
        self._price_book.record_close(ts_event, instrument_id, close)
        self._try_execute_pending(ts_event)

    def _enqueue_pending(self, signal_ts: int, payload: TPayload) -> None:
        self._pending_by_ts[signal_ts] = payload
        required = self._required_sorted_instruments(payload)
        self.log.info(
            f"Queued pending: signal_ts={signal_ts}, required_count={len(required)}"
        )
        self._try_execute_pending(signal_ts)

    def _try_execute_pending(self, current_ts: int) -> None:
        if not self._pending_by_ts:
            return

        for signal_ts in sorted(list(self._pending_by_ts)):
            if signal_ts > current_ts:
                continue

            payload = self._pending_by_ts[signal_ts]
            required = self._required_sorted_instruments(payload)
            execution_prices, missing = self._price_book.build_snapshot(signal_ts, required)
            if missing:
                newer_count = self._price_book.newer_timestamp_count(signal_ts)
                if newer_count >= self._max_pending_wait_timestamps:
                    self.log.warning(
                        f"Drop pending: signal_ts={signal_ts}, "
                        f"required_count={len(required)}, ready_count={len(execution_prices)}, "
                        f"missing_instruments={missing[:10]}, reason=missing_prices_timeout"
                    )
                    del self._pending_by_ts[signal_ts]
                else:
                    self.log.debug(
                        f"Wait pending: signal_ts={signal_ts}, "
                        f"required_count={len(required)}, ready_count={len(execution_prices)}, "
                        f"missing_instruments={missing[:10]}, newer_timestamps={newer_count}"
                    )
                continue

            import hashlib
            prices_str = str(sorted(execution_prices.items()))
            prices_hash = hashlib.md5(prices_str.encode()).hexdigest()[:8]
            self.log.info(
                f"Execute pending: signal_ts={signal_ts}, "
                f"required_count={len(required)}, ready_count={len(execution_prices)}, "
                f"prices_hash={prices_hash}"
            )
            # Fail-fast by design: exceptions propagate to caller.
            self._on_pending_ready(signal_ts, payload, execution_prices)
            del self._pending_by_ts[signal_ts]

    def _required_sorted_instruments(self, payload: TPayload) -> list[str]:
        return sorted(set(self._required_instruments(payload)))

    @abstractmethod
    def _required_instruments(self, payload: TPayload) -> Iterable[str]:
        """Return required instruments for execution readiness."""

    @abstractmethod
    def _on_pending_ready(
        self,
        signal_ts: int,
        payload: TPayload,
        execution_prices: dict[str, float],
    ) -> None:
        """Handle execution when same-ts snapshot is fully ready."""
