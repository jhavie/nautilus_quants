# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Event-time aligned close price snapshots for deterministic signal execution."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable


class EventTimePriceBook:
    """Stores per-instrument closes keyed by bar ``ts_event``."""

    def __init__(self, max_timestamps: int = 6) -> None:
        if max_timestamps < 1:
            msg = "max_timestamps must be >= 1"
            raise ValueError(msg)

        self._max_timestamps = max_timestamps
        self._closes_by_ts: OrderedDict[int, dict[str, float]] = OrderedDict()

    def record_close(self, ts_event: int, instrument_id: str, close: float) -> None:
        """Record one instrument close for a specific event timestamp."""
        closes = self._closes_by_ts.get(ts_event)
        if closes is None:
            closes = {}
            self._closes_by_ts[ts_event] = closes
        closes[instrument_id] = close

        while len(self._closes_by_ts) > self._max_timestamps:
            self._closes_by_ts.popitem(last=False)

    def build_snapshot(
        self,
        ts_event: int,
        required_instruments: Iterable[str],
    ) -> tuple[dict[str, float], list[str]]:
        """
        Build a deterministic price snapshot for one timestamp.

        Returns:
        - ``price_map`` for instruments that exist exactly at ``ts_event``
        - ``missing`` instruments without a close at ``ts_event``
        """
        required = sorted(set(required_instruments))
        closes = self._closes_by_ts.get(ts_event)
        if closes is None:
            return {}, required

        price_map: dict[str, float] = {}
        missing: list[str] = []
        for instrument_id in required:
            close = closes.get(instrument_id)
            if close is None:
                missing.append(instrument_id)
            else:
                price_map[instrument_id] = close

        return price_map, missing

    def get_latest_closes(self) -> dict[str, float]:
        """Return the most recent close price for each instrument across all stored timestamps."""
        result: dict[str, float] = {}
        for closes in self._closes_by_ts.values():
            result.update(closes)  # later timestamps overwrite earlier
        return result

    def newer_timestamp_count(self, ts_event: int) -> int:
        """Return how many cached timestamps are newer than ``ts_event``."""
        return sum(1 for cached_ts in self._closes_by_ts if cached_ts > ts_event)

