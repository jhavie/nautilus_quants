# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
ExposureManager - Risk exposure pool controlling close/open execution ordering.

Analogous to a thread pool: tasks are submitted, the pool controls concurrency
and ordering. Each primary completion releases one secondary (one-to-one).
When all primaries complete, remaining secondaries are released in bulk.

Two policies:
- CLOSE_FIRST: execute all closes first, release opens as closes complete
- OPEN_FIRST: execute all opens first, release closes as opens complete
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ExposurePolicy(Enum):
    """Execution ordering policy for the ExposureManager."""

    CLOSE_FIRST = "close_first"
    OPEN_FIRST = "open_first"


class ExposureManager:
    """
    Risk exposure management pool.

    Controls close/open execution ordering with one-to-one release semantics.

    Parameters
    ----------
    policy : ExposurePolicy, default CLOSE_FIRST
        Execution ordering policy.
    """

    def __init__(self, policy: ExposurePolicy = ExposurePolicy.CLOSE_FIRST) -> None:
        self._policy = policy
        self._pending_primary: set[str] = set()
        self._queued_secondary: list[dict[str, Any]] = []
        self._stopping: bool = False

    def submit_plan(
        self,
        closes: list[dict[str, Any]],
        opens: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Submit a rebalance plan.

        Returns
        -------
        tuple[list[dict], list[dict]]
            (immediate_closes, immediate_opens) to execute now.
            Remaining orders are queued and released via on_close_filled/on_open_filled.
        """
        if self._policy == ExposurePolicy.CLOSE_FIRST:
            primary, secondary = closes, opens
        else:
            primary, secondary = opens, closes

        if not primary:
            if self._policy == ExposurePolicy.CLOSE_FIRST:
                return [], list(secondary)
            return list(secondary), []

        self._pending_primary = {o["instrument_id"] for o in primary}
        self._queued_secondary = list(secondary)

        if self._policy == ExposurePolicy.CLOSE_FIRST:
            return list(primary), []
        return [], list(primary)

    def on_close_filled(self, instrument_id: str) -> list[dict[str, Any]]:
        """
        Notify that a close position has been confirmed.

        CLOSE_FIRST: close is primary → releases one secondary.
        OPEN_FIRST: close is secondary → no release.
        """
        if self._stopping:
            return []
        if self._policy == ExposurePolicy.CLOSE_FIRST:
            return self._on_primary_filled(instrument_id)
        return []

    def on_open_filled(self, instrument_id: str) -> list[dict[str, Any]]:
        """
        Notify that an open position has been confirmed.

        OPEN_FIRST: open is primary → releases one secondary.
        CLOSE_FIRST: open is secondary → no release.
        """
        if self._stopping:
            return []
        if self._policy == ExposurePolicy.OPEN_FIRST:
            return self._on_primary_filled(instrument_id)
        return []

    def _on_primary_filled(self, instrument_id: str) -> list[dict[str, Any]]:
        """Primary filled → release one secondary, or all if primaries exhausted."""
        self._pending_primary.discard(instrument_id)

        if not self._pending_primary:
            released = self._queued_secondary
            self._queued_secondary = []
            return released

        if self._queued_secondary:
            return [self._queued_secondary.pop(0)]
        return []

    def on_stop(self) -> None:
        """Discard all pending and queued orders."""
        self._stopping = True
        self._pending_primary.clear()
        self._queued_secondary.clear()

    @property
    def has_pending(self) -> bool:
        """Whether there are pending or queued orders."""
        return bool(self._pending_primary) or bool(self._queued_secondary)

    @property
    def state_summary(self) -> str:
        """Human-readable state for logging."""
        if self._stopping:
            return "STOPPING"
        if not self._pending_primary and not self._queued_secondary:
            return "IDLE"
        return (
            f"PROCESSING(pending={len(self._pending_primary)}, "
            f"queued={len(self._queued_secondary)})"
        )
