# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Buffer — Rolling panel data accumulator.

Accumulates per-instrument bar data into rolling DataFrame panels
(rows = timestamps, columns = instruments) for cross-sectional + time-series
factor evaluation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd


class Buffer:
    """Rolling panel data accumulator.

    Maintains a sliding window of ``max_history`` timestamps.  Each OHLCV
    field is stored as a ``pd.DataFrame[T x N]`` where rows are timestamps
    and columns are instrument IDs.

    Usage::

        buf = Buffer(max_history=500)

        # For each bar received:
        buf.append("AAPL", ts=1, bar_data={"open": 150, "high": 155, ...})
        buf.append("GOOGL", ts=1, bar_data={"open": 2800, ...})

        # When all instruments have reported for this timestamp:
        buf.flush_timestamp(ts=1)

        # Retrieve panels for evaluation:
        panels = buf.to_panel()  # {"open": DataFrame, "close": DataFrame, ...}
    """

    FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

    def __init__(self, max_history: int = 500, extra_fields: tuple[str, ...] = ()) -> None:
        self._max_history = max_history
        self._extra_fields = extra_fields
        self._all_fields = self.FIELDS + extra_fields

        # Staging area: ts -> {instrument_id -> {field -> value}}
        self._staging: dict[int, dict[str, dict[str, float]]] = {}

        # Committed data: field -> OrderedDict[ts -> {instrument_id -> value}]
        self._data: dict[str, OrderedDict[int, dict[str, float]]] = {
            f: OrderedDict() for f in self._all_fields
        }

        # Track all known instruments (in insertion order)
        self._instruments: list[str] = []
        self._instrument_set: set[str] = set()

        # Track flushed timestamps
        self._timestamps: list[int] = []

    def append(self, instrument_id: str, ts: int, bar_data: dict[str, float]) -> None:
        """Append a single bar to the staging area.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier (e.g., "AAPL", "BTC/USDT").
        ts : int
            Timestamp (nanoseconds).
        bar_data : dict
            Bar data with keys like "open", "high", "low", "close", "volume".
        """
        if instrument_id not in self._instrument_set:
            self._instruments.append(instrument_id)
            self._instrument_set.add(instrument_id)

        if ts not in self._staging:
            self._staging[ts] = {}
        self._staging[ts][instrument_id] = bar_data

    def flush_timestamp(self, ts: int) -> None:
        """Commit staged data for a timestamp to the rolling panel.

        Should be called once all instruments have reported for ``ts``.
        The staging area for ``ts`` is consumed and cleared.
        """
        staged = self._staging.pop(ts, None)
        if staged is None:
            return

        # Commit each field
        for field_name in self._all_fields:
            row: dict[str, float] = {}
            for inst_id, bar in staged.items():
                row[inst_id] = bar.get(field_name, float("nan"))
            self._data[field_name][ts] = row

            # Trim to max_history (0 = unlimited)
            if self._max_history > 0:
                while len(self._data[field_name]) > self._max_history:
                    self._data[field_name].popitem(last=False)

        # Track timestamps
        self._timestamps.append(ts)
        if self._max_history > 0 and len(self._timestamps) > self._max_history:
            self._timestamps = self._timestamps[-self._max_history:]

    def to_panel(self) -> dict[str, pd.DataFrame]:
        """Build DataFrame panels from committed data.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of field name -> DataFrame[T x N] where T = timestamps
            and N = instruments.  Missing values are NaN.
        """
        instruments = sorted(self._instruments)
        panels: dict[str, pd.DataFrame] = {}

        for field_name in self._all_fields:
            rows = self._data[field_name]
            if not rows:
                panels[field_name] = pd.DataFrame(
                    columns=instruments, dtype=float,
                )
                continue

            ts_list = list(rows.keys())
            matrix = np.full((len(ts_list), len(instruments)), np.nan)

            for i, ts in enumerate(ts_list):
                row = rows[ts]
                for j, inst in enumerate(instruments):
                    if inst in row:
                        matrix[i, j] = row[inst]

            panels[field_name] = pd.DataFrame(
                matrix,
                index=pd.Index(ts_list, name="timestamp"),
                columns=instruments,
            )

        return panels

    @property
    def instruments(self) -> list[str]:
        """Return list of known instrument IDs in insertion order."""
        return list(self._instruments)

    @property
    def timestamps(self) -> list[int]:
        """Return list of flushed timestamps."""
        return list(self._timestamps)

    @property
    def n_timestamps(self) -> int:
        """Return number of committed timestamps."""
        first_field = next(iter(self._data.values()), None)
        return len(first_field) if first_field else 0

    @property
    def n_instruments(self) -> int:
        """Return number of known instruments."""
        return len(self._instruments)

    def inject_staged_field(
        self,
        ts: int,
        field_name: str,
        source_instrument: str,
        source_field: str = "close",
    ) -> None:
        """Copy one instrument's staged value to a new field for all instruments.

        Must be called AFTER all instruments have reported for ``ts``
        (staging complete) but BEFORE ``flush_timestamp(ts)``.  This
        ensures broadcast values are correct and consistent across all
        instruments at the same timestamp.

        Parameters
        ----------
        ts : int
            Timestamp (nanoseconds) in the staging area.
        field_name : str
            Target field name (e.g. "btc_close").
        source_instrument : str
            Instrument whose value to broadcast.
        source_field : str
            Field to read from the source instrument (default "close").
        """
        staged = self._staging.get(ts)
        if not staged:
            return
        source_data = staged.get(source_instrument)
        if source_data is None:
            return
        value = source_data.get(source_field)
        if value is None:
            return
        for inst_data in staged.values():
            inst_data[field_name] = value

    def reset(self) -> None:
        """Clear all data."""
        self._staging.clear()
        for od in self._data.values():
            od.clear()
        self._instruments.clear()
        self._instrument_set.clear()
        self._timestamps.clear()
