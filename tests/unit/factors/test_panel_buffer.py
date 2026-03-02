# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PanelBuffer — rolling panel data accumulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.engine.panel_buffer import PanelBuffer


class TestPanelBufferBasic:
    """Basic append / flush / to_panel tests."""

    def test_empty_panel(self) -> None:
        buf = PanelBuffer(max_history=100)
        panel = buf.to_panel()
        assert "close" in panel
        assert panel["close"].empty

    def test_single_bar(self) -> None:
        buf = PanelBuffer(max_history=100)
        buf.append("BTC", 1, {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000})
        buf.flush_timestamp(1)
        panel = buf.to_panel()

        assert panel["close"].shape == (1, 1)
        assert panel["close"].iloc[0]["BTC"] == 102
        assert panel["volume"].iloc[0]["BTC"] == 1000

    def test_two_instruments_same_timestamp(self) -> None:
        buf = PanelBuffer(max_history=100)
        buf.append("BTC", 1, {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000})
        buf.append("ETH", 1, {"open": 50, "high": 55, "low": 48, "close": 52, "volume": 2000})
        buf.flush_timestamp(1)
        panel = buf.to_panel()

        assert panel["close"].shape == (1, 2)
        assert panel["close"].iloc[0]["BTC"] == 102
        assert panel["close"].iloc[0]["ETH"] == 52

    def test_multiple_timestamps(self) -> None:
        buf = PanelBuffer(max_history=100)
        for ts in range(1, 4):
            buf.append("BTC", ts, {"open": 100 + ts, "high": 105 + ts, "low": 95 + ts, "close": 102 + ts, "volume": 1000 * ts})
            buf.append("ETH", ts, {"open": 50 + ts, "high": 55 + ts, "low": 48 + ts, "close": 52 + ts, "volume": 2000 * ts})
            buf.flush_timestamp(ts)

        panel = buf.to_panel()
        assert panel["close"].shape == (3, 2)
        assert panel["close"].iloc[2]["BTC"] == 105  # 102 + 3

    def test_instruments_property(self) -> None:
        buf = PanelBuffer(max_history=100)
        buf.append("BTC", 1, {"open": 100, "close": 102, "high": 105, "low": 95, "volume": 1000})
        buf.append("ETH", 1, {"open": 50, "close": 52, "high": 55, "low": 48, "volume": 2000})
        buf.flush_timestamp(1)
        assert buf.instruments == ["BTC", "ETH"]

    def test_timestamps_property(self) -> None:
        buf = PanelBuffer(max_history=100)
        for ts in [10, 20, 30]:
            buf.append("BTC", ts, {"open": 100, "close": 102, "high": 105, "low": 95, "volume": 1000})
            buf.flush_timestamp(ts)
        assert buf.timestamps == [10, 20, 30]

    def test_n_timestamps(self) -> None:
        buf = PanelBuffer(max_history=100)
        assert buf.n_timestamps == 0
        buf.append("BTC", 1, {"open": 100, "close": 102, "high": 105, "low": 95, "volume": 1000})
        buf.flush_timestamp(1)
        assert buf.n_timestamps == 1


class TestPanelBufferRolling:
    """Test max_history rolling window."""

    def test_rolling_window_trims(self) -> None:
        buf = PanelBuffer(max_history=3)
        for ts in range(1, 6):
            buf.append("BTC", ts, {"open": ts, "high": ts, "low": ts, "close": ts, "volume": ts})
            buf.flush_timestamp(ts)

        panel = buf.to_panel()
        assert panel["close"].shape == (3, 1)
        # Only last 3 timestamps remain
        assert list(panel["close"].index) == [3, 4, 5]

    def test_rolling_window_max_history_1(self) -> None:
        buf = PanelBuffer(max_history=1)
        for ts in range(1, 4):
            buf.append("BTC", ts, {"open": ts, "close": ts, "high": ts, "low": ts, "volume": ts})
            buf.flush_timestamp(ts)

        panel = buf.to_panel()
        assert panel["close"].shape == (1, 1)
        assert panel["close"].iloc[0]["BTC"] == 3


class TestPanelBufferEdgeCases:
    """Edge cases and missing data."""

    def test_missing_field_becomes_nan(self) -> None:
        buf = PanelBuffer(max_history=100)
        buf.append("BTC", 1, {"open": 100, "close": 102})  # missing high, low, volume
        buf.flush_timestamp(1)
        panel = buf.to_panel()
        assert np.isnan(panel["volume"].iloc[0]["BTC"])

    def test_missing_instrument_at_timestamp_becomes_nan(self) -> None:
        buf = PanelBuffer(max_history=100)
        buf.append("BTC", 1, {"open": 100, "close": 102, "high": 105, "low": 95, "volume": 1000})
        buf.append("ETH", 1, {"open": 50, "close": 52, "high": 55, "low": 48, "volume": 2000})
        buf.flush_timestamp(1)
        # Only BTC at ts=2
        buf.append("BTC", 2, {"open": 101, "close": 103, "high": 106, "low": 96, "volume": 1100})
        buf.flush_timestamp(2)

        panel = buf.to_panel()
        assert panel["close"].shape == (2, 2)
        assert panel["close"].iloc[1]["BTC"] == 103
        assert np.isnan(panel["close"].iloc[1]["ETH"])

    def test_flush_nonexistent_timestamp_noop(self) -> None:
        buf = PanelBuffer(max_history=100)
        buf.flush_timestamp(999)  # no staged data for ts=999
        assert buf.n_timestamps == 0

    def test_reset(self) -> None:
        buf = PanelBuffer(max_history=100)
        buf.append("BTC", 1, {"open": 100, "close": 102, "high": 105, "low": 95, "volume": 1000})
        buf.flush_timestamp(1)
        buf.reset()
        assert buf.n_timestamps == 0
        assert buf.instruments == []
        assert buf.to_panel()["close"].empty

    def test_extra_fields(self) -> None:
        buf = PanelBuffer(max_history=100, extra_fields=("quote_volume",))
        buf.append("BTC", 1, {"open": 100, "close": 102, "high": 105, "low": 95, "volume": 1000, "quote_volume": 50000})
        buf.flush_timestamp(1)
        panel = buf.to_panel()
        assert "quote_volume" in panel
        assert panel["quote_volume"].iloc[0]["BTC"] == 50000
