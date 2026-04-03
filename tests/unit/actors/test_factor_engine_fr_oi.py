# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for FactorEngineActor FR/OI injection logic.

These tests exercise the standalone helper functions and the data injection
code paths in isolation, without requiring Nautilus infrastructure
(MessageBus, Clock, etc.).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nautilus_quants.actors.factor_engine import (
    _detect_extra_bar_fields,
    _extract_bar_data,
)


# ---------------------------------------------------------------------------
# Helpers to build mock Bar objects
# ---------------------------------------------------------------------------


def _make_mock_bar(
    *,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    volume: float = 1000.0,
    extra_dict: dict | None = None,
) -> MagicMock:
    """Create a mock Bar with OHLCV getset descriptors and optional __dict__.

    MagicMock stores our attribute assignments in its ``__dict__``, which is
    what ``_extract_bar_data`` iterates.  The OHLCV keys are already extracted
    via explicit ``bar.open`` etc., so duplicates in ``__dict__`` are filtered
    out by the ``k not in data`` guard.  Non-numeric mock internals
    (``method_calls``, ``call_args_list``, ...) are silently skipped by the
    ``float(v)`` try/except inside ``_extract_bar_data``.
    """
    bar = MagicMock()
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    if extra_dict is not None:
        bar.__dict__.update(extra_dict)
    return bar


# ---------------------------------------------------------------------------
# _extract_bar_data
# ---------------------------------------------------------------------------


class TestExtractBarDataStandard:
    """Verify _extract_bar_data extracts OHLCV correctly."""

    def test_standard_ohlcv(self) -> None:
        """Standard Bar should yield open/high/low/close/volume."""
        bar = _make_mock_bar(
            open_=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000.0,
        )

        data = _extract_bar_data(bar)

        assert data["open"] == pytest.approx(100.0)
        assert data["high"] == pytest.approx(110.0)
        assert data["low"] == pytest.approx(90.0)
        assert data["close"] == pytest.approx(105.0)
        assert data["volume"] == pytest.approx(1000.0)
        assert len(data) == 5

    def test_extra_fields_from_dict(self) -> None:
        """BinanceBar-style extra fields in __dict__ are included."""
        bar = _make_mock_bar(
            extra_dict={
                "quote_volume": 5000000.0,
                "count": 12345,
                "taker_buy_base_volume": 600.0,
            },
        )

        data = _extract_bar_data(bar)

        # OHLCV + 3 extra fields
        assert "quote_volume" in data
        assert data["quote_volume"] == pytest.approx(5000000.0)
        assert data["count"] == pytest.approx(12345.0)
        assert data["taker_buy_base_volume"] == pytest.approx(600.0)
        assert len(data) == 8  # 5 OHLCV + 3 extra

    def test_private_keys_excluded(self) -> None:
        """Keys starting with underscore in __dict__ are excluded."""
        bar = _make_mock_bar(
            extra_dict={
                "_internal": 999.0,
                "quote_volume": 5000.0,
            },
        )

        data = _extract_bar_data(bar)

        assert "_internal" not in data
        assert "quote_volume" in data

    def test_non_numeric_extra_fields_skipped(self) -> None:
        """Non-numeric extra fields are silently skipped."""
        bar = _make_mock_bar(
            extra_dict={
                "quote_volume": 5000.0,
                "some_string": "not_a_number",
            },
        )

        data = _extract_bar_data(bar)

        assert "quote_volume" in data
        assert "some_string" not in data


# ---------------------------------------------------------------------------
# Funding rate injection in on_bar
# ---------------------------------------------------------------------------


class TestFundingRateInjection:
    """Simulate the FR injection logic from on_bar."""

    def test_funding_rate_injected_into_bar_data(self) -> None:
        """When _latest_funding_rates has a rate for the instrument,
        bar_data should contain the funding_rate field.
        """
        bar = _make_mock_bar()
        bar_data = _extract_bar_data(bar)

        # Simulate the injection logic from FactorEngineActor.on_bar
        latest_funding_rates = {"BTCUSDT.BINANCE": 0.0001}
        instrument_id = "BTCUSDT.BINANCE"

        if instrument_id in latest_funding_rates:
            bar_data["funding_rate"] = latest_funding_rates[instrument_id]

        assert "funding_rate" in bar_data
        assert bar_data["funding_rate"] == pytest.approx(0.0001)

    def test_funding_rate_not_injected_when_missing(self) -> None:
        """When no funding rate is cached, bar_data should not have
        the funding_rate field.
        """
        bar = _make_mock_bar()
        bar_data = _extract_bar_data(bar)

        latest_funding_rates: dict[str, float] = {}
        instrument_id = "BTCUSDT.BINANCE"

        if instrument_id in latest_funding_rates:
            bar_data["funding_rate"] = latest_funding_rates[instrument_id]

        assert "funding_rate" not in bar_data

    def test_funding_rate_forward_fill(self) -> None:
        """Funding rate persists across multiple bars (forward-fill).

        The on_funding_rate handler caches the latest rate, and on_bar
        always reads from the cache, so the same rate applies to every
        bar until a new FundingRateUpdate arrives.
        """
        latest_funding_rates: dict[str, float] = {}
        instrument_id = "BTCUSDT.BINANCE"

        # Simulate on_funding_rate: cache rate
        latest_funding_rates[instrument_id] = 0.0002

        # Simulate on_bar for 3 consecutive bars
        results = []
        for _ in range(3):
            bar_data = _extract_bar_data(_make_mock_bar())
            if instrument_id in latest_funding_rates:
                bar_data["funding_rate"] = latest_funding_rates[instrument_id]
            results.append(bar_data.get("funding_rate"))

        assert all(r == pytest.approx(0.0002) for r in results)

    def test_funding_rate_updates_on_new_event(self) -> None:
        """When a new FundingRateUpdate arrives, the cached rate updates."""
        latest_funding_rates: dict[str, float] = {}
        instrument_id = "BTCUSDT.BINANCE"

        # First FR event
        latest_funding_rates[instrument_id] = 0.0001

        bar_data_1 = _extract_bar_data(_make_mock_bar())
        if instrument_id in latest_funding_rates:
            bar_data_1["funding_rate"] = latest_funding_rates[instrument_id]
        assert bar_data_1["funding_rate"] == pytest.approx(0.0001)

        # Second FR event (rate changed)
        latest_funding_rates[instrument_id] = -0.0003

        bar_data_2 = _extract_bar_data(_make_mock_bar())
        if instrument_id in latest_funding_rates:
            bar_data_2["funding_rate"] = latest_funding_rates[instrument_id]
        assert bar_data_2["funding_rate"] == pytest.approx(-0.0003)


# ---------------------------------------------------------------------------
# OI injection in on_bar
# ---------------------------------------------------------------------------


class TestOiInjection:
    """Simulate the OI lookup injection from on_bar."""

    def test_oi_injected_when_matching_ts(self) -> None:
        """When _oi_lookup has data for the instrument and matching ts,
        bar_data should contain open_interest and open_interest_value.
        """
        bar_data = _extract_bar_data(_make_mock_bar())

        ts = 1700000000000 * 1_000_000  # ns
        oi_lookup = {
            "BTCUSDT.BINANCE": {
                ts: {
                    "open_interest": 5000.0,
                    "open_interest_value": 175000000.0,
                },
            },
        }
        instrument_id = "BTCUSDT.BINANCE"

        # Replicate the on_bar injection logic
        if oi_lookup and instrument_id in oi_lookup:
            oi_data = oi_lookup[instrument_id].get(ts)
            if oi_data:
                bar_data.update(oi_data)

        assert bar_data["open_interest"] == pytest.approx(5000.0)
        assert bar_data["open_interest_value"] == pytest.approx(175000000.0)

    def test_oi_not_injected_when_no_matching_ts(self) -> None:
        """When the bar ts does not exist in the OI lookup, no OI fields
        should appear in bar_data.
        """
        bar_data = _extract_bar_data(_make_mock_bar())

        ts_bar = 1700000000000 * 1_000_000
        ts_oi = 1700014400000 * 1_000_000  # Different timestamp
        oi_lookup = {
            "BTCUSDT.BINANCE": {
                ts_oi: {
                    "open_interest": 5000.0,
                    "open_interest_value": 175000000.0,
                },
            },
        }
        instrument_id = "BTCUSDT.BINANCE"

        if oi_lookup and instrument_id in oi_lookup:
            oi_data = oi_lookup[instrument_id].get(ts_bar)
            if oi_data:
                bar_data.update(oi_data)

        assert "open_interest" not in bar_data
        assert "open_interest_value" not in bar_data

    def test_oi_not_injected_when_instrument_missing(self) -> None:
        """When the instrument has no OI data at all, bar_data is unaffected."""
        bar_data = _extract_bar_data(_make_mock_bar())

        ts = 1700000000000 * 1_000_000
        oi_lookup: dict[str, dict[int, dict[str, float]]] = {}
        instrument_id = "BTCUSDT.BINANCE"

        if oi_lookup and instrument_id in oi_lookup:
            oi_data = oi_lookup[instrument_id].get(ts)
            if oi_data:
                bar_data.update(oi_data)

        assert "open_interest" not in bar_data

    def test_oi_and_fr_combined_injection(self) -> None:
        """Both funding rate and OI can be injected into the same bar_data."""
        bar_data = _extract_bar_data(_make_mock_bar())

        instrument_id = "BTCUSDT.BINANCE"
        ts = 1700000000000 * 1_000_000

        # FR injection
        latest_funding_rates = {instrument_id: 0.0001}
        if instrument_id in latest_funding_rates:
            bar_data["funding_rate"] = latest_funding_rates[instrument_id]

        # OI injection
        oi_lookup = {
            instrument_id: {
                ts: {
                    "open_interest": 5000.0,
                    "open_interest_value": 175000000.0,
                },
            },
        }
        if oi_lookup and instrument_id in oi_lookup:
            oi_data = oi_lookup[instrument_id].get(ts)
            if oi_data:
                bar_data.update(oi_data)

        # All 8 fields present: OHLCV(5) + FR(1) + OI(2)
        assert len(bar_data) == 8
        assert bar_data["funding_rate"] == pytest.approx(0.0001)
        assert bar_data["open_interest"] == pytest.approx(5000.0)
        assert bar_data["open_interest_value"] == pytest.approx(175000000.0)
