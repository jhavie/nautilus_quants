# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for EventTimePriceBook."""

from nautilus_quants.common.event_time_price_book import EventTimePriceBook


def test_build_snapshot_is_deterministic_with_unsorted_insert_order() -> None:
    book = EventTimePriceBook(max_timestamps=6)
    book.record_close(100, "B.BINANCE", 20.0)
    book.record_close(100, "A.BINANCE", 10.0)

    snapshot, missing = book.build_snapshot(100, ["B.BINANCE", "A.BINANCE"])

    assert missing == []
    assert list(snapshot.keys()) == ["A.BINANCE", "B.BINANCE"]
    assert snapshot["A.BINANCE"] == 10.0
    assert snapshot["B.BINANCE"] == 20.0


def test_build_snapshot_never_falls_back_to_other_timestamps() -> None:
    book = EventTimePriceBook(max_timestamps=6)
    book.record_close(100, "A.BINANCE", 10.0)
    book.record_close(200, "B.BINANCE", 20.0)

    snapshot, missing = book.build_snapshot(100, ["A.BINANCE", "B.BINANCE"])

    assert snapshot == {"A.BINANCE": 10.0}
    assert missing == ["B.BINANCE"]


def test_get_latest_closes_returns_most_recent_price() -> None:
    """Later timestamps overwrite earlier ones for the same instrument."""
    book = EventTimePriceBook(max_timestamps=6)
    book.record_close(100, "A.BINANCE", 10.0)
    book.record_close(100, "B.BINANCE", 20.0)
    book.record_close(200, "A.BINANCE", 15.0)  # overwrites 10.0
    book.record_close(200, "C.BINANCE", 30.0)

    result = book.get_latest_closes()
    assert result == {"A.BINANCE": 15.0, "B.BINANCE": 20.0, "C.BINANCE": 30.0}


def test_get_latest_closes_empty_book() -> None:
    book = EventTimePriceBook(max_timestamps=6)
    assert book.get_latest_closes() == {}


def test_window_evicts_oldest_timestamp() -> None:
    book = EventTimePriceBook(max_timestamps=2)
    book.record_close(100, "A.BINANCE", 10.0)
    book.record_close(200, "A.BINANCE", 20.0)
    book.record_close(300, "A.BINANCE", 30.0)

    snapshot_oldest, missing_oldest = book.build_snapshot(100, ["A.BINANCE"])
    snapshot_latest, missing_latest = book.build_snapshot(300, ["A.BINANCE"])

    assert snapshot_oldest == {}
    assert missing_oldest == ["A.BINANCE"]
    assert snapshot_latest == {"A.BINANCE": 30.0}
    assert missing_latest == []
