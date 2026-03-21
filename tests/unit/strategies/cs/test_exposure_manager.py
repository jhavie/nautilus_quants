# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for ExposureManager."""

import pytest

from nautilus_quants.strategies.cs.exposure_manager import ExposureManager, ExposurePolicy


def _close(inst_id: str, **kw) -> dict:
    return {"instrument_id": inst_id, "action": "CLOSE", "order_side": "SELL", **kw}


def _open(inst_id: str, **kw) -> dict:
    return {"instrument_id": inst_id, "action": "OPEN", "order_side": "BUY", **kw}


class TestCloseFirst:
    """CLOSE_FIRST policy: execute all closes, release opens one-by-one."""

    def test_no_closes_returns_opens_immediately(self):
        mgr = ExposureManager(ExposurePolicy.CLOSE_FIRST)
        closes, opens = mgr.submit_plan([], [_open("SUI")])
        assert closes == []
        assert len(opens) == 1
        assert opens[0]["instrument_id"] == "SUI"

    def test_closes_first_opens_queued(self):
        mgr = ExposureManager(ExposurePolicy.CLOSE_FIRST)
        closes, opens = mgr.submit_plan(
            [_close("DOGE"), _close("XRP")],
            [_open("SUI"), _open("APT")],
        )
        assert len(closes) == 2
        assert opens == []
        assert mgr.has_pending

    def test_one_close_releases_one_open(self):
        mgr = ExposureManager(ExposurePolicy.CLOSE_FIRST)
        mgr.submit_plan(
            [_close("DOGE"), _close("XRP")],
            [_open("SUI"), _open("APT")],
        )
        released = mgr.on_close_filled("DOGE")
        assert len(released) == 1
        assert released[0]["instrument_id"] == "SUI"

    def test_all_closes_release_remaining_opens(self):
        mgr = ExposureManager(ExposurePolicy.CLOSE_FIRST)
        mgr.submit_plan(
            [_close("DOGE"), _close("XRP")],
            [_open("SUI"), _open("APT"), _open("ARB")],
        )
        mgr.on_close_filled("DOGE")  # releases SUI
        released = mgr.on_close_filled("XRP")  # releases APT + ARB
        assert len(released) == 2
        assert not mgr.has_pending

    def test_more_closes_than_opens(self):
        mgr = ExposureManager(ExposurePolicy.CLOSE_FIRST)
        mgr.submit_plan(
            [_close("A"), _close("B"), _close("C")],
            [_open("X")],
        )
        released_a = mgr.on_close_filled("A")
        assert len(released_a) == 1  # releases X

        released_b = mgr.on_close_filled("B")
        assert released_b == []  # no more opens

        released_c = mgr.on_close_filled("C")
        assert released_c == []
        assert not mgr.has_pending

    def test_on_open_filled_is_noop_in_close_first(self):
        mgr = ExposureManager(ExposurePolicy.CLOSE_FIRST)
        mgr.submit_plan([_close("DOGE")], [_open("SUI")])
        released = mgr.on_open_filled("SUI")
        assert released == []

    def test_on_stop_clears_all(self):
        mgr = ExposureManager(ExposurePolicy.CLOSE_FIRST)
        mgr.submit_plan(
            [_close("DOGE")],
            [_open("SUI"), _open("APT")],
        )
        mgr.on_stop()
        assert not mgr.has_pending
        released = mgr.on_close_filled("DOGE")
        assert released == []


class TestOpenFirst:
    """OPEN_FIRST policy: execute all opens, release closes one-by-one."""

    def test_no_opens_returns_closes_immediately(self):
        mgr = ExposureManager(ExposurePolicy.OPEN_FIRST)
        closes, opens = mgr.submit_plan([_close("DOGE")], [])
        assert len(closes) == 1
        assert opens == []

    def test_opens_first_closes_queued(self):
        mgr = ExposureManager(ExposurePolicy.OPEN_FIRST)
        closes, opens = mgr.submit_plan(
            [_close("DOGE"), _close("XRP")],
            [_open("SUI"), _open("APT")],
        )
        assert closes == []
        assert len(opens) == 2
        assert mgr.has_pending

    def test_one_open_releases_one_close(self):
        mgr = ExposureManager(ExposurePolicy.OPEN_FIRST)
        mgr.submit_plan(
            [_close("DOGE"), _close("XRP")],
            [_open("SUI"), _open("APT")],
        )
        released = mgr.on_open_filled("SUI")
        assert len(released) == 1
        assert released[0]["instrument_id"] == "DOGE"

    def test_all_opens_release_remaining_closes(self):
        mgr = ExposureManager(ExposurePolicy.OPEN_FIRST)
        mgr.submit_plan(
            [_close("DOGE"), _close("XRP"), _close("ARB")],
            [_open("SUI"), _open("APT")],
        )
        mgr.on_open_filled("SUI")  # releases DOGE
        released = mgr.on_open_filled("APT")  # releases XRP + ARB
        assert len(released) == 2
        assert not mgr.has_pending

    def test_on_close_filled_is_noop_in_open_first(self):
        mgr = ExposureManager(ExposurePolicy.OPEN_FIRST)
        mgr.submit_plan([_close("DOGE")], [_open("SUI")])
        released = mgr.on_close_filled("DOGE")
        assert released == []


class TestStateSummary:
    def test_idle(self):
        mgr = ExposureManager()
        assert mgr.state_summary == "IDLE"

    def test_processing(self):
        mgr = ExposureManager()
        mgr.submit_plan([_close("A")], [_open("B")])
        assert "PROCESSING" in mgr.state_summary

    def test_stopping(self):
        mgr = ExposureManager()
        mgr.on_stop()
        assert mgr.state_summary == "STOPPING"
