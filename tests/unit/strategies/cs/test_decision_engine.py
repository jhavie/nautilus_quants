# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for DecisionEngineActor._compute_orders logic."""

import pytest

from nautilus_quants.strategies.cs.decision_engine import DecisionEngineActor
from nautilus_quants.strategies.cs.config import DecisionEngineActorConfig


def _make_actor(n_long=2, n_short=2, position_value=1000.0) -> DecisionEngineActor:
    config = DecisionEngineActorConfig(
        n_long=n_long,
        n_short=n_short,
        position_value=position_value,
    )
    return DecisionEngineActor(config)


class TestComputeOrders:
    """Test the _compute_orders pure logic (called directly, no Actor lifecycle)."""

    def test_fresh_start_no_positions(self):
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(composite, current_long=set(), current_short=set())

        opens = [o for o in orders if o["action"] == "OPEN"]
        assert len(opens) == 4  # 2 long + 2 short
        long_opens = [o for o in opens if o["order_side"] == "BUY"]
        short_opens = [o for o in opens if o["order_side"] == "SELL"]
        assert len(long_opens) == 2  # A, B (lowest)
        assert len(short_opens) == 2  # C, D (highest)

    def test_no_change_when_already_in_target(self):
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A", "B"},
            current_short={"C", "D"},
        )
        assert orders == []

    def test_flip_short_to_long_generates_single_flip(self):
        """NETTING mode: flip generates one FLIP order, not CLOSE + OPEN."""
        actor = _make_actor(n_long=2, n_short=2)
        # D was highest (short), now D becomes lowest (long target)
        composite = {"D": 0.5, "A": 1.0, "B": 3.0, "C": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A"},
            current_short={"C", "D"},
        )
        # D: single FLIP order (not CLOSE + OPEN)
        flip_d = [o for o in orders if o["instrument_id"] == "D" and o["action"] == "FLIP"]
        close_d = [o for o in orders if o["instrument_id"] == "D" and o["action"] == "CLOSE"]
        open_d = [o for o in orders if o["instrument_id"] == "D" and o["action"] == "OPEN"]
        assert len(flip_d) == 1
        assert len(close_d) == 0  # No separate close
        assert len(open_d) == 0  # No separate open
        assert flip_d[0]["order_side"] == "BUY"
        assert flip_d[0]["reduce_only"] is False
        assert "FLIP_TO_LONG" in flip_d[0]["tags"]

    def test_flip_long_to_short_generates_single_flip(self):
        """NETTING mode: long→short flip generates one FLIP order."""
        actor = _make_actor(n_long=2, n_short=2)
        # A was lowest (long), now A becomes highest (short target)
        composite = {"B": 1.0, "C": 2.0, "D": 3.0, "A": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A", "B"},
            current_short={"D"},
        )
        flip_a = [o for o in orders if o["instrument_id"] == "A" and o["action"] == "FLIP"]
        assert len(flip_a) == 1
        assert flip_a[0]["order_side"] == "SELL"
        assert flip_a[0]["reduce_only"] is False
        assert "FLIP_TO_SHORT" in flip_a[0]["tags"]

    def test_flip_quote_quantity_is_position_value(self):
        """FLIP quote_quantity = position_value (1×), not 2×."""
        actor = _make_actor(position_value=500.0)
        composite = {"D": 0.5, "A": 1.0, "B": 3.0, "C": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A"},
            current_short={"C", "D"},
        )
        flip_d = [o for o in orders if o["instrument_id"] == "D" and o["action"] == "FLIP"]
        assert len(flip_d) == 1
        assert flip_d[0]["quote_quantity"] == 500.0

    def test_delisting_protection(self):
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A", "DELISTED"},
            current_short={"C"},
        )
        close_delisted = [o for o in orders if o["instrument_id"] == "DELISTED"]
        assert len(close_delisted) == 1
        assert "NO_FACTOR_DATA" in close_delisted[0]["tags"]

    def test_quote_quantity_set_for_opens(self):
        actor = _make_actor(position_value=500.0)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(composite, current_long=set(), current_short=set())
        opens = [o for o in orders if o["action"] == "OPEN"]
        for o in opens:
            assert o["quote_quantity"] == 500.0
            assert o["reduce_only"] is False

    def test_close_orders_have_zero_quote_quantity(self):
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A", "GONE"},
            current_short=set(),
        )
        closes = [o for o in orders if o["action"] == "CLOSE"]
        for c in closes:
            assert c["quote_quantity"] == 0
            assert c["reduce_only"] is True

    def test_new_open_when_no_existing_position(self):
        """When target is long but no existing position, should be OPEN not FLIP."""
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long=set(),
            current_short={"C", "D"},
        )
        open_a = [o for o in orders if o["instrument_id"] == "A" and o["action"] == "OPEN"]
        flip_a = [o for o in orders if o["instrument_id"] == "A" and o["action"] == "FLIP"]
        assert len(open_a) == 1
        assert len(flip_a) == 0
