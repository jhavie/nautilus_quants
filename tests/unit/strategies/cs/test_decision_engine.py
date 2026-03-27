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


def _opens(orders):
    """Orders with target > 0 for instruments not currently held."""
    return [o for o in orders if o["target_quote_quantity"] > 0 and "NEW_" in str(o.get("tags", []))]


def _closes(orders):
    """Orders with target = 0."""
    return [o for o in orders if o["target_quote_quantity"] == 0]


def _flips(orders):
    """Orders tagged with FLIP_TO_*."""
    return [o for o in orders if any("FLIP_TO" in t for t in o.get("tags", []))]


def _holds(orders):
    """Orders tagged with HOLD_*."""
    return [o for o in orders if any("HOLD_" in t for t in o.get("tags", []))]


class TestComputeOrders:
    """Test the _compute_orders pure logic (called directly, no Actor lifecycle)."""

    def test_fresh_start_no_positions(self):
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(composite, current_long=set(), current_short=set())

        new_orders = [o for o in orders if o["target_quote_quantity"] > 0]
        assert len(new_orders) == 4  # 2 long + 2 short
        long_opens = [o for o in new_orders if o["order_side"] == "BUY"]
        short_opens = [o for o in new_orders if o["order_side"] == "SELL"]
        assert len(long_opens) == 2  # A, B (lowest)
        assert len(short_opens) == 2  # C, D (highest)

    def test_no_change_when_already_in_target(self):
        """Fixed mode: HOLD instruments produce no orders (qlib parity)."""
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A", "B"},
            current_short={"C", "D"},
        )
        assert orders == []

    def test_flip_short_to_long(self):
        """NETTING mode: D goes from short to long target."""
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"D": 0.5, "A": 1.0, "B": 3.0, "C": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A"},
            current_short={"C", "D"},
        )
        d_orders = [o for o in orders if o["instrument_id"] == "D"]
        assert len(d_orders) == 1
        assert d_orders[0]["order_side"] == "BUY"
        assert d_orders[0]["target_quote_quantity"] > 0
        assert "FLIP_TO_LONG" in d_orders[0]["tags"]

    def test_flip_long_to_short(self):
        """NETTING mode: A goes from long to short target."""
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"B": 1.0, "C": 2.0, "D": 3.0, "A": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A", "B"},
            current_short={"D"},
        )
        a_orders = [o for o in orders if o["instrument_id"] == "A"]
        flip = [o for o in a_orders if "FLIP_TO_SHORT" in o.get("tags", [])]
        assert len(flip) == 1
        assert flip[0]["order_side"] == "SELL"

    def test_flip_target_quote_quantity_is_position_value(self):
        actor = _make_actor(position_value=500.0)
        composite = {"D": 0.5, "A": 1.0, "B": 3.0, "C": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A"},
            current_short={"C", "D"},
        )
        d_orders = [o for o in orders if o["instrument_id"] == "D" and "FLIP_TO_LONG" in o.get("tags", [])]
        assert len(d_orders) == 1
        assert d_orders[0]["target_quote_quantity"] == 500.0

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
        assert close_delisted[0]["target_quote_quantity"] == 0
        assert "NO_FACTOR_DATA" in close_delisted[0]["tags"]

    def test_target_quote_quantity_set_for_new_positions(self):
        actor = _make_actor(position_value=500.0)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(composite, current_long=set(), current_short=set())
        new_orders = [o for o in orders if "NEW_" in str(o.get("tags", []))]
        for o in new_orders:
            assert o["target_quote_quantity"] == 500.0

    def test_close_orders_have_zero_target_quote_quantity(self):
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long={"A", "GONE"},
            current_short=set(),
        )
        closes = _closes(orders)
        assert len(closes) >= 1
        for c in closes:
            assert c["target_quote_quantity"] == 0

    def test_new_open_when_no_existing_position(self):
        """When target is long but no existing position, tag is NEW_LONG not FLIP."""
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(
            composite,
            current_long=set(),
            current_short={"C", "D"},
        )
        a_orders = [o for o in orders if o["instrument_id"] == "A"]
        assert len(a_orders) == 1
        assert "NEW_LONG" in a_orders[0]["tags"]
        assert not any("FLIP" in t for t in a_orders[0]["tags"])

    def test_order_dict_has_no_intent_field(self):
        """Verify intent field is not present in order dicts."""
        actor = _make_actor(n_long=2, n_short=2)
        composite = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        orders = actor._compute_orders(composite, current_long=set(), current_short=set())
        for o in orders:
            assert "intent" not in o
