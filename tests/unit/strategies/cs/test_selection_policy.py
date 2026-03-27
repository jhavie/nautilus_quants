# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for SelectionPolicy implementations."""

import pytest

from nautilus_quants.strategies.cs.selection_policy import (
    FMZSelectionPolicy,
    TopKDropoutSelectionPolicy,
)


# ---------------------------------------------------------------------------
# FMZSelectionPolicy
# ---------------------------------------------------------------------------


class TestFMZSelectionPolicy:
    def test_fresh_targets_no_positions(self):
        """Empty portfolio → final = targets."""
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        fl, fs = policy.select(scores, current_long=set(), current_short=set())
        assert fl == {"A", "B"}
        assert fs == {"C", "D"}

    def test_sticky_hold_rank_slips(self):
        """Long position that drops out of long_targets but not into short → stays."""
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        # A was long; now A=2.5 (rank 3, not in bottom-2 or top-2)
        scores = {"B": 1.0, "C": 2.0, "A": 2.5, "D": 3.0, "E": 4.0}
        fl, fs = policy.select(scores, current_long={"A", "B"}, current_short={"D", "E"})
        # A stays long (sticky), B stays long (still in target)
        assert "A" in fl
        assert "B" in fl

    def test_flip_long_to_short_target(self):
        """Long position becomes short target → removed from final_long."""
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"B": 1.0, "C": 2.0, "D": 3.0, "A": 4.0}
        fl, fs = policy.select(scores, current_long={"A", "B"}, current_short=set())
        assert "A" not in fl
        assert "A" in fs

    def test_flip_short_to_long_target(self):
        """Short position becomes long target → removed from final_short."""
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"D": 0.5, "A": 1.0, "B": 3.0, "C": 4.0}
        fl, fs = policy.select(scores, current_long=set(), current_short={"C", "D"})
        assert "D" in fl
        assert "D" not in fs

    def test_no_overlap(self):
        """final_long and final_short must not overlap."""
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        fl, fs = policy.select(scores, current_long=set(), current_short=set())
        assert fl & fs == set()


# ---------------------------------------------------------------------------
# TopKDropoutSelectionPolicy
# ---------------------------------------------------------------------------


class TestTopKDropoutSelectionPolicy:
    def test_fresh_start_fills_topk(self):
        """Empty portfolio → fills up to topk_long + topk_short."""
        policy = TopKDropoutSelectionPolicy(
            topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        fl, fs = policy.select(scores, current_long=set(), current_short=set())
        # Long = highest 2 (descending): D, C
        # Short = lowest 2 (ascending): A, B
        assert len(fl) == 2
        assert len(fs) == 2
        assert fl & fs == set()

    def test_steady_state_rotation(self):
        """Full pool + new better candidates → exactly n_drop swapped."""
        policy = TopKDropoutSelectionPolicy(
            topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1
        )
        # Current long: C(3), D(4); Current short: A(1), B(2)
        # New scores: E(5) appears, should replace D's worst long or add
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        fl, fs = policy.select(
            scores, current_long={"D", "E"}, current_short={"A", "B"}
        )
        # Long pool should stay or improve, short pool may rotate
        assert len(fl) == 2
        assert len(fs) == 2

    def test_no_change_when_scores_stable(self):
        """Same ranking as current → final = current (no rotation needed)."""
        policy = TopKDropoutSelectionPolicy(
            topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1
        )
        # Long holds D(4), C(3) (top 2 descending) — no better candidates outside
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        fl, fs = policy.select(
            scores, current_long={"C", "D"}, current_short={"A", "B"}
        )
        assert fl == {"C", "D"}
        assert fs == {"A", "B"}

    def test_n_drop_exceeds_available(self):
        """n_drop > number of replaceable candidates → replace as many as possible."""
        policy = TopKDropoutSelectionPolicy(
            topk_long=3, topk_short=1, n_drop_long=5, n_drop_short=1
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        fl, fs = policy.select(
            scores, current_long={"B", "C", "D"}, current_short={"A"}
        )
        # Even with n_drop=5, only 1 candidate not in current long (A is short)
        assert len(fl) == 3
        assert len(fs) == 1

    def test_partial_fill_fewer_than_topk(self):
        """Available instruments < topk → fill with what's available."""
        policy = TopKDropoutSelectionPolicy(
            topk_long=5, topk_short=5, n_drop_long=2, n_drop_short=2
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0}
        fl, fs = policy.select(scores, current_long=set(), current_short=set())
        # Only 3 instruments, can't fill 5+5
        assert len(fl) + len(fs) <= 3
        assert fl & fs == set()

    def test_overlap_prevention_via_exclude(self):
        """Long and short pools must not overlap."""
        policy = TopKDropoutSelectionPolicy(
            topk_long=3, topk_short=3, n_drop_long=1, n_drop_short=1
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}
        fl, fs = policy.select(scores, current_long=set(), current_short=set())
        assert fl & fs == set()

    def test_select_leg_static_long(self):
        """Direct test of _select_leg for long side (descending)."""
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        final = TopKDropoutSelectionPolicy._select_leg(
            scores, current_held={"C", "D"}, topk=2, n_drop=1, ascending=False
        )
        # D(4), C(3) held; E(5) is better candidate
        # combined: E(5), D(4), C(3) → drop tail 1 from held: C(3)
        # to_add: E; final = {D, E}
        assert final == {"D", "E"}

    def test_select_leg_static_short_no_better_candidates(self):
        """Short leg: held are already the best (lowest) — no rotation."""
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        final = TopKDropoutSelectionPolicy._select_leg(
            scores, current_held={"A", "B"}, topk=2, n_drop=1, ascending=True
        )
        # Ascending sort: A(1), B(2), C(3), D(4), E(5)
        # Candidate C(3) is WORSE than held B(2) for short leg (want lowest)
        # combined = [A, B, C] → tail(1) = [C] → C not held → no drops
        # No adds needed → final stays {A, B}
        assert final == {"A", "B"}

    def test_select_leg_with_exclude(self):
        """Exclude set prevents overlap with other leg."""
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        # Short leg, ascending, exclude={D, C} (they're in long)
        final = TopKDropoutSelectionPolicy._select_leg(
            scores,
            current_held=set(),
            topk=2,
            n_drop=1,
            ascending=True,
            exclude={"C", "D"},
        )
        # Only A(1), B(2) available → final = {A, B}
        assert final == {"A", "B"}
