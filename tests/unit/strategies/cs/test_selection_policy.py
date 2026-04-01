# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for SelectionPolicy implementations."""

import math

import pytest

from nautilus_quants.strategies.cs.selection_policy import (
    FMZSelectionPolicy,
    TargetPosition,
    TopKDropoutSelectionPolicy,
)
from nautilus_quants.strategies.cs.worldquant_selection_policy import (
    WorldQuantSelectionPolicy,
)


# ---------------------------------------------------------------------------
# TargetPosition
# ---------------------------------------------------------------------------


class TestTargetPosition:
    def test_positive_weight_is_long(self):
        t = TargetPosition("BTC", 0.5, 1.0)
        assert t.weight > 0

    def test_negative_weight_is_short(self):
        t = TargetPosition("BTC", -0.3, 1.0)
        assert t.weight < 0

    def test_frozen(self):
        t = TargetPosition("BTC", 0.5, 1.0)
        with pytest.raises(AttributeError):
            t.weight = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _longs(targets: list[TargetPosition]) -> set[str]:
    return {t.symbol for t in targets if t.weight > 0}


def _shorts(targets: list[TargetPosition]) -> set[str]:
    return {t.symbol for t in targets if t.weight < 0}


def _is_sorted_by_factor(targets: list[TargetPosition]) -> bool:
    factors = [t.factor for t in targets]
    return factors == sorted(factors)


# ---------------------------------------------------------------------------
# FMZSelectionPolicy
# ---------------------------------------------------------------------------


class TestFMZSelectionPolicy:
    def test_fresh_targets_no_positions(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        # Highest scores = long, lowest = short (PR #71)
        assert _longs(targets) == {"C", "D"}
        assert _shorts(targets) == {"A", "B"}

    def test_sticky_hold_rank_slips(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        # Sorted: B(1), C(2), A(2.5), D(3), E(4) → long={D,E}, short={B,C}
        # A slipped from top-2 but not in short targets → sticky hold
        scores = {"B": 1.0, "C": 2.0, "A": 2.5, "D": 3.0, "E": 4.0}
        targets = policy.select(scores, current_long={"A", "E"}, current_short={"B", "C"})
        assert "A" in _longs(targets)
        assert "E" in _longs(targets)

    def test_flip_long_to_short_target(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        # Sorted: B(1), C(2), D(3), A(4) → long={A,D}, short={B,C}
        # B was long → flips to short
        scores = {"B": 1.0, "C": 2.0, "D": 3.0, "A": 4.0}
        targets = policy.select(scores, current_long={"A", "B"}, current_short=set())
        assert "B" not in _longs(targets)
        assert "B" in _shorts(targets)

    def test_flip_short_to_long_target(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        # Sorted: D(0.5), A(1), B(3), C(4) → long={B,C}, short={D,A}
        # C was short → flips to long
        scores = {"D": 0.5, "A": 1.0, "B": 3.0, "C": 4.0}
        targets = policy.select(scores, current_long=set(), current_short={"C", "D"})
        assert "C" in _longs(targets)
        assert "C" not in _shorts(targets)

    def test_no_overlap(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _longs(targets) & _shorts(targets) == set()

    def test_equal_weight(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        weights = [abs(t.weight) for t in targets]
        assert all(w == pytest.approx(weights[0]) for w in weights)

    def test_sorted_by_factor(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=2)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _is_sorted_by_factor(targets)

    def test_n_short_zero(self):
        policy = FMZSelectionPolicy(n_long=2, n_short=0)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _longs(targets) == {"B", "C"}
        assert _shorts(targets) == set()


# ---------------------------------------------------------------------------
# TopKDropoutSelectionPolicy
# ---------------------------------------------------------------------------


class TestTopKDropoutSelectionPolicy:
    def test_fresh_start_fills_topk(self):
        policy = TopKDropoutSelectionPolicy(
            topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1,
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert len(_longs(targets)) == 2
        assert len(_shorts(targets)) == 2
        assert _longs(targets) & _shorts(targets) == set()

    def test_steady_state_rotation(self):
        policy = TopKDropoutSelectionPolicy(
            topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1,
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        targets = policy.select(
            scores, current_long={"D", "E"}, current_short={"A", "B"},
        )
        assert len(_longs(targets)) == 2
        assert len(_shorts(targets)) == 2

    def test_no_change_when_scores_stable(self):
        policy = TopKDropoutSelectionPolicy(
            topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1,
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(
            scores, current_long={"C", "D"}, current_short={"A", "B"},
        )
        assert _longs(targets) == {"C", "D"}
        assert _shorts(targets) == {"A", "B"}

    def test_overlap_prevention(self):
        policy = TopKDropoutSelectionPolicy(
            topk_long=3, topk_short=3, n_drop_long=1, n_drop_short=1,
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _longs(targets) & _shorts(targets) == set()

    def test_returns_target_positions_sorted(self):
        policy = TopKDropoutSelectionPolicy(
            topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1,
        )
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _is_sorted_by_factor(targets)

    def test_select_leg_static_long(self):
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        final = TopKDropoutSelectionPolicy._select_leg(
            scores, current_held={"C", "D"}, topk=2, n_drop=1, ascending=False,
        )
        assert final == {"D", "E"}

    def test_select_leg_static_short_no_better_candidates(self):
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        final = TopKDropoutSelectionPolicy._select_leg(
            scores, current_held={"A", "B"}, topk=2, n_drop=1, ascending=True,
        )
        assert final == {"A", "B"}


# ---------------------------------------------------------------------------
# WorldQuantSelectionPolicy
# ---------------------------------------------------------------------------


class TestWorldQuantSelectionPolicy:
    def test_delay_warmup_returns_none(self):
        """delay=1: first call returns None (warmup, no opinion)."""
        policy = WorldQuantSelectionPolicy(delay=1)
        scores = {"A": 1.0, "B": -1.0}
        targets = policy.select(scores, current_long={"A"}, current_short={"B"})
        assert targets is None

    def test_delay_zero_no_warmup(self):
        """delay=0: first call produces targets immediately."""
        policy = WorldQuantSelectionPolicy(delay=0)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert len(targets) > 0

    def test_delay_one_second_call_returns_targets(self):
        """delay=1: second call uses first call's data."""
        policy = WorldQuantSelectionPolicy(delay=1)
        scores_t1 = {"A": 1.0, "B": 2.0, "C": 3.0}
        policy.select(scores_t1, current_long=set(), current_short=set())
        scores_t2 = {"A": 2.0, "B": 1.0, "C": 3.0}
        targets = policy.select(scores_t2, current_long=set(), current_short=set())
        # Should use scores_t1 (delayed)
        assert len(targets) > 0

    def test_neutralize_sum_zero(self):
        alpha = {"A": 3.0, "B": 1.0, "C": 2.0}
        result = WorldQuantSelectionPolicy._neutralize(alpha)
        assert sum(result.values()) == pytest.approx(0.0)

    def test_scale_sum_abs_one(self):
        alpha = {"A": 0.5, "B": -0.3, "C": 0.2}
        result = WorldQuantSelectionPolicy._scale(alpha)
        assert sum(abs(v) for v in result.values()) == pytest.approx(1.0)

    def test_scale_all_zeros(self):
        alpha = {"A": 0.0, "B": 0.0}
        result = WorldQuantSelectionPolicy._scale(alpha)
        assert result == alpha

    def test_decay_zero_passthrough(self):
        policy = WorldQuantSelectionPolicy(decay=0)
        alpha = {"A": 0.5, "B": -0.5}
        result = policy._apply_decay(alpha)
        assert result == alpha

    def test_decay_weighted_average(self):
        policy = WorldQuantSelectionPolicy(decay=3)
        # Feed 3 periods
        policy._apply_decay({"A": 1.0})  # weight=1
        policy._apply_decay({"A": 2.0})  # weight=2
        result = policy._apply_decay({"A": 3.0})  # weight=3
        # Expected: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
        assert result["A"] == pytest.approx(14.0 / 6.0)

    def test_truncation_caps_weight(self):
        policy = WorldQuantSelectionPolicy(truncation=0.1)
        alpha = {"A": 0.5, "B": -0.3, "C": 0.2}
        result = policy._truncate(alpha)
        assert all(abs(v) <= 0.1 for v in result.values())

    def test_truncation_iterative_convergence(self):
        """Iterative truncation + re-scale converges."""
        policy = WorldQuantSelectionPolicy(delay=0, truncation=0.05)
        scores = {f"S{i}": float(i) for i in range(20)}
        targets = policy.select(scores, current_long=set(), current_short=set())
        for t in targets:
            assert abs(t.weight) <= 0.06  # convergence tolerance

    def test_returns_signed_weights(self):
        """Long weights positive, short weights negative."""
        policy = WorldQuantSelectionPolicy(delay=0, neutralization="MARKET")
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        for t in targets:
            if t.weight > 0:
                assert t.symbol in _longs(targets)
            elif t.weight < 0:
                assert t.symbol in _shorts(targets)

    def test_enable_long_false(self):
        policy = WorldQuantSelectionPolicy(delay=0, enable_long=False)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _longs(targets) == set()

    def test_enable_short_false(self):
        policy = WorldQuantSelectionPolicy(delay=0, enable_short=False)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _shorts(targets) == set()

    def test_sorted_by_factor(self):
        policy = WorldQuantSelectionPolicy(delay=0)
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        assert _is_sorted_by_factor(targets)

    def test_nan_filtered(self):
        policy = WorldQuantSelectionPolicy(delay=0)
        scores = {"A": 1.0, "B": float("nan"), "C": 3.0}
        targets = policy.select(scores, current_long=set(), current_short=set())
        symbols = {t.symbol for t in targets}
        assert "B" not in symbols
