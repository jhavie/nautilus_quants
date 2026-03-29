# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Regression tests: new architecture produces same results as old architecture.

Three equivalence axes:
1. FMZ/TopK selection sets — new list[TargetPosition] == old tuple[set, set]
2. DecisionEngine orders — position_mode="fixed" == old rebalance_to_weights=False
3. WorldQuant pipeline — WorldQuantSelectionPolicy weights == WorldQuantAlphaStrategy._process_alpha
"""

import math

import pytest

from nautilus_quants.strategies.cs.config import DecisionEngineActorConfig
from nautilus_quants.strategies.cs.decision_engine import DecisionEngineActor
from nautilus_quants.strategies.cs.selection_policy import (
    FMZSelectionPolicy,
    TargetPosition,
    TopKDropoutSelectionPolicy,
)
from nautilus_quants.strategies.cs.worldquant_selection_policy import (
    WorldQuantSelectionPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _longs(targets: list[TargetPosition]) -> set[str]:
    return {t.symbol for t in targets if t.weight > 0}


def _shorts(targets: list[TargetPosition]) -> set[str]:
    return {t.symbol for t in targets if t.weight < 0}


# ---------------------------------------------------------------------------
# 1. FMZ Selection Equivalence
# ---------------------------------------------------------------------------

class _OldFMZSelectionPolicy:
    """Original FMZ implementation (tuple[set, set] return) for comparison."""

    def __init__(self, n_long: int, n_short: int) -> None:
        self._n_long = n_long
        self._n_short = n_short

    def select(
        self, scores: dict[str, float],
        current_long: set[str], current_short: set[str],
    ) -> tuple[set[str], set[str]]:
        sorted_symbols = sorted(scores.items(), key=lambda x: (x[1], x[0]))
        long_targets = set(s for s, _ in sorted_symbols[:self._n_long])
        short_targets = set(s for s, _ in sorted_symbols[-self._n_short:])
        final_long = (current_long - short_targets) | long_targets
        final_short = (current_short - long_targets) | short_targets
        return final_long, final_short


class TestFMZEquivalence:
    """New FMZSelectionPolicy produces same long/short sets as old version."""

    SCENARIOS = [
        # (name, scores, current_long, current_short)
        ("fresh_start", {"A": 1, "B": 2, "C": 3, "D": 4}, set(), set()),
        ("all_held", {"A": 1, "B": 2, "C": 3, "D": 4}, {"A", "B"}, {"C", "D"}),
        ("sticky_hold", {"B": 1, "C": 2, "A": 2.5, "D": 3, "E": 4}, {"A", "B"}, {"D", "E"}),
        ("flip_needed", {"B": 1, "C": 2, "D": 3, "A": 4}, {"A", "B"}, set()),
        ("partial_held", {"A": 1, "B": 2, "C": 3, "D": 4}, {"A"}, {"D"}),
        ("many_instruments", {chr(65 + i): float(i) for i in range(10)}, {"A", "B"}, {"H", "I"}),
    ]

    @pytest.mark.parametrize("name,scores,cl,cs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_long_short_sets_match(self, name, scores, cl, cs):
        old = _OldFMZSelectionPolicy(n_long=2, n_short=2)
        new = FMZSelectionPolicy(n_long=2, n_short=2)

        old_long, old_short = old.select(scores, cl, cs)
        targets = new.select(scores, cl, cs)

        assert _longs(targets) == old_long, f"long mismatch: {_longs(targets)} != {old_long}"
        assert _shorts(targets) == old_short, f"short mismatch: {_shorts(targets)} != {old_short}"


# ---------------------------------------------------------------------------
# 2. TopKDropout Selection Equivalence
# ---------------------------------------------------------------------------

class _OldTopKDropoutSelectionPolicy:
    """Original TopKDropout implementation for comparison."""

    def __init__(self, topk_long, topk_short, n_drop_long, n_drop_short):
        self._topk_long = topk_long
        self._topk_short = topk_short
        self._n_drop_long = n_drop_long
        self._n_drop_short = n_drop_short

    def select(self, scores, current_long, current_short):
        final_long = self._select_leg(scores, current_long, self._topk_long, self._n_drop_long, ascending=False)
        final_short = self._select_leg(scores, current_short, self._topk_short, self._n_drop_short, ascending=True, exclude=final_long)
        return final_long, final_short

    @staticmethod
    def _select_leg(scores, current_held, topk, n_drop, ascending=False, exclude=None):
        exclude = exclude or set()
        available = {k: v for k, v in scores.items() if k not in exclude}
        sorted_all = sorted(available.items(), key=lambda x: (x[1], x[0]), reverse=not ascending)
        last = [s for s, _ in sorted_all if s in current_held]
        not_held = [s for s, _ in sorted_all if s not in current_held]
        n_to_add = max(0, n_drop + topk - len(last))
        candidates = not_held[:n_to_add]
        combined_set = set(last) | set(candidates)
        combined = [s for s, _ in sorted_all if s in combined_set]
        tail = combined[-n_drop:] if n_drop > 0 else []
        to_drop = set(s for s in tail if s in current_held)
        n_buy = len(to_drop) + topk - len(last)
        to_add = candidates[:max(0, n_buy)]
        excess_over_topk = max(0, len(last) - topk)
        max_effective_drops = len(to_add) + excess_over_topk
        if len(to_drop) > max_effective_drops:
            drop_list = [s for s in combined if s in to_drop]
            to_drop = set(drop_list[-max_effective_drops:]) if max_effective_drops > 0 else set()
        return (set(last) - to_drop) | set(to_add)


class TestTopKDropoutEquivalence:
    SCENARIOS = [
        ("fresh", {"A": 1, "B": 2, "C": 3, "D": 4}, set(), set()),
        ("stable", {"A": 1, "B": 2, "C": 3, "D": 4}, {"C", "D"}, {"A", "B"}),
        ("rotation", {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, {"D", "E"}, {"A", "B"}),
        ("partial", {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, {"C"}, {"A"}),
    ]

    @pytest.mark.parametrize("name,scores,cl,cs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_long_short_sets_match(self, name, scores, cl, cs):
        old = _OldTopKDropoutSelectionPolicy(topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1)
        new = TopKDropoutSelectionPolicy(topk_long=2, topk_short=2, n_drop_long=1, n_drop_short=1)

        old_long, old_short = old.select(scores, cl, cs)
        targets = new.select(scores, cl, cs)

        assert _longs(targets) == old_long
        assert _shorts(targets) == old_short


# ---------------------------------------------------------------------------
# 3. DecisionEngine Orders Equivalence (fixed mode)
# ---------------------------------------------------------------------------

class _OldDecisionEngineOrders:
    """Reproduce old _compute_orders logic for comparison."""

    def __init__(self, n_long, n_short, position_value):
        self._n_long = n_long
        self._n_short = n_short
        self._position_value = position_value
        self._policy = _OldFMZSelectionPolicy(n_long, n_short)

    def compute_orders(self, composite, current_long, current_short):
        orders = []
        instruments_with_data = set(composite.keys())
        for inst_id in sorted(current_long - instruments_with_data):
            orders.append({"instrument_id": inst_id, "order_side": "SELL", "target_quote_quantity": 0, "tags": ["NO_FACTOR_DATA"]})
        for inst_id in sorted(current_short - instruments_with_data):
            orders.append({"instrument_id": inst_id, "order_side": "BUY", "target_quote_quantity": 0, "tags": ["NO_FACTOR_DATA"]})

        final_long, final_short = self._policy.select(composite, current_long & instruments_with_data, current_short & instruments_with_data)
        sorted_symbols = sorted(composite.items(), key=lambda x: (x[1], x[0]))
        rank_lookup = {s: i for i, (s, _) in enumerate(sorted_symbols)}

        for inst_id in sorted(instruments_with_data):
            in_fl = inst_id in final_long
            in_fs = inst_id in final_short
            was_long = inst_id in current_long
            was_short = inst_id in current_short
            rank = rank_lookup.get(inst_id, -1)

            if in_fl:
                if was_long:
                    continue  # HOLD in fixed mode
                tag = "FLIP_TO_LONG" if was_short else "NEW_LONG"
                orders.append({"instrument_id": inst_id, "order_side": "BUY", "target_quote_quantity": self._position_value, "tags": [tag, f"rank:{rank}"]})
            elif in_fs:
                if was_short:
                    continue  # HOLD in fixed mode
                tag = "FLIP_TO_SHORT" if was_long else "NEW_SHORT"
                orders.append({"instrument_id": inst_id, "order_side": "SELL", "target_quote_quantity": self._position_value, "tags": [tag, f"rank:{rank}"]})
            else:
                if was_long:
                    orders.append({"instrument_id": inst_id, "order_side": "SELL", "target_quote_quantity": 0, "tags": ["DROPPED_LONG", f"rank:{rank}"]})
                elif was_short:
                    orders.append({"instrument_id": inst_id, "order_side": "BUY", "target_quote_quantity": 0, "tags": ["DROPPED_SHORT", f"rank:{rank}"]})
        return orders


class TestDecisionEngineOrdersEquivalence:
    """New DecisionEngine (position_mode=fixed) produces same orders as old version."""

    SCENARIOS = [
        ("fresh", {"A": 1, "B": 2, "C": 3, "D": 4}, set(), set()),
        ("hold", {"A": 1, "B": 2, "C": 3, "D": 4}, {"A", "B"}, {"C", "D"}),
        ("flip", {"D": 0.5, "A": 1, "B": 3, "C": 4}, {"A"}, {"C", "D"}),
        ("delist", {"A": 1, "B": 2, "C": 3, "D": 4}, {"A", "GONE"}, set()),
        ("mixed", {"B": 1, "C": 2, "D": 3, "A": 4}, {"A", "B"}, {"D"}),
    ]

    @pytest.mark.parametrize("name,comp,cl,cs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_orders_match(self, name, comp, cl, cs):
        old = _OldDecisionEngineOrders(n_long=2, n_short=2, position_value=500.0)
        old_orders = old.compute_orders(comp, cl, cs)

        new_actor = DecisionEngineActor(DecisionEngineActorConfig(
            n_long=2, n_short=2, position_value=500.0, position_mode="fixed",
        ))
        new_orders = new_actor._compute_orders(comp, cl, cs)

        # Compare essential fields: instrument_id, order_side, target_quote_quantity
        def _key(o):
            return (o["instrument_id"], o["order_side"], o["target_quote_quantity"])

        old_set = {_key(o) for o in old_orders}
        new_set = {_key(o) for o in new_orders}
        assert new_set == old_set, f"Orders differ:\n  old: {old_set}\n  new: {new_set}"


# ---------------------------------------------------------------------------
# 4. WorldQuant Pipeline Numerical Equivalence
# ---------------------------------------------------------------------------

class _OldWorldQuantPipeline:
    """Reproduce WorldQuantAlphaStrategy pipeline methods for comparison."""

    def __init__(self, delay=1, decay=0, neutralization="MARKET", truncation=0.0):
        self.delay = delay
        self.decay = decay
        self.neutralization = neutralization
        self.truncation = truncation
        self._prev_alpha = None
        self._alpha_history = []

    def apply_delay(self, raw):
        if self.delay == 0:
            return raw
        prev = self._prev_alpha
        self._prev_alpha = raw
        return prev

    def process_alpha(self, alpha):
        alpha = {k: v for k, v in alpha.items() if not math.isnan(v)}
        if not alpha:
            return {}
        if self.neutralization == "MARKET":
            alpha = self._neutralize(alpha)
        alpha = self._scale(alpha)
        alpha = self._apply_decay(alpha)
        alpha = self._scale(alpha)
        if self.truncation > 0:
            for _ in range(20):
                truncated = self._truncate(alpha)
                alpha = self._scale(truncated)
                if max(abs(v) for v in alpha.values()) <= self.truncation + 1e-10:
                    break
        return alpha

    @staticmethod
    def _neutralize(alpha):
        valid = {k: v for k, v in alpha.items() if not math.isnan(v)}
        if not valid:
            return alpha
        mean = sum(valid.values()) / len(valid)
        result = {k: v - mean for k, v in valid.items()}
        for k, v in alpha.items():
            if math.isnan(v):
                result[k] = v
        return result

    @staticmethod
    def _scale(alpha):
        valid = {k: v for k, v in alpha.items() if not math.isnan(v)}
        total_abs = sum(abs(v) for v in valid.values())
        if total_abs == 0:
            return alpha
        return {k: v / total_abs for k, v in valid.items()}

    def _apply_decay(self, alpha):
        if self.decay == 0:
            return alpha
        self._alpha_history.append(alpha)
        if len(self._alpha_history) > self.decay:
            self._alpha_history.pop(0)
        n = len(self._alpha_history)
        weights = list(range(1, n + 1))
        total_weight = sum(weights)
        all_keys = set()
        for hist in self._alpha_history:
            all_keys.update(hist.keys())
        result = {}
        for key in sorted(all_keys):
            weighted_sum = sum(w * hist.get(key, 0.0) for w, hist in zip(weights, self._alpha_history))
            result[key] = weighted_sum / total_weight
        return result

    def _truncate(self, alpha):
        t = self.truncation
        return {k: max(-t, min(t, v)) for k, v in alpha.items()}


class TestWorldQuantPipelineEquivalence:
    """WorldQuantSelectionPolicy produces same weights as original pipeline."""

    def test_basic_neutralize_scale(self):
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        old = _OldWorldQuantPipeline(delay=0)
        new = WorldQuantSelectionPolicy(delay=0)

        old_weights = old.process_alpha(dict(scores))
        targets = new.select(scores, set(), set())
        new_weights = {t.symbol: t.weight for t in targets}

        # Compare non-zero weights (zero = no position, correctly excluded by new policy)
        for k, v in old_weights.items():
            if abs(v) > 1e-8:
                assert new_weights.get(k) == pytest.approx(v, abs=1e-10), \
                    f"{k}: old={v}, new={new_weights.get(k)}"
            else:
                assert k not in new_weights, f"{k} has zero weight but still in targets"

    def test_with_delay(self):
        old = _OldWorldQuantPipeline(delay=1)
        new = WorldQuantSelectionPolicy(delay=1)

        scores_t1 = {"A": 1.0, "B": 3.0, "C": 5.0}
        scores_t2 = {"A": 2.0, "B": 4.0, "C": 6.0}

        # Warmup
        old.apply_delay(scores_t1)
        new.select(scores_t1, set(), set())

        # Second call uses t1 data
        old_alpha = old.apply_delay(scores_t2)
        old_weights = old.process_alpha(dict(old_alpha))
        targets = new.select(scores_t2, set(), set())
        new_weights = {t.symbol: t.weight for t in targets}

        for k, v in old_weights.items():
            if abs(v) > 1e-8:
                assert new_weights.get(k) == pytest.approx(v, abs=1e-10)
            else:
                assert k not in new_weights

    def test_with_decay(self):
        old = _OldWorldQuantPipeline(delay=0, decay=3)
        new = WorldQuantSelectionPolicy(delay=0, decay=3)

        periods = [
            {"A": 1.0, "B": 2.0, "C": 3.0},
            {"A": 3.0, "B": 1.0, "C": 2.0},
            {"A": 2.0, "B": 3.0, "C": 1.0},
        ]

        for scores in periods:
            old_weights = old.process_alpha(dict(scores))
            targets = new.select(scores, set(), set())

        new_weights = {t.symbol: t.weight for t in targets}
        for k in old_weights:
            assert new_weights.get(k) == pytest.approx(old_weights[k], abs=1e-10)

    def test_with_truncation(self):
        old = _OldWorldQuantPipeline(delay=0, truncation=0.1)
        new = WorldQuantSelectionPolicy(delay=0, truncation=0.1)

        scores = {"A": 10.0, "B": 1.0, "C": 2.0, "D": 3.0}
        old_weights = old.process_alpha(dict(scores))
        targets = new.select(scores, set(), set())
        new_weights = {t.symbol: t.weight for t in targets}

        for k in old_weights:
            assert new_weights.get(k) == pytest.approx(old_weights[k], abs=1e-10)

    def test_with_nan(self):
        old = _OldWorldQuantPipeline(delay=0)
        new = WorldQuantSelectionPolicy(delay=0)

        scores = {"A": 1.0, "B": float("nan"), "C": 3.0}
        old_weights = old.process_alpha(dict(scores))
        targets = new.select(scores, set(), set())
        new_weights = {t.symbol: t.weight for t in targets}

        for k in old_weights:
            if not math.isnan(old_weights[k]):
                assert new_weights.get(k) == pytest.approx(old_weights[k], abs=1e-10)

    def test_full_pipeline_10_instruments(self):
        """Larger instrument set with all pipeline steps."""
        old = _OldWorldQuantPipeline(delay=0, decay=2, truncation=0.15)
        new = WorldQuantSelectionPolicy(delay=0, decay=2, truncation=0.15)

        periods = [
            {f"S{i}": float(i * 2 + j) for i in range(10)}
            for j in range(3)
        ]

        for scores in periods:
            old_weights = old.process_alpha(dict(scores))
            targets = new.select(scores, set(), set())

        new_weights = {t.symbol: t.weight for t in targets}
        for k in old_weights:
            if not math.isnan(old_weights.get(k, 0)):
                assert new_weights.get(k) == pytest.approx(old_weights[k], abs=1e-10), \
                    f"{k}: old={old_weights[k]}, new={new_weights.get(k)}"
