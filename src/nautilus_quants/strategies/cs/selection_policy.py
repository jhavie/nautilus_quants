# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
SelectionPolicy — pluggable instrument selection for DecisionEngineActor.

Implementations:
- FMZSelectionPolicy: sort + bottom-N long / top-N short, sticky hold
- TopKDropoutSelectionPolicy: active rotation with n_drop per leg
"""

from __future__ import annotations

from typing import Protocol


class SelectionPolicy(Protocol):
    """Protocol for instrument selection algorithms.

    Returns the desired final portfolio state as two sets.
    DecisionEngineActor diffs against current positions to produce intents.
    """

    def select(
        self,
        scores: dict[str, float],
        current_long: set[str],
        current_short: set[str],
    ) -> tuple[set[str], set[str]]:
        """
        Select instruments for long and short legs.

        Parameters
        ----------
        scores : dict[str, float]
            Composite factor values keyed by instrument ID.
        current_long : set[str]
            Currently held long positions (with factor data).
        current_short : set[str]
            Currently held short positions (with factor data).

        Returns
        -------
        tuple[set[str], set[str]]
            (final_long_set, final_short_set)
        """
        ...


class FMZSelectionPolicy:
    """Sort + bottom-N long / top-N short. Sticky: only flip triggers action.

    Instruments that drop out of target range but don't flip to the opposite
    side remain in their current leg (no active rotation).
    """

    def __init__(self, n_long: int, n_short: int) -> None:
        self._n_long = n_long
        self._n_short = n_short

    def select(
        self,
        scores: dict[str, float],
        current_long: set[str],
        current_short: set[str],
    ) -> tuple[set[str], set[str]]:
        sorted_symbols = sorted(scores.items(), key=lambda x: (x[1], x[0]))
        long_targets = set(s for s, _ in sorted_symbols[: self._n_long])
        short_targets = set(s for s, _ in sorted_symbols[-self._n_short :])
        # Sticky: keep current positions unless they should flip
        final_long = (current_long - short_targets) | long_targets
        final_short = (current_short - long_targets) | short_targets
        return final_long, final_short


class TopKDropoutSelectionPolicy:
    """Active rotation: drop worst n_drop per leg, replace with top candidates.

    Each leg maintains an independent TopK pool. Every rebalance, the worst
    n_drop instruments in the pool are ejected and replaced by the best
    candidates not currently in the pool.

    Ported from qlib LongShortTopKStrategy (incremental mode, no hold_thresh).
    """

    def __init__(
        self,
        topk_long: int,
        topk_short: int,
        n_drop_long: int,
        n_drop_short: int,
    ) -> None:
        self._topk_long = topk_long
        self._topk_short = topk_short
        self._n_drop_long = n_drop_long
        self._n_drop_short = n_drop_short

    def select(
        self,
        scores: dict[str, float],
        current_long: set[str],
        current_short: set[str],
    ) -> tuple[set[str], set[str]]:
        final_long = self._select_leg(
            scores,
            current_long,
            self._topk_long,
            self._n_drop_long,
            ascending=False,
        )
        final_short = self._select_leg(
            scores,
            current_short,
            self._topk_short,
            self._n_drop_short,
            ascending=True,
            exclude=final_long,
        )
        return final_long, final_short

    @staticmethod
    def _select_leg(
        scores: dict[str, float],
        current_held: set[str],
        topk: int,
        n_drop: int,
        ascending: bool = False,
        exclude: set[str] | None = None,
    ) -> set[str]:
        """Select one leg via TopK Dropout rotation.

        Parameters
        ----------
        scores : dict[str, float]
            All instrument scores.
        current_held : set[str]
            Currently held instruments in this leg.
        topk : int
            Target pool size.
        n_drop : int
            Max instruments to rotate out per rebalance.
        ascending : bool
            True for short leg (lowest scores best), False for long (highest best).
        exclude : set[str] | None
            Instruments to exclude (e.g., already in the other leg).

        Returns
        -------
        set[str]
            Final set of instruments for this leg.
        """
        exclude = exclude or set()
        available = {k: v for k, v in scores.items() if k not in exclude}

        # Sort: for long leg descending (best=highest), for short ascending (best=lowest)
        sorted_all = sorted(available.items(), key=lambda x: (x[1], x[0]), reverse=not ascending)

        # Current held instruments that still have scores (not delisted)
        last = [s for s, _ in sorted_all if s in current_held]

        # Candidates: instruments not in current held, sorted by quality
        not_held = [s for s, _ in sorted_all if s not in current_held]

        # How many new candidates to consider
        n_to_add = max(0, n_drop + topk - len(last))
        candidates = not_held[:n_to_add]

        # Combine held + candidates, re-sort by quality (best first)
        combined_set = set(last) | set(candidates)
        combined = [s for s, _ in sorted_all if s in combined_set]

        # Identify worst n_drop from held to eject (tail of combined ∩ held)
        tail = combined[-n_drop:] if n_drop > 0 else []
        to_drop = set(s for s in tail if s in current_held)

        # How many to actually add
        n_buy = len(to_drop) + topk - len(last)
        to_add = candidates[: max(0, n_buy)]

        # Guard: don't shrink pool below topk when no candidates can replace drops.
        # Only allow drops that are covered by adds, or that trim an over-full pool.
        excess_over_topk = max(0, len(last) - topk)
        max_effective_drops = len(to_add) + excess_over_topk
        if len(to_drop) > max_effective_drops:
            # Keep the worst (furthest from head in combined) — drop_list is best-first
            drop_list = [s for s in combined if s in to_drop]
            to_drop = set(drop_list[-max_effective_drops:]) if max_effective_drops > 0 else set()

        final = (set(last) - to_drop) | set(to_add)
        return final
