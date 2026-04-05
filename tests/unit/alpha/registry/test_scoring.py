# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for scoring module — greedy_select and retire_evicted_factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.alpha.registry.scoring import (
    greedy_select,
    retire_evicted_factors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scores(ids: list[str], scores: list[float]) -> pd.DataFrame:
    """Create a scores DataFrame indexed by factor_id."""
    return pd.DataFrame(
        {"final_score": scores},
        index=pd.Index(ids, name="factor_id"),
    )


def _make_corr(ids: list[str], matrix: list[list[float]]) -> pd.DataFrame:
    """Create a correlation matrix DataFrame."""
    return pd.DataFrame(matrix, index=ids, columns=ids)


@pytest.fixture()
def target_db() -> RegistryDatabase:
    db = RegistryDatabase(":memory:")
    yield db
    db.close()


@pytest.fixture()
def target_repo(target_db: RegistryDatabase) -> FactorRepository:
    return FactorRepository(target_db)


def _seed_active_factors(
    repo: FactorRepository, factor_ids: list[str],
) -> None:
    """Insert factors with status='active' into the repo."""
    for fid in factor_ids:
        record = FactorRecord(
            factor_id=fid,
            expression=f"rank({fid})",
            status="active",
            source="test",
        )
        repo.upsert_factor(record)
        # Force active status
        repo._db.execute(
            "UPDATE factors SET status = 'active' WHERE factor_id = ?",
            [fid],
        )


# ---------------------------------------------------------------------------
# greedy_select — additive (gatekeeper) mode
# ---------------------------------------------------------------------------


class TestGreedySelectAdditive:
    """Test greedy_select with existing_ids (additive/gatekeeper mode)."""

    def test_existing_blocks_correlated_candidate(self) -> None:
        """Existing factors prevent correlated candidates from entering."""
        ids = ["E1", "C1", "C2"]
        scores = _make_scores(["C1", "C2"], [0.9, 0.8])
        corr = _make_corr(ids, [
            [1.0, 0.6, 0.1],  # E1 correlated with C1, not C2
            [0.6, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])

        result = greedy_select(
            scores, corr, max_corr=0.5, max_factors=10,
            existing_ids=["E1"],
        )

        assert "C1" not in result  # blocked by E1
        assert "C2" in result      # passes

    def test_existing_not_returned(self) -> None:
        """Existing factors are never in the returned list."""
        ids = ["E1", "C1"]
        scores = _make_scores(["C1"], [0.9])
        corr = _make_corr(ids, [
            [1.0, 0.1],
            [0.1, 1.0],
        ])

        result = greedy_select(
            scores, corr, max_corr=0.5, max_factors=10,
            existing_ids=["E1"],
        )

        assert "E1" not in result
        assert "C1" in result


# ---------------------------------------------------------------------------
# greedy_select — competitive (no gatekeeper) mode
# ---------------------------------------------------------------------------


class TestGreedySelectCompetitive:
    """Test greedy_select WITHOUT existing_ids (competitive mode)."""

    def test_higher_score_wins(self) -> None:
        """Higher-scored factor is selected; lower-scored correlated one is not."""
        ids = ["F1", "F2", "F3"]
        scores = _make_scores(ids, [0.9, 0.8, 0.5])
        corr = _make_corr(ids, [
            [1.0, 0.7, 0.1],  # F1 and F2 are correlated
            [0.7, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])

        result = greedy_select(scores, corr, max_corr=0.5, max_factors=10)

        assert result == ["F1", "F3"]  # F2 blocked by F1

    def test_no_gatekeepers_allows_better_set(self) -> None:
        """Without gatekeepers, a globally better set can be selected."""
        # Scenario: E1 (existing, score=0.5) blocks C1 (score=0.9) in additive
        # In competitive, C1 wins because E1 isn't pre-seeded
        ids = ["C1", "C2", "E1"]
        scores = _make_scores(ids, [0.9, 0.7, 0.5])
        corr = _make_corr(ids, [
            [1.0, 0.1, 0.6],  # C1 and E1 are correlated
            [0.1, 1.0, 0.1],
            [0.6, 0.1, 1.0],
        ])

        # Additive: E1 gatekeeper blocks C1
        additive = greedy_select(
            scores, corr, max_corr=0.5, max_factors=10,
            existing_ids=["E1"],
        )
        assert "C1" not in additive

        # Competitive: C1 wins (higher score), E1 would be blocked
        competitive = greedy_select(
            scores, corr, max_corr=0.5, max_factors=10,
        )
        assert "C1" in competitive
        assert "E1" not in competitive  # blocked by C1

    def test_max_factors_respected(self) -> None:
        """max_factors limits the selection count."""
        ids = ["F1", "F2", "F3"]
        scores = _make_scores(ids, [0.9, 0.8, 0.7])
        corr = _make_corr(ids, [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        result = greedy_select(scores, corr, max_corr=0.5, max_factors=2)

        assert len(result) == 2
        assert result == ["F1", "F2"]


# ---------------------------------------------------------------------------
# retire_evicted_factors
# ---------------------------------------------------------------------------


class TestRetireEvictedFactors:
    """Test archiving of factors evicted in competitive mode."""

    def test_evicts_non_selected(
        self, target_db: RegistryDatabase, target_repo: FactorRepository,
    ) -> None:
        """Factors in existing but not in selected are archived."""
        _seed_active_factors(target_repo, ["F1", "F2", "F3"])

        evicted = retire_evicted_factors(
            target_db,
            selected_ids=["F1", "F3"],  # F2 not selected
            existing_ids=["F1", "F2", "F3"],
        )

        assert evicted == ["F2"]
        f2 = target_repo.get_factor("F2")
        assert f2 is not None
        assert f2.status == "archived"

        # F1 and F3 remain active
        assert target_repo.get_factor("F1").status == "active"
        assert target_repo.get_factor("F3").status == "active"

    def test_evicts_nothing_when_all_selected(
        self, target_db: RegistryDatabase, target_repo: FactorRepository,
    ) -> None:
        """No evictions when all existing factors are re-selected."""
        _seed_active_factors(target_repo, ["F1", "F2"])

        evicted = retire_evicted_factors(
            target_db,
            selected_ids=["F1", "F2", "F3"],
            existing_ids=["F1", "F2"],
        )

        assert evicted == []
        assert target_repo.get_factor("F1").status == "active"
        assert target_repo.get_factor("F2").status == "active"

    def test_evicts_all_when_none_selected(
        self, target_db: RegistryDatabase, target_repo: FactorRepository,
    ) -> None:
        """All existing factors are evicted when none are re-selected."""
        _seed_active_factors(target_repo, ["F1", "F2"])

        evicted = retire_evicted_factors(
            target_db,
            selected_ids=["F3", "F4"],
            existing_ids=["F1", "F2"],
        )

        assert set(evicted) == {"F1", "F2"}
        assert target_repo.get_factor("F1").status == "archived"
        assert target_repo.get_factor("F2").status == "archived"

    def test_empty_existing_no_crash(
        self, target_db: RegistryDatabase,
    ) -> None:
        """No crash when existing_ids is empty."""
        evicted = retire_evicted_factors(
            target_db, selected_ids=["F1"], existing_ids=[],
        )
        assert evicted == []
