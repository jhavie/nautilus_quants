# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for conflict detection and repair in the factor registry."""

from __future__ import annotations

import json

import pytest

from nautilus_quants.alpha.registry.audit import (
    find_conflicting_factors,
    repair_factors,
)
from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import AnalysisMetrics, FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository


@pytest.fixture()
def db() -> RegistryDatabase:
    database = RegistryDatabase(":memory:")
    yield database
    database.close()


@pytest.fixture()
def repo(db: RegistryDatabase) -> FactorRepository:
    return FactorRepository(db)


def _setup_conflict(repo: FactorRepository) -> None:
    """Create a factor with metrics from two conflicting expressions."""
    # Register factor with expression A
    repo.upsert_factor(FactorRecord(
        factor_id="src_f1",
        expression="rank(close)",
        source="src",
    ))

    # Save config snapshot A
    cfg_a = {
        "factors": {"f1": {"expression": "rank(close)", "tags": []}},
        "metadata": {"source": "src"},
    }
    cfg_id_a = repo.save_config_snapshot(cfg_a, "factors", config_name="a")

    # Save metrics for expression A
    repo.save_metrics([
        AnalysisMetrics(
            run_id="run_a", factor_id="src_f1", period="1d",
            ic_mean=0.05, icir=0.3,
            factor_config_id=cfg_id_a,
        ),
    ])

    # Now simulate the bug: update factor expression to B
    repo.upsert_factor(FactorRecord(
        factor_id="src_f1",
        expression="rank(open)",
        source="src",
    ))

    # Save config snapshot B
    cfg_b = {
        "factors": {"f1": {"expression": "rank(open)", "tags": []}},
        "metadata": {"source": "src"},
    }
    cfg_id_b = repo.save_config_snapshot(cfg_b, "factors", config_name="b")

    # Save metrics for expression B
    repo.save_metrics([
        AnalysisMetrics(
            run_id="run_b", factor_id="src_f1", period="1d",
            ic_mean=-0.05, icir=-0.3,
            factor_config_id=cfg_id_b,
        ),
    ])


class TestFindConflicts:
    def test_detects_mismatch(self, repo: FactorRepository) -> None:
        _setup_conflict(repo)
        conflicts = find_conflicting_factors(repo)
        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.factor_id == "src_f1"
        assert len(c.orphan_groups) == 1
        assert c.orphan_groups[0].run_ids == ["run_a"]

    def test_no_conflicts_returns_empty(self, repo: FactorRepository) -> None:
        repo.upsert_factor(FactorRecord(
            factor_id="f1", expression="rank(close)",
        ))
        cfg = {"factors": {"f1": {"expression": "rank(close)"}}}
        cfg_id = repo.save_config_snapshot(cfg, "factors")
        repo.save_metrics([
            AnalysisMetrics(
                run_id="r1", factor_id="f1", period="1d",
                factor_config_id=cfg_id,
            ),
        ])
        assert find_conflicting_factors(repo) == []


class TestRepairSplit:
    def test_creates_new_factor_and_reassigns_metrics(
        self, repo: FactorRepository,
    ) -> None:
        _setup_conflict(repo)
        actions = repair_factors(repo, dry_run=False)

        assert len(actions) == 1
        a = actions[0]
        assert a.action == "split"
        assert a.factor_id == "src_f1"
        assert a.new_factor_id.startswith("src_f1_")

        # Original factor keeps expression B metrics
        original_metrics = repo.get_metrics("src_f1")
        assert len(original_metrics) == 1
        assert original_metrics[0].run_id == "run_b"

        # New factor has expression A metrics
        new_metrics = repo.get_metrics(a.new_factor_id)
        assert len(new_metrics) == 1
        assert new_metrics[0].run_id == "run_a"

        # New factor exists with correct expression
        new_factor = repo.get_factor(a.new_factor_id)
        assert new_factor is not None
        assert new_factor.expression == "rank(close)"


class TestRepairMerge:
    def test_merges_into_existing_factor(
        self, repo: FactorRepository,
    ) -> None:
        """When orphaned expression matches an existing factor, merge."""
        _setup_conflict(repo)
        # Create another factor with the same expression as orphan (rank(close))
        repo.upsert_factor(FactorRecord(
            factor_id="src_alias", expression="rank(close)", source="src",
        ))

        actions = repair_factors(repo, dry_run=False)

        assert len(actions) == 1
        a = actions[0]
        assert a.action == "merge"
        assert a.new_factor_id == "src_alias"

        # Metrics reassigned to existing factor
        merged_metrics = repo.get_metrics("src_alias")
        assert len(merged_metrics) == 1
        assert merged_metrics[0].run_id == "run_a"

        # No new factor created with hash suffix
        factors = repo.list_factors()
        hash_suffixed = [
            f for f in factors if f.factor_id.startswith("src_f1_")
        ]
        assert len(hash_suffixed) == 0


class TestRepairDryRun:
    def test_no_changes(self, repo: FactorRepository) -> None:
        _setup_conflict(repo)
        actions = repair_factors(repo, dry_run=True)
        assert len(actions) == 1

        # Nothing should have changed
        metrics = repo.get_metrics("src_f1")
        assert len(metrics) == 2  # Both still there
        assert repo.get_factor(actions[0].new_factor_id) is None
