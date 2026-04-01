# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for FactorRepository — CRUD, versioning, status, import, context."""

from __future__ import annotations

import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import (
    AnalysisResult,
    ConfigContext,
    FactorRecord,
)
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.factors.config import FactorConfig, FactorDefinition


@pytest.fixture()
def repo() -> FactorRepository:
    db = RegistryDatabase(":memory:")
    repository = FactorRepository(db)
    yield repository
    db.close()


def _make_record(
    factor_id: str = "alpha001",
    expression: str = "rank(close)",
    **kwargs,
) -> FactorRecord:
    return FactorRecord(factor_id=factor_id, expression=expression, **kwargs)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestUpsertAndGet:
    def test_insert_new(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record())
        f = repo.get_factor("alpha001")
        assert f is not None
        assert f.factor_id == "alpha001"
        assert f.expression == "rank(close)"
        assert f.status == "candidate"

    def test_update_expression_creates_version(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(expression="rank(close)"))
        repo.upsert_factor(_make_record(expression="rank(open)"))

        f = repo.get_factor("alpha001")
        assert f is not None
        assert f.expression == "rank(open)"

        versions = repo.get_versions("alpha001")
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[0].expression == "rank(close)"
        assert versions[1].version == 2
        assert versions[1].expression == "rank(open)"

    def test_update_same_expression_no_new_version(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(expression="rank(close)"))
        repo.upsert_factor(_make_record(expression="rank(close)", description="updated desc"))
        versions = repo.get_versions("alpha001")
        assert len(versions) == 1

    def test_get_nonexistent(self, repo: FactorRepository) -> None:
        assert repo.get_factor("nonexistent") is None

    def test_delete(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record())
        repo.delete_factor("alpha001")
        assert repo.get_factor("alpha001") is None

    def test_delete_nonexistent_no_error(self, repo: FactorRepository) -> None:
        repo.delete_factor("nonexistent")


# ---------------------------------------------------------------------------
# List with filters
# ---------------------------------------------------------------------------


class TestListFactors:
    @pytest.fixture(autouse=True)
    def _seed(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record("a", "e1", category="momentum", source="alpha101", status="active"))
        repo.upsert_factor(_make_record("b", "e2", category="volume", source="alpha101", status="candidate"))
        repo.upsert_factor(_make_record("c", "e3", category="momentum", source="fmz", status="archived"))
        repo.upsert_factor(_make_record("d", "e4", category="volatility", source="mined", status="active"))

    def test_list_all(self, repo: FactorRepository) -> None:
        assert len(repo.list_factors()) == 4

    def test_filter_status(self, repo: FactorRepository) -> None:
        result = repo.list_factors(status="active")
        assert {f.factor_id for f in result} == {"a", "d"}

    def test_filter_category(self, repo: FactorRepository) -> None:
        result = repo.list_factors(category="momentum")
        assert {f.factor_id for f in result} == {"a", "c"}

    def test_filter_source(self, repo: FactorRepository) -> None:
        result = repo.list_factors(source="alpha101")
        assert {f.factor_id for f in result} == {"a", "b"}

    def test_combined_filters(self, repo: FactorRepository) -> None:
        result = repo.list_factors(status="active", category="momentum")
        assert [f.factor_id for f in result] == ["a"]

    def test_sort_by(self, repo: FactorRepository) -> None:
        result = repo.list_factors(sort_by="factor_id")
        assert [f.factor_id for f in result] == ["a", "b", "c", "d"]

    def test_limit(self, repo: FactorRepository) -> None:
        result = repo.list_factors(sort_by="factor_id", limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestSetStatus:
    def test_candidate_to_active(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(status="candidate"))
        repo.set_status("alpha001", "active")
        assert repo.get_factor("alpha001").status == "active"

    def test_active_to_archived(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(status="active"))
        repo.set_status("alpha001", "archived")
        assert repo.get_factor("alpha001").status == "archived"

    def test_archived_to_candidate(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(status="archived"))
        repo.set_status("alpha001", "candidate")
        assert repo.get_factor("alpha001").status == "candidate"

    def test_invalid_transition_raises(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(status="candidate"))
        with pytest.raises(ValueError, match="非法状态转换"):
            repo.set_status("alpha001", "archived")

    def test_nonexistent_factor_raises(self, repo: FactorRepository) -> None:
        with pytest.raises(ValueError, match="因子不存在"):
            repo.set_status("nonexistent", "active")


# ---------------------------------------------------------------------------
# Import from FactorConfig
# ---------------------------------------------------------------------------


class TestImportFromConfig:
    def test_import_counts(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            name="test",
            version="1.0",
            factors=[
                FactorDefinition("f1", "rank(close)", category="momentum"),
                FactorDefinition("f2", "ts_std(close, 24)", category="volatility"),
            ],
            variables={"returns": "delta(close, 1) / delay(close, 1)"},
            parameters={"window": 24},
        )
        new, updated, unchanged = repo.import_from_config(config, source="test_src")
        assert new == 2
        assert updated == 0
        assert unchanged == 0

    def test_import_preserves_source(self, repo: FactorRepository) -> None:
        config = FactorConfig(factors=[FactorDefinition("f1", "rank(close)")])
        repo.import_from_config(config, source="alpha101")
        f = repo.get_factor("f1")
        assert f.source == "alpha101"

    def test_import_preserves_category(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            factors=[FactorDefinition("f1", "rank(close)", category="momentum")]
        )
        repo.import_from_config(config)
        assert repo.get_factor("f1").category == "momentum"

    def test_import_detects_update(self, repo: FactorRepository) -> None:
        config1 = FactorConfig(factors=[FactorDefinition("f1", "rank(close)")])
        repo.import_from_config(config1)

        config2 = FactorConfig(factors=[FactorDefinition("f1", "rank(open)")])
        new, updated, unchanged = repo.import_from_config(config2)
        assert new == 0
        assert updated == 1
        assert unchanged == 0

    def test_import_detects_unchanged(self, repo: FactorRepository) -> None:
        config = FactorConfig(factors=[FactorDefinition("f1", "rank(close)")])
        repo.import_from_config(config)
        new, updated, unchanged = repo.import_from_config(config)
        assert unchanged == 1

    def test_import_stores_context(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            name="fmz_4factor",
            variables={"ret": "delta(close, 1)"},
            parameters={"w": 24},
        )
        repo.import_from_config(config, context_id="fmz_4factor")
        ctx = repo.get_context("fmz_4factor")
        assert ctx is not None
        assert ctx.variables == {"ret": "delta(close, 1)"}
        assert ctx.parameters == {"w": 24}

    def test_import_default_context_id(self, repo: FactorRepository) -> None:
        """context_id defaults to config.name."""
        config = FactorConfig(name="my_config", factors=[FactorDefinition("f1", "close")])
        repo.import_from_config(config)
        ctx = repo.get_context("my_config")
        assert ctx is not None


# ---------------------------------------------------------------------------
# Config context
# ---------------------------------------------------------------------------


class TestConfigContext:
    def test_upsert_and_get(self, repo: FactorRepository) -> None:
        ctx = ConfigContext(
            context_id="ctx1",
            variables={"v": "expr"},
            parameters={"p": 42},
            metadata={"name": "test"},
        )
        repo.upsert_context(ctx)
        loaded = repo.get_context("ctx1")
        assert loaded is not None
        assert loaded.variables == {"v": "expr"}
        assert loaded.parameters == {"p": 42}
        assert loaded.metadata == {"name": "test"}

    def test_upsert_overwrites(self, repo: FactorRepository) -> None:
        repo.upsert_context(ConfigContext("ctx1", variables={"a": "1"}))
        repo.upsert_context(ConfigContext("ctx1", variables={"b": "2"}))
        loaded = repo.get_context("ctx1")
        assert loaded.variables == {"b": "2"}

    def test_get_nonexistent(self, repo: FactorRepository) -> None:
        assert repo.get_context("nonexistent") is None

    def test_list_contexts(self, repo: FactorRepository) -> None:
        repo.upsert_context(ConfigContext("a"))
        repo.upsert_context(ConfigContext("b"))
        ctxs = repo.list_contexts()
        assert len(ctxs) == 2


# ---------------------------------------------------------------------------
# Analysis results (interface ready for 035)
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_save_and_get(self, repo: FactorRepository) -> None:
        results = [
            AnalysisResult("f1", "4h", 1, ic_mean=0.05, icir=0.3, analyzed_at="2026-01-01"),
            AnalysisResult("f1", "4h", 4, ic_mean=0.03, icir=0.2, analyzed_at="2026-01-01"),
        ]
        repo.save_analysis(results)
        loaded = repo.get_analysis("f1", "4h")
        assert len(loaded) == 2
        assert loaded[0].period == 1
        assert loaded[0].ic_mean == 0.05

    def test_get_empty(self, repo: FactorRepository) -> None:
        assert repo.get_analysis("nonexistent", "4h") == []
