# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for FactorRepository — CRUD, config snapshots, status, metrics."""

from __future__ import annotations

import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import (
    AnalysisMetrics,
    ConfigSnapshot,
    FactorRecord,
)
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.factors.config import FactorConfig, FactorDefinition


@pytest.fixture()
def db() -> RegistryDatabase:
    database = RegistryDatabase(":memory:")
    yield database
    database.close()


@pytest.fixture()
def repo(db: RegistryDatabase) -> FactorRepository:
    return FactorRepository(db)


def _make_record(
    factor_id: str = "alpha001",
    expression: str = "rank(close)",
    **kwargs,
) -> FactorRecord:
    return FactorRecord(factor_id=factor_id, expression=expression, **kwargs)


def _make_metrics(
    run_id: str = "run1",
    factor_id: str = "f1",
    period: str = "1",
    **kwargs,
) -> AnalysisMetrics:
    return AnalysisMetrics(run_id=run_id, factor_id=factor_id, period=period, **kwargs)


# ---------------------------------------------------------------------------
# Factor CRUD
# ---------------------------------------------------------------------------


class TestUpsertAndGet:
    def test_insert_new(self, repo: FactorRepository) -> None:
        result = repo.upsert_factor(_make_record())
        assert result == "new"
        f = repo.get_factor("alpha001")
        assert f is not None
        assert f.factor_id == "alpha001"
        assert f.expression == "rank(close)"
        assert f.status == "candidate"

    def test_insert_with_all_fields(self, repo: FactorRepository) -> None:
        record = _make_record(
            prototype="rank_family",
            description="Test factor",
            source="alpha101",
            tags=["reversal", "volume"],
            parameters={"window": 24},
            variables={"ret": "delta(close, 1)"},
        )
        repo.upsert_factor(record)
        f = repo.get_factor("alpha001")
        assert f is not None
        assert f.prototype == "rank_family"
        assert f.description == "Test factor"
        assert f.source == "alpha101"
        assert f.tags == ["reversal", "volume"]
        assert f.parameters == {"window": 24}
        assert f.variables == {"ret": "delta(close, 1)"}

    def test_update_expression(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(expression="rank(close)"))
        result = repo.upsert_factor(_make_record(expression="rank(open)"))
        assert result == "updated"
        f = repo.get_factor("alpha001")
        assert f is not None
        assert f.expression == "rank(open)"

    def test_update_prototype(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(prototype=""))
        result = repo.upsert_factor(_make_record(prototype="momentum_family"))
        assert result == "updated"
        f = repo.get_factor("alpha001")
        assert f.prototype == "momentum_family"

    def test_update_tags(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(tags=["momentum"]))
        result = repo.upsert_factor(_make_record(tags=["momentum", "reversal"]))
        assert result == "updated"
        f = repo.get_factor("alpha001")
        assert f.tags == ["momentum", "reversal"]

    def test_update_parameters(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(parameters={"w": 24}))
        result = repo.upsert_factor(_make_record(parameters={"w": 48}))
        assert result == "updated"
        f = repo.get_factor("alpha001")
        assert f.parameters == {"w": 48}

    def test_update_variables(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(variables={"ret": "close/delay(close,1)"}))
        result = repo.upsert_factor(
            _make_record(variables={"ret": "delta(close,1)/delay(close,1)"}),
        )
        assert result == "updated"

    def test_unchanged_returns_unchanged(self, repo: FactorRepository) -> None:
        record = _make_record(
            expression="rank(close)",
            description="desc",
            prototype="proto",
            tags=["t1"],
            parameters={"p": 1},
            variables={"v": "expr"},
        )
        repo.upsert_factor(record)
        result = repo.upsert_factor(record)
        assert result == "unchanged"

    def test_get_nonexistent(self, repo: FactorRepository) -> None:
        assert repo.get_factor("nonexistent") is None

    def test_delete(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record())
        repo.delete_factor("alpha001")
        assert repo.get_factor("alpha001") is None

    def test_delete_nonexistent_no_error(self, repo: FactorRepository) -> None:
        repo.delete_factor("nonexistent")

    def test_created_at_populated(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record())
        f = repo.get_factor("alpha001")
        assert f is not None
        assert f.created_at != ""

    def test_updated_at_changes_on_update(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(expression="v1"))
        f1 = repo.get_factor("alpha001")
        repo.upsert_factor(_make_record(expression="v2"))
        f2 = repo.get_factor("alpha001")
        assert f2.updated_at >= f1.updated_at

    def test_get_returns_correct_types(self, repo: FactorRepository) -> None:
        """tags is list, parameters is dict, variables is dict."""
        repo.upsert_factor(_make_record(
            tags=["a", "b"],
            parameters={"k": 1},
            variables={"v": "expr"},
        ))
        f = repo.get_factor("alpha001")
        assert isinstance(f.tags, list)
        assert isinstance(f.parameters, dict)
        assert isinstance(f.variables, dict)

    def test_get_empty_json_fields(self, repo: FactorRepository) -> None:
        """Default empty tags/parameters/variables."""
        repo.upsert_factor(_make_record())
        f = repo.get_factor("alpha001")
        assert f.tags == []
        assert f.parameters == {}
        assert f.variables == {}


# ---------------------------------------------------------------------------
# List with filters
# ---------------------------------------------------------------------------


class TestListFactors:
    @pytest.fixture(autouse=True)
    def _seed(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record(
            "a", "e1", prototype="momentum", source="alpha101", status="active",
        ))
        repo.upsert_factor(_make_record(
            "b", "e2", prototype="volume", source="alpha101", status="candidate",
        ))
        repo.upsert_factor(_make_record(
            "c", "e3", prototype="momentum", source="fmz", status="archived",
        ))
        repo.upsert_factor(_make_record(
            "d", "e4", prototype="volatility", source="mined", status="active",
        ))

    def test_list_all(self, repo: FactorRepository) -> None:
        assert len(repo.list_factors()) == 4

    def test_filter_status(self, repo: FactorRepository) -> None:
        result = repo.list_factors(status="active")
        assert {f.factor_id for f in result} == {"a", "d"}

    def test_filter_source(self, repo: FactorRepository) -> None:
        result = repo.list_factors(source="alpha101")
        assert {f.factor_id for f in result} == {"a", "b"}

    def test_filter_prototype(self, repo: FactorRepository) -> None:
        result = repo.list_factors(prototype="momentum")
        assert {f.factor_id for f in result} == {"a", "c"}

    def test_combined_filters(self, repo: FactorRepository) -> None:
        result = repo.list_factors(status="active", prototype="momentum")
        assert [f.factor_id for f in result] == ["a"]

    def test_sort_by(self, repo: FactorRepository) -> None:
        result = repo.list_factors(sort_by="factor_id")
        assert [f.factor_id for f in result] == ["a", "b", "c", "d"]

    def test_limit(self, repo: FactorRepository) -> None:
        result = repo.list_factors(sort_by="factor_id", limit=2)
        assert len(result) == 2

    def test_descending(self, repo: FactorRepository) -> None:
        result = repo.list_factors(sort_by="factor_id", descending=True)
        assert [f.factor_id for f in result] == ["d", "c", "b", "a"]

    def test_invalid_sort_by_defaults_to_factor_id(
        self, repo: FactorRepository,
    ) -> None:
        result = repo.list_factors(sort_by="nonexistent_col")
        assert [f.factor_id for f in result] == ["a", "b", "c", "d"]

    def test_list_empty(self, db: RegistryDatabase) -> None:
        """Empty repo returns empty list."""
        empty_repo = FactorRepository(db)
        # Clear seeded data
        db.execute("DELETE FROM factors")
        assert empty_repo.list_factors() == []


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
        with pytest.raises(ValueError, match="Cannot transition"):
            repo.set_status("alpha001", "archived")

    def test_nonexistent_factor_raises(self, repo: FactorRepository) -> None:
        with pytest.raises(ValueError, match="Factor not found"):
            repo.set_status("nonexistent", "active")

    def test_invalid_status_raises(self, repo: FactorRepository) -> None:
        repo.upsert_factor(_make_record())
        with pytest.raises(ValueError, match="Invalid status"):
            repo.set_status("alpha001", "deleted")


# ---------------------------------------------------------------------------
# Register from FactorConfig
# ---------------------------------------------------------------------------


class TestRegisterFromConfig:
    def test_register_counts(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            source="test_src",
            factors=[
                FactorDefinition("f1", "rank(close)", tags=["momentum"]),
                FactorDefinition("f2", "ts_std(close, 24)", tags=["volatility"]),
            ],
            variables={"returns": "delta(close, 1) / delay(close, 1)"},
            parameters={"window": 24},
        )
        new, updated, unchanged = repo.register_factors_from_config(config)
        assert new == 2
        assert updated == 0
        assert unchanged == 0

    def test_register_preserves_source(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            source="alpha101",
            factors=[FactorDefinition("f1", "rank(close)")],
        )
        repo.register_factors_from_config(config)
        f = repo.get_factor("alpha101_f1")
        assert f is not None
        assert f.source == "alpha101"

    def test_register_generates_factor_id(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            source="alpha101",
            factors=[FactorDefinition("alpha044_8h", "rank(close)")],
        )
        repo.register_factors_from_config(config)
        f = repo.get_factor("alpha101_alpha044_8h")
        assert f is not None

    def test_register_preserves_tags(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            source="src",
            factors=[
                FactorDefinition("f1", "rank(close)", tags=["momentum", "reversal"]),
            ],
        )
        repo.register_factors_from_config(config)
        f = repo.get_factor("src_f1")
        assert f.tags == ["momentum", "reversal"]

    def test_register_preserves_prototype(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            source="src",
            factors=[
                FactorDefinition("f1", "rank(close)", prototype="rank_family"),
            ],
        )
        repo.register_factors_from_config(config)
        f = repo.get_factor("src_f1")
        assert f.prototype == "rank_family"

    def test_register_stores_parameters_and_variables(
        self, repo: FactorRepository,
    ) -> None:
        config = FactorConfig(
            source="src",
            factors=[FactorDefinition("f1", "rank(close)")],
            parameters={"window": 24},
            variables={"ret": "delta(close, 1)"},
        )
        repo.register_factors_from_config(config)
        f = repo.get_factor("src_f1")
        assert f.parameters == {"window": 24}
        assert f.variables == {"ret": "delta(close, 1)"}

    def test_register_detects_update(self, repo: FactorRepository) -> None:
        config1 = FactorConfig(
            source="s",
            factors=[FactorDefinition("f1", "rank(close)")],
        )
        repo.register_factors_from_config(config1)

        config2 = FactorConfig(
            source="s",
            factors=[FactorDefinition("f1", "rank(open)")],
        )
        new, updated, unchanged = repo.register_factors_from_config(config2)
        assert new == 0
        assert updated == 1
        assert unchanged == 0

    def test_register_detects_unchanged(self, repo: FactorRepository) -> None:
        config = FactorConfig(
            source="s",
            factors=[FactorDefinition("f1", "rank(close)")],
        )
        repo.register_factors_from_config(config)
        new, updated, unchanged = repo.register_factors_from_config(config)
        assert unchanged == 1

    def test_register_no_source_prefix(self, repo: FactorRepository) -> None:
        """When source is empty, factor_id = key only."""
        config = FactorConfig(
            source="",
            factors=[FactorDefinition("sma_60", "ts_mean(close, 60)")],
        )
        repo.register_factors_from_config(config)
        f = repo.get_factor("sma_60")
        assert f is not None


# ---------------------------------------------------------------------------
# Config snapshots
# ---------------------------------------------------------------------------


class TestConfigSnapshot:
    def test_save_and_get(self, repo: FactorRepository) -> None:
        config_dict = {"factors": {"f1": {"expression": "rank(close)"}}}
        config_id = repo.save_config_snapshot(
            config_dict, config_type="factors", config_name="test_cfg",
        )
        assert config_id.startswith("factors_")

        snap = repo.get_config_snapshot(config_id)
        assert snap is not None
        assert snap.type == "factors"
        assert snap.config_name == "test_cfg"
        assert snap.config_json == config_dict
        assert snap.config_hash != ""
        assert snap.created_at != ""

    def test_deduplication_by_hash(self, repo: FactorRepository) -> None:
        """Same content returns the same config_id, no duplicate row."""
        config_dict = {"key": "value"}
        id1 = repo.save_config_snapshot(config_dict, config_type="factors")
        id2 = repo.save_config_snapshot(config_dict, config_type="factors")
        assert id1 == id2

    def test_different_content_different_id(self, repo: FactorRepository) -> None:
        id1 = repo.save_config_snapshot({"a": 1}, config_type="factors")
        id2 = repo.save_config_snapshot({"a": 2}, config_type="factors")
        assert id1 != id2

    def test_file_path_stored(self, repo: FactorRepository) -> None:
        config_id = repo.save_config_snapshot(
            {"x": 1}, config_type="analysis",
            file_path="/path/to/config.yaml",
        )
        snap = repo.get_config_snapshot(config_id)
        assert snap.file_path == "/path/to/config.yaml"

    def test_get_nonexistent(self, repo: FactorRepository) -> None:
        assert repo.get_config_snapshot("nonexistent") is None

    def test_different_types_same_content(self, repo: FactorRepository) -> None:
        """Different config_type with same content produce different IDs."""
        id1 = repo.save_config_snapshot({"x": 1}, config_type="factors")
        id2 = repo.save_config_snapshot({"x": 1}, config_type="analysis")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Analysis metrics
# ---------------------------------------------------------------------------


class TestSaveMetrics:
    def test_save_and_get(self, repo: FactorRepository) -> None:
        metrics = [
            _make_metrics(
                "run1", "f1", "1",
                ic_mean=0.05, ic_std=0.02, icir=2.5,
                t_stat_ic=3.1, p_value_ic=0.001,
                t_stat_nw=2.8, p_value_nw=0.003,
                n_eff=100, ic_skew=-0.1, ic_kurtosis=3.0, n_samples=500,
                win_rate=0.55, monotonicity=0.8,
                ic_half_life=10.0, ic_linearity=0.9, ic_ar1=0.3,
                coverage=0.95,
                mean_return=0.001, turnover=0.15,
                factor_config_id="cfg1",
                analysis_config_id="acfg1",
                output_dir="/out",
                timeframe="8h",
            ),
            _make_metrics(
                "run1", "f1", "4",
                ic_mean=0.03, icir=1.5,
                timeframe="8h",
            ),
        ]
        repo.save_metrics(metrics)
        loaded = repo.get_metrics("f1")
        assert len(loaded) == 2

    def test_all_20_metric_fields(self, repo: FactorRepository) -> None:
        m = _make_metrics(
            ic_mean=0.05, ic_std=0.02, icir=2.5,
            t_stat_ic=3.1, p_value_ic=0.001,
            t_stat_nw=2.8, p_value_nw=0.003,
            n_eff=100, ic_skew=-0.1, ic_kurtosis=3.0, n_samples=500,
            win_rate=0.55, monotonicity=0.8,
            ic_half_life=10.0, ic_linearity=0.9, ic_ar1=0.3,
            coverage=0.95,
            mean_return=0.001, turnover=0.15,
        )
        repo.save_metrics([m])
        loaded = repo.get_metrics("f1")
        assert len(loaded) == 1
        r = loaded[0]
        assert r.ic_mean == 0.05
        assert r.ic_std == 0.02
        assert r.icir == 2.5
        assert r.t_stat_ic == 3.1
        assert r.p_value_ic == 0.001
        assert r.t_stat_nw == 2.8
        assert r.p_value_nw == 0.003
        assert r.n_eff == 100
        assert r.ic_skew == pytest.approx(-0.1)
        assert r.ic_kurtosis == 3.0
        assert r.n_samples == 500
        assert r.win_rate == 0.55
        assert r.monotonicity == 0.8
        assert r.ic_half_life == 10.0
        assert r.ic_linearity == 0.9
        assert r.ic_ar1 == 0.3
        assert r.coverage == 0.95
        assert r.mean_return == 0.001
        assert r.turnover == 0.15

    def test_nullable_metric_fields(self, repo: FactorRepository) -> None:
        """Metrics with None values are stored and retrieved as None."""
        m = _make_metrics(ic_mean=None, icir=None, win_rate=None)
        repo.save_metrics([m])
        loaded = repo.get_metrics("f1")
        r = loaded[0]
        assert r.ic_mean is None
        assert r.icir is None
        assert r.win_rate is None

    def test_save_metrics_replace_on_duplicate(self, repo: FactorRepository) -> None:
        """INSERT OR REPLACE by PK (run_id, factor_id, period)."""
        m1 = _make_metrics("run1", "f1", "1", ic_mean=0.05)
        repo.save_metrics([m1])
        m2 = _make_metrics("run1", "f1", "1", ic_mean=0.10)
        repo.save_metrics([m2])
        loaded = repo.get_metrics("f1")
        assert len(loaded) == 1
        assert loaded[0].ic_mean == 0.10

    def test_save_metrics_config_refs(self, repo: FactorRepository) -> None:
        m = _make_metrics(
            factor_config_id="fcfg1",
            analysis_config_id="acfg1",
        )
        repo.save_metrics([m])
        loaded = repo.get_metrics("f1")
        assert loaded[0].factor_config_id == "fcfg1"
        assert loaded[0].analysis_config_id == "acfg1"

    def test_get_metrics_empty(self, repo: FactorRepository) -> None:
        assert repo.get_metrics("nonexistent") == []


class TestGetMetricsFilters:
    @pytest.fixture(autouse=True)
    def _seed_metrics(self, repo: FactorRepository) -> None:
        metrics = [
            _make_metrics("run1", "f1", "1", ic_mean=0.05, timeframe="8h"),
            _make_metrics("run1", "f1", "4", ic_mean=0.03, timeframe="8h"),
            _make_metrics("run2", "f1", "1", ic_mean=0.06, timeframe="8h"),
            # Use run3 so PK (run3, f1, 1) does not collide with (run1, f1, 1)
            _make_metrics("run3", "f1", "1", ic_mean=0.04, timeframe="4h"),
            _make_metrics("run1", "f2", "1", ic_mean=0.07, timeframe="8h"),
        ]
        repo.save_metrics(metrics)

    def test_filter_by_factor_id(self, repo: FactorRepository) -> None:
        result = repo.get_metrics("f2")
        assert len(result) == 1
        assert result[0].factor_id == "f2"

    def test_filter_by_run_id(self, repo: FactorRepository) -> None:
        result = repo.get_metrics("f1", run_id="run2")
        assert len(result) == 1
        assert result[0].run_id == "run2"

    def test_filter_by_timeframe(self, repo: FactorRepository) -> None:
        result = repo.get_metrics("f1", timeframe="4h")
        assert len(result) == 1
        assert result[0].timeframe == "4h"

    def test_combined_filters(self, repo: FactorRepository) -> None:
        result = repo.get_metrics("f1", run_id="run1", timeframe="8h")
        assert len(result) == 2


class TestGetLatestMetrics:
    def test_returns_latest_run(self, repo: FactorRepository) -> None:
        m1 = _make_metrics(
            "run_old", "f1", "1", ic_mean=0.01,
            timeframe="8h", created_at="2026-01-01T00:00:00+00:00",
        )
        m2 = _make_metrics(
            "run_new", "f1", "1", ic_mean=0.02,
            timeframe="8h", created_at="2026-02-01T00:00:00+00:00",
        )
        m3 = _make_metrics(
            "run_new", "f1", "4", ic_mean=0.03,
            timeframe="8h", created_at="2026-02-01T00:00:00+00:00",
        )
        repo.save_metrics([m1, m2, m3])
        result = repo.get_latest_metrics("f1", "8h")
        assert len(result) == 2
        assert all(m.run_id == "run_new" for m in result)

    def test_returns_empty_for_unknown(self, repo: FactorRepository) -> None:
        assert repo.get_latest_metrics("nonexistent", "8h") == []


class TestGetBestFactors:
    @pytest.fixture(autouse=True)
    def _seed(self, repo: FactorRepository) -> None:
        metrics = [
            _make_metrics("r1", "f1", "1", icir=-0.5, ic_mean=0.05,
                          monotonicity=0.8, timeframe="8h"),
            _make_metrics("r1", "f2", "1", icir=0.3, ic_mean=0.03,
                          monotonicity=0.6, timeframe="8h"),
            _make_metrics("r1", "f3", "1", icir=-0.1, ic_mean=0.01,
                          monotonicity=0.9, timeframe="8h"),
        ]
        repo.save_metrics(metrics)

    def test_default_sort_by_abs_icir(self, repo: FactorRepository) -> None:
        result = repo.get_best_factors("8h", "1")
        assert len(result) == 3
        # Sorted by ABS(icir) DESC: -0.5, 0.3, -0.1
        assert result[0].factor_id == "f1"
        assert result[1].factor_id == "f2"
        assert result[2].factor_id == "f3"

    def test_sort_by_monotonicity(self, repo: FactorRepository) -> None:
        result = repo.get_best_factors("8h", "1", metric="monotonicity")
        assert result[0].factor_id == "f3"  # 0.9

    def test_limit(self, repo: FactorRepository) -> None:
        result = repo.get_best_factors("8h", "1", limit=2)
        assert len(result) == 2

    def test_invalid_metric_defaults_to_icir(
        self, repo: FactorRepository,
    ) -> None:
        result = repo.get_best_factors("8h", "1", metric="nonexistent")
        assert result[0].factor_id == "f1"

    def test_empty_for_wrong_timeframe(self, repo: FactorRepository) -> None:
        result = repo.get_best_factors("1h", "1")
        assert result == []

    def test_nulls_last(self, repo: FactorRepository) -> None:
        m_null = _make_metrics("r2", "f_null", "1", icir=None, timeframe="8h")
        repo.save_metrics([m_null])
        result = repo.get_best_factors("8h", "1")
        # f_null with None icir should appear last
        assert result[-1].factor_id == "f_null"
