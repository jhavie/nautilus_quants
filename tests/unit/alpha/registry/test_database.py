# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for RegistryDatabase — DuckDB connection and v2 schema initialization."""

from __future__ import annotations

import json

import pytest

from nautilus_quants.alpha.registry.database import RegistryDatabase


@pytest.fixture()
def db() -> RegistryDatabase:
    """In-memory DuckDB for each test."""
    database = RegistryDatabase(":memory:")
    yield database
    database.close()


EXPECTED_TABLES = sorted([
    "alpha_analysis_metrics",
    "backtest_factors",
    "backtest_run_metrics",
    "configs_snapshot",
    "factors",
])


class TestSchemaCreation:
    """All five v2 tables are created on init."""

    def test_all_tables_created(self, db: RegistryDatabase) -> None:
        tables = db.fetch_all(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        )
        table_names = sorted(row[0] for row in tables)
        assert table_names == EXPECTED_TABLES

    def test_idempotent_init(self, db: RegistryDatabase) -> None:
        """Calling _init_tables again should not fail or duplicate tables."""
        db._init_tables()
        tables = db.fetch_all(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        )
        assert len(tables) == 5

    def test_factors_table_columns(self, db: RegistryDatabase) -> None:
        cols = db.fetch_all(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'factors' ORDER BY ordinal_position"
        )
        col_names = [r[0] for r in cols]
        expected = [
            "factor_id", "prototype", "expression", "expression_hash",
            "description",
            "source", "status", "tags", "parameters", "variables",
            "created_at", "updated_at",
        ]
        assert col_names == expected

    def test_configs_snapshot_table_columns(self, db: RegistryDatabase) -> None:
        cols = db.fetch_all(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'configs_snapshot' ORDER BY ordinal_position"
        )
        col_names = [r[0] for r in cols]
        expected = [
            "config_id", "type", "config_name", "config_json",
            "file_path", "config_hash", "created_at",
        ]
        assert col_names == expected

    def test_alpha_analysis_metrics_pk(self, db: RegistryDatabase) -> None:
        """Composite PK (run_id, factor_id, period) allows multiple periods."""
        db.execute(
            "INSERT INTO alpha_analysis_metrics "
            "(run_id, factor_id, period, ic_mean, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ["run1", "f1", "1", 0.05, "2026-01-01"],
        )
        db.execute(
            "INSERT INTO alpha_analysis_metrics "
            "(run_id, factor_id, period, ic_mean, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ["run1", "f1", "4", 0.03, "2026-01-01"],
        )
        rows = db.fetch_all(
            "SELECT period, ic_mean FROM alpha_analysis_metrics "
            "WHERE factor_id = ? ORDER BY period",
            ["f1"],
        )
        assert len(rows) == 2
        assert rows[0][1] == 0.05
        assert rows[1][1] == 0.03

    def test_backtest_factors_composite_pk(self, db: RegistryDatabase) -> None:
        """backtest_factors has PK (backtest_id, factor_id)."""
        db.execute(
            "INSERT INTO backtest_run_metrics "
            "(backtest_id, output_dir, started_at) "
            "VALUES (?, ?, ?)",
            ["bt1", "/out", "2026-01-01"],
        )
        db.execute(
            "INSERT INTO backtest_factors (backtest_id, factor_id, role) "
            "VALUES (?, ?, ?)",
            ["bt1", "f1", "component"],
        )
        db.execute(
            "INSERT INTO backtest_factors (backtest_id, factor_id, role) "
            "VALUES (?, ?, ?)",
            ["bt1", "f2", "composite"],
        )
        rows = db.fetch_all(
            "SELECT factor_id, role FROM backtest_factors "
            "WHERE backtest_id = ? ORDER BY factor_id",
            ["bt1"],
        )
        assert len(rows) == 2
        assert rows[0] == ("f1", "component")
        assert rows[1] == ("f2", "composite")


class TestForEnvironment:
    """RegistryDatabase.for_environment() factory method."""

    def test_creates_db_file(self, tmp_path) -> None:
        db = RegistryDatabase.for_environment(env="test", db_dir=tmp_path)
        db.close()
        assert (tmp_path / "test.duckdb").exists()

    def test_dev_environment(self, tmp_path) -> None:
        db = RegistryDatabase.for_environment(env="dev", db_dir=tmp_path)
        db.close()
        assert (tmp_path / "dev.duckdb").exists()

    def test_prod_environment(self, tmp_path) -> None:
        db = RegistryDatabase.for_environment(env="prod", db_dir=tmp_path)
        db.close()
        assert (tmp_path / "prod.duckdb").exists()

    def test_creates_parent_dirs(self, tmp_path) -> None:
        db_dir = tmp_path / "deep" / "nested" / "dir"
        db = RegistryDatabase.for_environment(env="test", db_dir=db_dir)
        db.close()
        assert (db_dir / "test.duckdb").exists()


class TestMemoryDatabase:
    """In-memory database behavior."""

    def test_memory_db_works(self) -> None:
        db = RegistryDatabase(":memory:")
        db.execute(
            "INSERT INTO factors "
            "(factor_id, expression, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ["f1", "rank(close)", "2026-01-01", "2026-01-01"],
        )
        row = db.fetch_one(
            "SELECT factor_id FROM factors WHERE factor_id = ?", ["f1"],
        )
        assert row is not None
        assert row[0] == "f1"
        db.close()

    def test_memory_db_isolated(self) -> None:
        """Two in-memory DBs are independent."""
        db1 = RegistryDatabase(":memory:")
        db2 = RegistryDatabase(":memory:")
        db1.execute(
            "INSERT INTO factors "
            "(factor_id, expression, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ["f1", "rank(close)", "2026-01-01", "2026-01-01"],
        )
        assert db1.fetch_one("SELECT * FROM factors WHERE factor_id = ?", ["f1"]) is not None
        assert db2.fetch_one("SELECT * FROM factors WHERE factor_id = ?", ["f1"]) is None
        db1.close()
        db2.close()


class TestExecuteAndFetch:
    """Basic CRUD operations via execute/fetch."""

    def test_insert_and_fetch_one(self, db: RegistryDatabase) -> None:
        db.execute(
            "INSERT INTO factors "
            "(factor_id, expression, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ["alpha001", "rank(close)", "2026-01-01", "2026-01-01"],
        )
        row = db.fetch_one(
            "SELECT factor_id, expression FROM factors "
            "WHERE factor_id = ?",
            ["alpha001"],
        )
        assert row == ("alpha001", "rank(close)")

    def test_fetch_one_returns_none(self, db: RegistryDatabase) -> None:
        row = db.fetch_one(
            "SELECT * FROM factors WHERE factor_id = ?",
            ["nonexistent"],
        )
        assert row is None

    def test_fetch_all_empty(self, db: RegistryDatabase) -> None:
        rows = db.fetch_all("SELECT * FROM factors")
        assert rows == []

    def test_fetch_all_multiple_rows(self, db: RegistryDatabase) -> None:
        for i in range(3):
            db.execute(
                "INSERT INTO factors "
                "(factor_id, expression, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                [f"alpha{i:03d}", f"expr_{i}", "2026-01-01", "2026-01-01"],
            )
        rows = db.fetch_all(
            "SELECT factor_id FROM factors ORDER BY factor_id",
        )
        assert len(rows) == 3
        assert [r[0] for r in rows] == ["alpha000", "alpha001", "alpha002"]

    def test_json_columns_round_trip(self, db: RegistryDatabase) -> None:
        """JSON columns (tags, parameters, variables) round-trip correctly."""
        tags = ["reversal", "volume"]
        params = {"short_window": 24, "long_window": 96}
        variables = {"returns": "delta(close, 1) / delay(close, 1)"}
        db.execute(
            "INSERT INTO factors "
            "(factor_id, expression, tags, parameters, variables, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                "f1", "rank(close)",
                json.dumps(tags),
                json.dumps(params),
                json.dumps(variables),
                "2026-01-01", "2026-01-01",
            ],
        )
        row = db.fetch_one(
            "SELECT tags, parameters, variables FROM factors "
            "WHERE factor_id = ?",
            ["f1"],
        )
        assert row is not None
        assert json.loads(row[0]) == tags
        assert json.loads(row[1]) == params
        assert json.loads(row[2]) == variables

    def test_configs_snapshot_json(self, db: RegistryDatabase) -> None:
        config_data = {"factors": {"f1": {"expression": "rank(close)"}}}
        db.execute(
            "INSERT INTO configs_snapshot "
            "(config_id, type, config_name, config_json, config_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                "cfg1", "factors", "test_config",
                json.dumps(config_data), "abc123", "2026-01-01",
            ],
        )
        row = db.fetch_one(
            "SELECT config_json FROM configs_snapshot WHERE config_id = ?",
            ["cfg1"],
        )
        assert row is not None
        assert json.loads(row[0]) == config_data


class TestCloseAndReconnect:
    """Database persistence across close/open cycles."""

    def test_data_persists(self, tmp_path) -> None:
        db_path = tmp_path / "test.duckdb"
        db1 = RegistryDatabase(db_path)
        db1.execute(
            "INSERT INTO factors "
            "(factor_id, expression, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ["alpha001", "rank(close)", "2026-01-01", "2026-01-01"],
        )
        db1.close()

        db2 = RegistryDatabase(db_path)
        row = db2.fetch_one(
            "SELECT factor_id FROM factors WHERE factor_id = ?",
            ["alpha001"],
        )
        db2.close()
        assert row is not None
        assert row[0] == "alpha001"


class TestFilePath:
    """Database file creation on disk."""

    def test_creates_parent_directory(self, tmp_path) -> None:
        db_path = tmp_path / "subdir" / "nested" / "test.duckdb"
        db = RegistryDatabase(db_path)
        db.close()
        assert db_path.exists()
