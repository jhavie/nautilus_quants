# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for RegistryDatabase — DuckDB connection and schema initialization."""

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


class TestRegistryDatabaseInit:
    """Table creation and connection lifecycle."""

    def test_tables_created(self, db: RegistryDatabase) -> None:
        tables = db.fetch_all(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        )
        table_names = [row[0] for row in tables]
        assert "factors" in table_names
        assert "factor_versions" in table_names
        assert "config_context" in table_names
        assert "analysis_cache" in table_names

    def test_idempotent_init(self, db: RegistryDatabase) -> None:
        """Calling _init_tables again should not fail."""
        db._init_tables()
        tables = db.fetch_all(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        )
        assert len(tables) == 4

    def test_close_and_reconnect(self, tmp_path) -> None:
        db_path = tmp_path / "test.duckdb"
        db1 = RegistryDatabase(db_path)
        db1.execute(
            "INSERT INTO factors (factor_id, expression, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ["alpha001", "close", "2026-01-01", "2026-01-01"],
        )
        db1.close()

        db2 = RegistryDatabase(db_path)
        row = db2.fetch_one("SELECT factor_id FROM factors WHERE factor_id = ?", ["alpha001"])
        db2.close()
        assert row is not None
        assert row[0] == "alpha001"


class TestExecuteAndFetch:
    """Basic CRUD operations via execute/fetch."""

    def test_insert_and_fetch_one(self, db: RegistryDatabase) -> None:
        db.execute(
            "INSERT INTO factors (factor_id, expression, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ["alpha001", "rank(close)", "2026-01-01", "2026-01-01"],
        )
        row = db.fetch_one("SELECT factor_id, expression FROM factors WHERE factor_id = ?", ["alpha001"])
        assert row == ("alpha001", "rank(close)")

    def test_fetch_one_returns_none(self, db: RegistryDatabase) -> None:
        row = db.fetch_one("SELECT * FROM factors WHERE factor_id = ?", ["nonexistent"])
        assert row is None

    def test_fetch_all_empty(self, db: RegistryDatabase) -> None:
        rows = db.fetch_all("SELECT * FROM factors")
        assert rows == []

    def test_fetch_all_multiple_rows(self, db: RegistryDatabase) -> None:
        for i in range(3):
            db.execute(
                "INSERT INTO factors (factor_id, expression, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                [f"alpha{i:03d}", f"expr_{i}", "2026-01-01", "2026-01-01"],
            )
        rows = db.fetch_all("SELECT factor_id FROM factors ORDER BY factor_id")
        assert len(rows) == 3
        assert [r[0] for r in rows] == ["alpha000", "alpha001", "alpha002"]

    def test_insert_factor_version(self, db: RegistryDatabase) -> None:
        db.execute(
            "INSERT INTO factor_versions (factor_id, version, expression, created_at) "
            "VALUES (?, ?, ?, ?)",
            ["alpha001", 1, "rank(close)", "2026-01-01"],
        )
        row = db.fetch_one(
            "SELECT version, expression FROM factor_versions "
            "WHERE factor_id = ? AND version = ?",
            ["alpha001", 1],
        )
        assert row == (1, "rank(close)")

    def test_config_context_json(self, db: RegistryDatabase) -> None:
        """JSON columns round-trip correctly."""
        variables = {"returns": "delta(close, 1) / delay(close, 1)"}
        parameters = {"short_window": 24, "long_window": 96}
        metadata = {"name": "test", "version": "1.0"}
        db.execute(
            "INSERT INTO config_context "
            "(context_id, variables, parameters, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                "test_ctx",
                json.dumps(variables),
                json.dumps(parameters),
                json.dumps(metadata),
                "2026-01-01",
                "2026-01-01",
            ],
        )
        row = db.fetch_one(
            "SELECT variables, parameters, metadata FROM config_context "
            "WHERE context_id = ?",
            ["test_ctx"],
        )
        assert row is not None
        assert json.loads(row[0]) == variables
        assert json.loads(row[1]) == parameters
        assert json.loads(row[2]) == metadata

    def test_analysis_cache_composite_pk(self, db: RegistryDatabase) -> None:
        db.execute(
            "INSERT INTO analysis_cache "
            "(factor_id, bar_spec, period, ic_mean, analyzed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ["alpha001", "4h", 1, 0.05, "2026-01-01"],
        )
        db.execute(
            "INSERT INTO analysis_cache "
            "(factor_id, bar_spec, period, ic_mean, analyzed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ["alpha001", "4h", 4, 0.03, "2026-01-01"],
        )
        rows = db.fetch_all(
            "SELECT period, ic_mean FROM analysis_cache "
            "WHERE factor_id = ? ORDER BY period",
            ["alpha001"],
        )
        assert len(rows) == 2
        assert rows[0] == (1, 0.05)
        assert rows[1] == (4, 0.03)


class TestFilePath:
    """Database file creation on disk."""

    def test_creates_parent_directory(self, tmp_path) -> None:
        db_path = tmp_path / "subdir" / "nested" / "test.duckdb"
        db = RegistryDatabase(db_path)
        db.close()
        assert db_path.exists()
