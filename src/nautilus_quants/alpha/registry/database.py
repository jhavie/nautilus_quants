# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""DuckDB connection management and schema initialization for the Factor Registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS factors (
    factor_id    VARCHAR PRIMARY KEY,
    expression   VARCHAR NOT NULL,
    description  VARCHAR DEFAULT '',
    category     VARCHAR DEFAULT '',
    source       VARCHAR DEFAULT '',
    status       VARCHAR DEFAULT 'candidate',
    context_id   VARCHAR DEFAULT '',
    created_at   VARCHAR NOT NULL,
    updated_at   VARCHAR NOT NULL,
    ic_mean      DOUBLE,
    icir         DOUBLE,
    score        DOUBLE,
    bar_spec     VARCHAR
);

CREATE TABLE IF NOT EXISTS factor_versions (
    factor_id    VARCHAR NOT NULL,
    version      INTEGER NOT NULL,
    expression   VARCHAR NOT NULL,
    reason       VARCHAR DEFAULT '',
    created_at   VARCHAR NOT NULL,
    PRIMARY KEY (factor_id, version)
);

CREATE TABLE IF NOT EXISTS config_context (
    context_id   VARCHAR PRIMARY KEY,
    variables    JSON DEFAULT '{}',
    parameters   JSON DEFAULT '{}',
    metadata     JSON DEFAULT '{}',
    created_at   VARCHAR NOT NULL,
    updated_at   VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_cache (
    factor_id    VARCHAR NOT NULL,
    version      INTEGER NOT NULL DEFAULT 1,
    bar_spec     VARCHAR NOT NULL,
    period       INTEGER NOT NULL,
    ic_mean      DOUBLE,
    ic_std       DOUBLE,
    icir         DOUBLE,
    mean_return  DOUBLE,
    turnover     DOUBLE,
    analyzed_at  VARCHAR NOT NULL,
    PRIMARY KEY (factor_id, version, bar_spec, period)
);
"""

# Default database path (relative to project root).
DEFAULT_DB_PATH = "data/factor_registry.duckdb"


class RegistryDatabase:
    """Thin wrapper around a DuckDB connection for the Factor Registry.

    Parameters
    ----------
    db_path : str | Path
        Path to the DuckDB database file.  Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(self._db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        for statement in _SCHEMA_SQL.strip().split(";"):
            statement = statement.strip()
            if statement:
                self._conn.execute(statement)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Raw DuckDB connection (for advanced queries)."""
        return self._conn

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        """Execute a write statement (INSERT/UPDATE/DELETE)."""
        if params:
            self._conn.execute(sql, params)
        else:
            self._conn.execute(sql)

    def fetch_one(self, sql: str, params: list[Any] | None = None) -> tuple | None:
        """Execute a query and return a single row (or None)."""
        if params:
            result = self._conn.execute(sql, params)
        else:
            result = self._conn.execute(sql)
        return result.fetchone()

    def fetch_all(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        """Execute a query and return all rows."""
        if params:
            result = self._conn.execute(sql, params)
        else:
            result = self._conn.execute(sql)
        return result.fetchall()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
