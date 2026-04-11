# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""DuckDB connection management and schema initialization for the Factor Registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import logging

import duckdb

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS factors (
    factor_id    VARCHAR PRIMARY KEY,
    prototype    VARCHAR DEFAULT '',
    expression   VARCHAR NOT NULL,
    expression_hash VARCHAR DEFAULT '',
    description  VARCHAR DEFAULT '',
    source       VARCHAR DEFAULT '',
    status       VARCHAR DEFAULT 'candidate',
    tags         JSON DEFAULT '[]',
    parameters   JSON DEFAULT '{}',
    variables    JSON DEFAULT '{}',
    created_at   TIMESTAMP NOT NULL,
    updated_at   TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS configs_snapshot (
    config_id     VARCHAR PRIMARY KEY,
    type          VARCHAR NOT NULL,
    config_name   VARCHAR DEFAULT '',
    config_json   JSON NOT NULL,
    file_path     VARCHAR DEFAULT '',
    config_hash   VARCHAR DEFAULT '',
    created_at    TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS alpha_analysis_metrics (
    run_id              VARCHAR NOT NULL,
    factor_id           VARCHAR NOT NULL,
    period              VARCHAR NOT NULL,
    ic_mean             DOUBLE,
    ic_std              DOUBLE,
    icir                DOUBLE,
    t_stat_ic           DOUBLE,
    p_value_ic          DOUBLE,
    t_stat_nw           DOUBLE,
    p_value_nw          DOUBLE,
    n_eff               INTEGER,
    ic_skew             DOUBLE,
    ic_kurtosis         DOUBLE,
    n_samples           INTEGER,
    win_rate            DOUBLE,
    monotonicity        DOUBLE,
    ic_half_life        DOUBLE,
    ic_linearity        DOUBLE,
    ic_ar1              DOUBLE,
    coverage            DOUBLE,
    mean_return         DOUBLE,
    turnover            DOUBLE,
    factor_config_id    VARCHAR,
    analysis_config_id  VARCHAR,
    output_dir          VARCHAR DEFAULT '',
    timeframe           VARCHAR DEFAULT '',
    created_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, factor_id, period)
);

CREATE TABLE IF NOT EXISTS backtest_run_metrics (
    backtest_id         VARCHAR PRIMARY KEY,
    config_id           VARCHAR,
    factor_config_id    VARCHAR,
    output_dir          VARCHAR NOT NULL,
    strategy_name       VARCHAR DEFAULT '',
    instrument_count    INTEGER DEFAULT 0,
    timeframe           VARCHAR DEFAULT '',
    started_at          TIMESTAMP NOT NULL,
    finished_at         TIMESTAMP,
    duration_seconds    DOUBLE,
    total_pnl           DOUBLE,
    total_pnl_pct       DOUBLE,
    sharpe_ratio        DOUBLE,
    max_drawdown        DOUBLE,
    win_rate            DOUBLE,
    statistics_json     JSON DEFAULT '{}',
    reports_json        JSON DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS backtest_factors (
    backtest_id  VARCHAR NOT NULL,
    factor_id    VARCHAR NOT NULL,
    role         VARCHAR DEFAULT 'component',
    PRIMARY KEY (backtest_id, factor_id)
);
"""

DEFAULT_DB_DIR = "logs/registry"
DEFAULT_ENV = "test"


class RegistryDatabase:
    """Thin wrapper around a DuckDB connection for the Factor Registry.

    Parameters
    ----------
    db_path : str | Path
        Path to the DuckDB database file.  Use ``":memory:"`` for tests.
    """

    def __init__(
        self, db_path: str | Path = ":memory:",
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ) -> None:
        import time as _time

        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, max_retries + 1):
            try:
                self._conn: duckdb.DuckDBPyConnection = duckdb.connect(
                    self._db_path,
                )
                break
            except duckdb.IOException as e:
                if "lock" in str(e).lower() and attempt < max_retries:
                    logger.warning(
                        "DuckDB lock conflict on %s — retry %d/%d in %.0fs",
                        self._db_path, attempt, max_retries, retry_delay,
                    )
                    _time.sleep(retry_delay)
                    continue
                if "lock" in str(e).lower():
                    logger.error(
                        "DuckDB lock conflict on %s — all %d retries exhausted.",
                        self._db_path, max_retries,
                    )
                raise
        self._init_tables()

    @classmethod
    def for_environment(
        cls,
        env: str = DEFAULT_ENV,
        db_dir: str | Path = DEFAULT_DB_DIR,
    ) -> RegistryDatabase:
        """Create a database instance for the given environment.

        Parameters
        ----------
        env : str
            Environment name: ``"test"``, ``"dev"``, or ``"prod"``.
        db_dir : str | Path
            Base directory for database files.
        """
        db_path = Path(db_dir) / f"{env}.duckdb"
        return cls(db_path)

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        for statement in _SCHEMA_SQL.strip().split(";"):
            statement = statement.strip()
            if statement:
                self._conn.execute(statement)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Apply incremental schema migrations for existing databases."""
        # v1 → v2: add expression_hash column to factors
        cols = {
            r[0]
            for r in self._conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'factors'"
            ).fetchall()
        }
        if "expression_hash" not in cols:
            self._conn.execute(
                "ALTER TABLE factors ADD COLUMN "
                "expression_hash VARCHAR DEFAULT ''"
            )

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

    def fetch_one(
        self, sql: str, params: list[Any] | None = None,
    ) -> tuple | None:
        """Execute a query and return a single row (or None)."""
        if params:
            result = self._conn.execute(sql, params)
        else:
            result = self._conn.execute(sql)
        return result.fetchone()

    def fetch_all(
        self, sql: str, params: list[Any] | None = None,
    ) -> list[tuple]:
        """Execute a query and return all rows."""
        if params:
            result = self._conn.execute(sql, params)
        else:
            result = self._conn.execute(sql)
        return result.fetchall()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
