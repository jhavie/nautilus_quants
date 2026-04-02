# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""FactorRepository — business-logic CRUD, versioning, status, and context management."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import (
    STATUS_TRANSITIONS,
    VALID_STATUSES,
    AnalysisResult,
    ConfigContext,
    FactorRecord,
    FactorVersion,
)
from nautilus_quants.factors.config import FactorConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FactorRepository:
    """High-level interface for the Factor Registry.

    Parameters
    ----------
    db : RegistryDatabase
        Opened database connection.
    """

    def __init__(self, db: RegistryDatabase) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Factor CRUD
    # ------------------------------------------------------------------

    def upsert_factor(self, record: FactorRecord) -> str:
        """Insert or update a factor.  Returns ``"new"`` / ``"updated"`` / ``"unchanged"``.

        On expression change, a new version record is auto-created.
        """
        now = _now_iso()
        existing = self.get_factor(record.factor_id)

        if existing is None:
            self._db.execute(
                "INSERT INTO factors "
                "(factor_id, expression, description, category, source, status, "
                " created_at, updated_at, ic_mean, icir, score, bar_spec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    record.factor_id,
                    record.expression,
                    record.description,
                    record.category,
                    record.source,
                    record.status,
                    now,
                    now,
                    record.ic_mean,
                    record.icir,
                    record.score,
                    record.bar_spec,
                ],
            )
            self._create_version(record.factor_id, record.expression, "initial", now)
            return "new"

        expression_changed = existing.expression != record.expression
        metadata_changed = (
            existing.description != record.description
            or existing.category != record.category
        )

        if not expression_changed and not metadata_changed:
            return "unchanged"

        # Update mutable fields; preserve status/source/scores from DB.
        self._db.execute(
            "UPDATE factors SET expression = ?, description = ?, category = ?, "
            "updated_at = ? WHERE factor_id = ?",
            [record.expression, record.description, record.category, now, record.factor_id],
        )
        if expression_changed:
            self._create_version(record.factor_id, record.expression, "", now)
        return "updated"

    def get_factor(self, factor_id: str) -> FactorRecord | None:
        """Get a single factor by ID."""
        row = self._db.fetch_one(
            "SELECT factor_id, expression, description, category, source, status, "
            "created_at, updated_at, ic_mean, icir, score, bar_spec "
            "FROM factors WHERE factor_id = ?",
            [factor_id],
        )
        if row is None:
            return None
        return FactorRecord(*row)

    def delete_factor(self, factor_id: str) -> None:
        """Delete a factor and its version history."""
        self._db.execute("DELETE FROM factor_versions WHERE factor_id = ?", [factor_id])
        self._db.execute("DELETE FROM factors WHERE factor_id = ?", [factor_id])

    # ------------------------------------------------------------------
    # List / Query
    # ------------------------------------------------------------------

    def list_factors(
        self,
        status: str | None = None,
        category: str | None = None,
        source: str | None = None,
        sort_by: str = "factor_id",
        descending: bool = False,
        limit: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[FactorRecord]:
        """Query factors with optional filters, sorting, and limit.

        Parameters
        ----------
        sort_by : str
            Column to sort by.  Use ``"abs_icir"`` for ``ABS(icir) DESC
            NULLS LAST`` (selects strongest-signal factors regardless of
            sign).
        descending : bool
            Sort descending (ignored when ``sort_by="abs_icir"``).
        exclude_ids : set[str] | None
            Factor IDs to exclude from results.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if exclude_ids:
            placeholders = ", ".join("?" for _ in exclude_ids)
            clauses.append(f"factor_id NOT IN ({placeholders})")
            params.extend(sorted(exclude_ids))

        where = " WHERE " + " AND ".join(clauses) if clauses else ""

        # Whitelist sort columns to prevent SQL injection.
        allowed_sort = {
            "factor_id", "category", "source", "status",
            "created_at", "updated_at", "ic_mean", "icir", "score",
            "abs_icir",
        }
        if sort_by not in allowed_sort:
            sort_by = "factor_id"

        if sort_by == "abs_icir":
            order_clause = "ABS(icir) DESC NULLS LAST"
        else:
            direction = "DESC" if descending else "ASC"
            order_clause = f"{sort_by} {direction} NULLS LAST"

        sql = (
            "SELECT factor_id, expression, description, category, source, status, "
            "created_at, updated_at, ic_mean, icir, score, bar_spec "
            f"FROM factors{where} ORDER BY {order_clause}"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self._db.fetch_all(sql, params if params else None)
        return [FactorRecord(*row) for row in rows]

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def set_status(self, factor_id: str, new_status: str) -> None:
        """Transition a factor's status.

        Raises
        ------
        ValueError
            If the factor doesn't exist or the transition is invalid.
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(f"无效状态: {new_status}（合法值: {VALID_STATUSES}）")

        factor = self.get_factor(factor_id)
        if factor is None:
            raise ValueError(f"因子不存在: {factor_id}")

        allowed = STATUS_TRANSITIONS.get(factor.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"非法状态转换 {factor.status} → {new_status}"
                f"（{factor.status} 只能转换到 {allowed}）"
            )

        self._db.execute(
            "UPDATE factors SET status = ?, updated_at = ? WHERE factor_id = ?",
            [new_status, _now_iso(), factor_id],
        )

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def get_versions(self, factor_id: str) -> list[FactorVersion]:
        """Get all version records for a factor, ordered by version number."""
        rows = self._db.fetch_all(
            "SELECT factor_id, version, expression, reason, created_at "
            "FROM factor_versions WHERE factor_id = ? ORDER BY version",
            [factor_id],
        )
        return [FactorVersion(*row) for row in rows]

    def _create_version(
        self, factor_id: str, expression: str, reason: str, created_at: str,
    ) -> None:
        row = self._db.fetch_one(
            "SELECT COALESCE(MAX(version), 0) FROM factor_versions WHERE factor_id = ?",
            [factor_id],
        )
        next_version = (row[0] if row else 0) + 1
        self._db.execute(
            "INSERT INTO factor_versions (factor_id, version, expression, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [factor_id, next_version, expression, reason, created_at],
        )

    # ------------------------------------------------------------------
    # Import from FactorConfig
    # ------------------------------------------------------------------

    def import_from_config(
        self,
        config: FactorConfig,
        source: str = "",
        context_id: str = "",
    ) -> tuple[int, int, int]:
        """Batch-import factors from a ``FactorConfig`` (as loaded from YAML).

        Returns ``(new_count, updated_count, unchanged_count)``.
        """
        new = updated = unchanged = 0
        for fdef in config.factors:
            record = FactorRecord(
                factor_id=fdef.name,
                expression=fdef.expression,
                description=fdef.description,
                category=fdef.category,
                source=source,
            )
            result = self.upsert_factor(record)
            if result == "new":
                new += 1
            elif result == "updated":
                updated += 1
            else:
                unchanged += 1

        # Store config context.
        cid = context_id or config.name
        ctx = ConfigContext(
            context_id=cid,
            variables=config.variables,
            parameters=config.parameters,
            metadata={
                "name": config.name,
                "version": config.version,
                "description": config.description,
            },
        )
        self.upsert_context(ctx)

        return new, updated, unchanged

    # ------------------------------------------------------------------
    # Config context
    # ------------------------------------------------------------------

    def upsert_context(self, ctx: ConfigContext) -> None:
        """Insert or update a config context."""
        now = _now_iso()
        existing = self.get_context(ctx.context_id)
        if existing is None:
            self._db.execute(
                "INSERT INTO config_context "
                "(context_id, variables, parameters, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ctx.context_id,
                    json.dumps(ctx.variables),
                    json.dumps(ctx.parameters),
                    json.dumps(ctx.metadata),
                    now,
                    now,
                ],
            )
        else:
            self._db.execute(
                "UPDATE config_context SET variables = ?, parameters = ?, "
                "metadata = ?, updated_at = ? WHERE context_id = ?",
                [
                    json.dumps(ctx.variables),
                    json.dumps(ctx.parameters),
                    json.dumps(ctx.metadata),
                    now,
                    ctx.context_id,
                ],
            )

    def get_context(self, context_id: str) -> ConfigContext | None:
        """Get a config context by ID."""
        row = self._db.fetch_one(
            "SELECT context_id, variables, parameters, metadata, created_at "
            "FROM config_context WHERE context_id = ?",
            [context_id],
        )
        if row is None:
            return None
        return ConfigContext(
            context_id=row[0],
            variables=json.loads(row[1]) if row[1] else {},
            parameters=json.loads(row[2]) if row[2] else {},
            metadata=json.loads(row[3]) if row[3] else {},
            created_at=row[4],
        )

    def list_contexts(self) -> list[ConfigContext]:
        """List all config contexts."""
        rows = self._db.fetch_all(
            "SELECT context_id, variables, parameters, metadata, created_at "
            "FROM config_context ORDER BY context_id"
        )
        return [
            ConfigContext(
                context_id=r[0],
                variables=json.loads(r[1]) if r[1] else {},
                parameters=json.loads(r[2]) if r[2] else {},
                metadata=json.loads(r[3]) if r[3] else {},
                created_at=r[4],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Analysis results (interface for Feature 035)
    # ------------------------------------------------------------------

    def save_analysis(self, results: list[AnalysisResult]) -> None:
        """Save analysis results (upsert by composite PK)."""
        for r in results:
            self._db.execute(
                "INSERT OR REPLACE INTO analysis_cache "
                "(factor_id, bar_spec, period, ic_mean, ic_std, icir, "
                " mean_return, turnover, analyzed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    r.factor_id, r.bar_spec, r.period,
                    r.ic_mean, r.ic_std, r.icir,
                    r.mean_return, r.turnover, r.analyzed_at,
                ],
            )

    def get_analysis(self, factor_id: str, bar_spec: str) -> list[AnalysisResult]:
        """Get analysis results for a factor at a given bar_spec."""
        rows = self._db.fetch_all(
            "SELECT factor_id, bar_spec, period, ic_mean, ic_std, icir, "
            "mean_return, turnover, analyzed_at "
            "FROM analysis_cache WHERE factor_id = ? AND bar_spec = ? ORDER BY period",
            [factor_id, bar_spec],
        )
        return [AnalysisResult(*r) for r in rows]
