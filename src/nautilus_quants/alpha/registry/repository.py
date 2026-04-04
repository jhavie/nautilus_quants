# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""FactorRepository — factor CRUD, config snapshots, and analysis metrics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import (
    STATUS_TRANSITIONS,
    VALID_STATUSES,
    AnalysisMetrics,
    ConfigSnapshot,
    FactorRecord,
)
from nautilus_quants.factors.config import FactorConfig, generate_factor_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config_hash(config_dict: dict[str, Any]) -> str:
    """SHA-256 of canonicalized JSON."""
    raw = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class FactorRepository:
    """High-level interface for the Factor Registry v2."""

    def __init__(self, db: RegistryDatabase) -> None:
        self._db = db

    # ── Factor CRUD ──

    def upsert_factor(self, record: FactorRecord) -> str:
        """Insert or update a factor. Returns "new" / "updated" / "unchanged"."""
        now = _now_iso()
        existing = self.get_factor(record.factor_id)

        if existing is None:
            self._db.execute(
                "INSERT INTO factors "
                "(factor_id, prototype, expression, description, source, "
                " status, tags, parameters, variables, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    record.factor_id,
                    record.prototype,
                    record.expression,
                    record.description,
                    record.source,
                    record.status,
                    json.dumps(record.tags),
                    json.dumps(record.parameters),
                    json.dumps(record.variables),
                    now,
                    now,
                ],
            )
            return "new"

        changed = (
            existing.expression != record.expression
            or existing.description != record.description
            or existing.prototype != record.prototype
            or existing.tags != record.tags
            or existing.parameters != record.parameters
            or existing.variables != record.variables
        )
        if not changed:
            return "unchanged"

        self._db.execute(
            "UPDATE factors SET prototype = ?, expression = ?, description = ?, "
            "tags = ?, parameters = ?, variables = ?, updated_at = ? "
            "WHERE factor_id = ?",
            [
                record.prototype,
                record.expression,
                record.description,
                json.dumps(record.tags),
                json.dumps(record.parameters),
                json.dumps(record.variables),
                now,
                record.factor_id,
            ],
        )
        return "updated"

    def get_factor(self, factor_id: str) -> FactorRecord | None:
        row = self._db.fetch_one(
            "SELECT factor_id, expression, prototype, description, source, "
            "status, tags, parameters, variables, created_at, updated_at "
            "FROM factors WHERE factor_id = ?",
            [factor_id],
        )
        if row is None:
            return None
        return FactorRecord(
            factor_id=row[0],
            expression=row[1],
            prototype=row[2] or "",
            description=row[3] or "",
            source=row[4] or "",
            status=row[5] or "candidate",
            tags=json.loads(row[6]) if row[6] else [],
            parameters=json.loads(row[7]) if row[7] else {},
            variables=json.loads(row[8]) if row[8] else {},
            created_at=str(row[9]) if row[9] else "",
            updated_at=str(row[10]) if row[10] else "",
        )

    def delete_factor(self, factor_id: str) -> None:
        self._db.execute(
            "DELETE FROM factors WHERE factor_id = ?", [factor_id],
        )

    def list_factors(
        self,
        status: str | None = None,
        source: str | None = None,
        prototype: str | None = None,
        tag: str | None = None,
        sort_by: str = "factor_id",
        descending: bool = False,
        limit: int | None = None,
    ) -> list[FactorRecord]:
        clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if prototype is not None:
            clauses.append("prototype = ?")
            params.append(prototype)
        if tag is not None:
            clauses.append("list_contains(tags::VARCHAR[], ?)")
            params.append(tag)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        allowed = {"factor_id", "source", "status", "prototype", "created_at"}
        if sort_by not in allowed:
            sort_by = "factor_id"
        direction = "DESC" if descending else "ASC"

        sql = (
            "SELECT factor_id, expression, prototype, description, source, "
            "status, tags, parameters, variables, created_at, updated_at "
            f"FROM factors{where} ORDER BY {sort_by} {direction}"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self._db.fetch_all(sql, params if params else None)
        return [
            FactorRecord(
                factor_id=r[0],
                expression=r[1],
                prototype=r[2] or "",
                description=r[3] or "",
                source=r[4] or "",
                status=r[5] or "candidate",
                tags=json.loads(r[6]) if r[6] else [],
                parameters=json.loads(r[7]) if r[7] else {},
                variables=json.loads(r[8]) if r[8] else {},
                created_at=str(r[9]) if r[9] else "",
                updated_at=str(r[10]) if r[10] else "",
            )
            for r in rows
        ]

    # ── Status transitions ──

    def set_status(self, factor_id: str, new_status: str) -> None:
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {new_status}")
        factor = self.get_factor(factor_id)
        if factor is None:
            raise ValueError(f"Factor not found: {factor_id}")
        allowed = STATUS_TRANSITIONS.get(factor.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition {factor.status} → {new_status} "
                f"(allowed: {allowed})"
            )
        self._db.execute(
            "UPDATE factors SET status = ?, updated_at = ? WHERE factor_id = ?",
            [new_status, _now_iso(), factor_id],
        )

    # ── Register from FactorConfig ──

    def register_factors_from_config(
        self, config: FactorConfig,
        only_names: set[str] | None = None,
    ) -> tuple[int, int, int]:
        """Batch-register factors from a FactorConfig.

        Args:
            config: Factor configuration to register.
            only_names: If provided, only register factors whose name is in this set.

        Returns (new, updated, unchanged).
        """
        new = updated = unchanged = 0
        source = config.source
        for fdef in config.factors:
            if only_names is not None and fdef.name not in only_names:
                continue
            fid = generate_factor_id(source, fdef.name)
            record = FactorRecord(
                factor_id=fid,
                expression=fdef.expression,
                prototype=fdef.prototype,
                description=fdef.description,
                source=source,
                tags=fdef.tags,
                parameters=config.parameters,
                variables=config.variables,
            )
            result = self.upsert_factor(record)
            if result == "new":
                new += 1
            elif result == "updated":
                updated += 1
            else:
                unchanged += 1
        return new, updated, unchanged

    # ── Config snapshots ──

    def save_config_snapshot(
        self,
        config_dict: dict[str, Any],
        config_type: str,
        config_name: str = "",
        file_path: str = "",
    ) -> str:
        """Save a config snapshot. Returns the config_id.

        Deduplicates by content hash — same content is not stored twice.
        """
        h = _config_hash(config_dict)
        config_id = f"{config_type}_{h[:12]}"

        existing = self._db.fetch_one(
            "SELECT config_id FROM configs_snapshot WHERE config_id = ?",
            [config_id],
        )
        if existing is not None:
            return config_id

        self._db.execute(
            "INSERT INTO configs_snapshot "
            "(config_id, type, config_name, config_json, file_path, "
            " config_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                config_id,
                config_type,
                config_name,
                json.dumps(config_dict, ensure_ascii=False),
                file_path,
                h,
                _now_iso(),
            ],
        )
        return config_id

    def get_config_snapshot(self, config_id: str) -> ConfigSnapshot | None:
        row = self._db.fetch_one(
            "SELECT config_id, type, config_name, config_json, file_path, "
            "config_hash, created_at FROM configs_snapshot WHERE config_id = ?",
            [config_id],
        )
        if row is None:
            return None
        return ConfigSnapshot(
            config_id=row[0],
            type=row[1],
            config_name=row[2] or "",
            config_json=json.loads(row[3]) if row[3] else {},
            file_path=row[4] or "",
            config_hash=row[5] or "",
            created_at=str(row[6]) if row[6] else "",
        )

    # ── Analysis metrics ──

    def save_metrics(self, metrics: list[AnalysisMetrics]) -> None:
        """Save analysis metrics (INSERT OR REPLACE by PK)."""
        for m in metrics:
            self._db.execute(
                "INSERT OR REPLACE INTO alpha_analysis_metrics "
                "(run_id, factor_id, period, ic_mean, ic_std, icir, "
                " t_stat_ic, p_value_ic, t_stat_nw, p_value_nw, n_eff, "
                " ic_skew, ic_kurtosis, n_samples, "
                " win_rate, monotonicity, ic_half_life, ic_linearity, ic_ar1, "
                " coverage, mean_return, turnover, "
                " factor_config_id, analysis_config_id, "
                " output_dir, timeframe, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    m.run_id, m.factor_id, m.period,
                    m.ic_mean, m.ic_std, m.icir,
                    m.t_stat_ic, m.p_value_ic, m.t_stat_nw, m.p_value_nw,
                    m.n_eff, m.ic_skew, m.ic_kurtosis, m.n_samples,
                    m.win_rate, m.monotonicity, m.ic_half_life,
                    m.ic_linearity, m.ic_ar1,
                    m.coverage, m.mean_return, m.turnover,
                    m.factor_config_id, m.analysis_config_id,
                    m.output_dir, m.timeframe,
                    m.created_at or _now_iso(),
                ],
            )

    def get_metrics(
        self,
        factor_id: str,
        run_id: str | None = None,
        timeframe: str | None = None,
    ) -> list[AnalysisMetrics]:
        clauses = ["factor_id = ?"]
        params: list[Any] = [factor_id]
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if timeframe is not None:
            clauses.append("timeframe = ?")
            params.append(timeframe)

        where = " AND ".join(clauses)
        rows = self._db.fetch_all(
            "SELECT run_id, factor_id, period, ic_mean, ic_std, icir, "
            "t_stat_ic, p_value_ic, t_stat_nw, p_value_nw, n_eff, "
            "ic_skew, ic_kurtosis, n_samples, "
            "win_rate, monotonicity, ic_half_life, ic_linearity, ic_ar1, "
            "coverage, mean_return, turnover, "
            "factor_config_id, analysis_config_id, output_dir, timeframe, "
            "created_at "
            f"FROM alpha_analysis_metrics WHERE {where} "
            "ORDER BY created_at DESC, period",
            params,
        )
        return [_row_to_metrics(r) for r in rows]

    def get_latest_metrics(
        self, factor_id: str, timeframe: str,
    ) -> list[AnalysisMetrics]:
        """Get metrics from the most recent run for a factor + timeframe."""
        row = self._db.fetch_one(
            "SELECT run_id FROM alpha_analysis_metrics "
            "WHERE factor_id = ? AND timeframe = ? "
            "ORDER BY created_at DESC LIMIT 1",
            [factor_id, timeframe],
        )
        if row is None:
            return []
        return self.get_metrics(factor_id, run_id=row[0])

    def get_best_factors(
        self,
        timeframe: str,
        period: str,
        metric: str = "icir",
        limit: int = 20,
    ) -> list[AnalysisMetrics]:
        """Get top factors by a metric for a given timeframe and period."""
        allowed_metrics = {
            "icir", "ic_mean", "monotonicity", "win_rate", "ic_linearity",
        }
        if metric not in allowed_metrics:
            metric = "icir"

        rows = self._db.fetch_all(
            "SELECT run_id, factor_id, period, ic_mean, ic_std, icir, "
            "t_stat_ic, p_value_ic, t_stat_nw, p_value_nw, n_eff, "
            "ic_skew, ic_kurtosis, n_samples, "
            "win_rate, monotonicity, ic_half_life, ic_linearity, ic_ar1, "
            "coverage, mean_return, turnover, "
            "factor_config_id, analysis_config_id, output_dir, timeframe, "
            "created_at "
            "FROM alpha_analysis_metrics "
            f"WHERE timeframe = ? AND period = ? "
            f"ORDER BY ABS({metric}) DESC NULLS LAST "
            f"LIMIT {int(limit)}",
            [timeframe, period],
        )
        return [_row_to_metrics(r) for r in rows]


def _row_to_metrics(r: tuple) -> AnalysisMetrics:
    return AnalysisMetrics(
        run_id=r[0],
        factor_id=r[1],
        period=str(r[2]) if r[2] is not None else "",
        ic_mean=r[3],
        ic_std=r[4],
        icir=r[5],
        t_stat_ic=r[6],
        p_value_ic=r[7],
        t_stat_nw=r[8],
        p_value_nw=r[9],
        n_eff=r[10],
        ic_skew=r[11],
        ic_kurtosis=r[12],
        n_samples=r[13],
        win_rate=r[14],
        monotonicity=r[15],
        ic_half_life=r[16],
        ic_linearity=r[17],
        ic_ar1=r[18],
        coverage=r[19],
        mean_return=r[20],
        turnover=r[21],
        factor_config_id=r[22] or "",
        analysis_config_id=r[23] or "",
        output_dir=r[24] or "",
        timeframe=r[25] or "",
        created_at=str(r[26]) if r[26] else "",
    )
