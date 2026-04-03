# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""BacktestRepository — backtest run tracking with factor associations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import (
    BacktestFactor,
    BacktestRunRecord,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BacktestRepository:
    """Backtest run persistence with M:N factor associations."""

    def __init__(self, db: RegistryDatabase) -> None:
        self._db = db

    def save_backtest(
        self,
        record: BacktestRunRecord,
        factor_ids: list[str] | None = None,
        composite_id: str | None = None,
    ) -> None:
        """Save a backtest run and its factor associations.

        Parameters
        ----------
        record : BacktestRunRecord
            The backtest run data.
        factor_ids : list[str] | None
            Factor IDs used in this backtest (role=component).
        composite_id : str | None
            The composite factor ID (role=composite), if any.
        """
        self._db.execute(
            "INSERT OR REPLACE INTO backtest_run_metrics "
            "(backtest_id, config_id, factor_config_id, output_dir, "
            " strategy_name, instrument_count, timeframe, "
            " started_at, finished_at, duration_seconds, "
            " total_pnl, total_pnl_pct, sharpe_ratio, max_drawdown, "
            " win_rate, statistics_json, reports_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.backtest_id,
                record.config_id,
                record.factor_config_id,
                record.output_dir,
                record.strategy_name,
                record.instrument_count,
                record.timeframe,
                record.started_at or _now_iso(),
                record.finished_at or None,
                record.duration_seconds,
                record.total_pnl,
                record.total_pnl_pct,
                record.sharpe_ratio,
                record.max_drawdown,
                record.win_rate,
                json.dumps(record.statistics_json, ensure_ascii=False),
                json.dumps(record.reports_json, ensure_ascii=False),
            ],
        )

        # Write factor associations
        all_factors: list[BacktestFactor] = []
        for fid in (factor_ids or []):
            all_factors.append(
                BacktestFactor(record.backtest_id, fid, "component"),
            )
        if composite_id:
            all_factors.append(
                BacktestFactor(record.backtest_id, composite_id, "composite"),
            )

        for bf in all_factors:
            self._db.execute(
                "INSERT OR REPLACE INTO backtest_factors "
                "(backtest_id, factor_id, role) VALUES (?, ?, ?)",
                [bf.backtest_id, bf.factor_id, bf.role],
            )

    def get_backtest(self, backtest_id: str) -> BacktestRunRecord | None:
        row = self._db.fetch_one(
            "SELECT backtest_id, config_id, factor_config_id, output_dir, "
            "strategy_name, instrument_count, timeframe, "
            "started_at, finished_at, duration_seconds, "
            "total_pnl, total_pnl_pct, sharpe_ratio, max_drawdown, "
            "win_rate, statistics_json, reports_json "
            "FROM backtest_run_metrics WHERE backtest_id = ?",
            [backtest_id],
        )
        if row is None:
            return None
        return _row_to_record(row)

    def list_backtests(
        self,
        factor_id: str | None = None,
        min_sharpe: float | None = None,
        limit: int | None = None,
    ) -> list[BacktestRunRecord]:
        """List backtests, optionally filtered by factor or Sharpe."""
        if factor_id is not None:
            clauses = ["bf.factor_id = ?"]
            params: list[Any] = [factor_id]
            if min_sharpe is not None:
                clauses.append("brm.sharpe_ratio >= ?")
                params.append(min_sharpe)
            where = " AND ".join(clauses)
            sql = (
                "SELECT DISTINCT brm.backtest_id, brm.config_id, "
                "brm.factor_config_id, brm.output_dir, "
                "brm.strategy_name, brm.instrument_count, brm.timeframe, "
                "brm.started_at, brm.finished_at, brm.duration_seconds, "
                "brm.total_pnl, brm.total_pnl_pct, brm.sharpe_ratio, "
                "brm.max_drawdown, brm.win_rate, "
                "brm.statistics_json, brm.reports_json "
                "FROM backtest_run_metrics brm "
                "JOIN backtest_factors bf ON brm.backtest_id = bf.backtest_id "
                f"WHERE {where} "
                "ORDER BY brm.started_at DESC"
            )
        else:
            params = []
            clauses_simple: list[str] = []
            if min_sharpe is not None:
                clauses_simple.append("sharpe_ratio >= ?")
                params.append(min_sharpe)
            where_s = (
                " WHERE " + " AND ".join(clauses_simple)
                if clauses_simple else ""
            )
            sql = (
                "SELECT backtest_id, config_id, factor_config_id, output_dir, "
                "strategy_name, instrument_count, timeframe, "
                "started_at, finished_at, duration_seconds, "
                "total_pnl, total_pnl_pct, sharpe_ratio, max_drawdown, "
                "win_rate, statistics_json, reports_json "
                f"FROM backtest_run_metrics{where_s} "
                "ORDER BY started_at DESC"
            )

        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self._db.fetch_all(sql, params if params else None)
        return [_row_to_record(r) for r in rows]

    def get_backtest_factors(
        self, backtest_id: str,
    ) -> list[BacktestFactor]:
        rows = self._db.fetch_all(
            "SELECT backtest_id, factor_id, role "
            "FROM backtest_factors WHERE backtest_id = ?",
            [backtest_id],
        )
        return [BacktestFactor(*r) for r in rows]


def _row_to_record(r: tuple) -> BacktestRunRecord:
    return BacktestRunRecord(
        backtest_id=r[0],
        config_id=r[1] or "",
        factor_config_id=r[2] or "",
        output_dir=r[3] or "",
        strategy_name=r[4] or "",
        instrument_count=r[5] or 0,
        timeframe=r[6] or "",
        started_at=str(r[7]) if r[7] else "",
        finished_at=str(r[8]) if r[8] else "",
        duration_seconds=r[9] or 0.0,
        total_pnl=r[10],
        total_pnl_pct=r[11],
        sharpe_ratio=r[12],
        max_drawdown=r[13],
        win_rate=r[14],
        statistics_json=json.loads(r[15]) if r[15] else {},
        reports_json=json.loads(r[16]) if r[16] else {},
    )
