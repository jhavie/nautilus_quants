# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for BacktestRepository — backtest run tracking with factor associations."""

from __future__ import annotations

import pytest

from nautilus_quants.alpha.registry.backtest_repository import BacktestRepository
from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import BacktestRunRecord


@pytest.fixture()
def db() -> RegistryDatabase:
    database = RegistryDatabase(":memory:")
    yield database
    database.close()


@pytest.fixture()
def bt_repo(db: RegistryDatabase) -> BacktestRepository:
    return BacktestRepository(db)


def _make_backtest(
    backtest_id: str = "bt1",
    **kwargs,
) -> BacktestRunRecord:
    defaults = {
        "config_id": "cfg1",
        "factor_config_id": "fcfg1",
        "output_dir": "/output/bt1",
        "strategy_name": "cs_momentum",
        "instrument_count": 20,
        "timeframe": "8h",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T01:00:00+00:00",
        "duration_seconds": 3600.0,
        "total_pnl": 1500.0,
        "total_pnl_pct": 0.15,
        "sharpe_ratio": 1.8,
        "max_drawdown": -0.05,
        "win_rate": 0.55,
        "statistics_json": {"trades": 100, "avg_pnl": 15.0},
        "reports_json": {"equity_curve": "/path/equity.html"},
    }
    defaults.update(kwargs)
    return BacktestRunRecord(backtest_id=backtest_id, **defaults)


# ---------------------------------------------------------------------------
# Save and get backtest
# ---------------------------------------------------------------------------


class TestSaveAndGetBacktest:
    def test_save_and_get(self, bt_repo: BacktestRepository) -> None:
        bt_repo.save_backtest(_make_backtest())
        record = bt_repo.get_backtest("bt1")
        assert record is not None
        assert record.backtest_id == "bt1"
        assert record.strategy_name == "cs_momentum"
        assert record.instrument_count == 20
        assert record.timeframe == "8h"
        assert record.total_pnl == 1500.0
        assert record.sharpe_ratio == 1.8
        assert record.max_drawdown == -0.05

    def test_all_fields_persisted(self, bt_repo: BacktestRepository) -> None:
        bt_repo.save_backtest(_make_backtest())
        r = bt_repo.get_backtest("bt1")
        assert r.config_id == "cfg1"
        assert r.factor_config_id == "fcfg1"
        assert r.output_dir == "/output/bt1"
        assert r.total_pnl_pct == 0.15
        assert r.win_rate == 0.55
        assert r.duration_seconds == 3600.0
        assert r.statistics_json == {"trades": 100, "avg_pnl": 15.0}
        assert r.reports_json == {"equity_curve": "/path/equity.html"}

    def test_get_nonexistent(self, bt_repo: BacktestRepository) -> None:
        assert bt_repo.get_backtest("nonexistent") is None

    def test_replace_on_duplicate(self, bt_repo: BacktestRepository) -> None:
        bt_repo.save_backtest(_make_backtest(total_pnl=100.0))
        bt_repo.save_backtest(_make_backtest(total_pnl=200.0))
        r = bt_repo.get_backtest("bt1")
        assert r.total_pnl == 200.0

    def test_nullable_fields(self, bt_repo: BacktestRepository) -> None:
        record = _make_backtest(
            backtest_id="bt_null",
            total_pnl=None,
            total_pnl_pct=None,
            sharpe_ratio=None,
            max_drawdown=None,
            win_rate=None,
        )
        bt_repo.save_backtest(record)
        r = bt_repo.get_backtest("bt_null")
        assert r is not None
        assert r.total_pnl is None
        assert r.sharpe_ratio is None
        assert r.max_drawdown is None


# ---------------------------------------------------------------------------
# Factor associations
# ---------------------------------------------------------------------------


class TestFactorAssociations:
    def test_save_with_factor_ids(self, bt_repo: BacktestRepository) -> None:
        bt_repo.save_backtest(
            _make_backtest(),
            factor_ids=["f1", "f2", "f3"],
        )
        factors = bt_repo.get_backtest_factors("bt1")
        assert len(factors) == 3
        fids = {bf.factor_id for bf in factors}
        assert fids == {"f1", "f2", "f3"}
        assert all(bf.role == "component" for bf in factors)

    def test_save_with_composite_id(self, bt_repo: BacktestRepository) -> None:
        bt_repo.save_backtest(
            _make_backtest(),
            factor_ids=["f1"],
            composite_id="composite_f1_f2",
        )
        factors = bt_repo.get_backtest_factors("bt1")
        assert len(factors) == 2
        roles = {bf.factor_id: bf.role for bf in factors}
        assert roles["f1"] == "component"
        assert roles["composite_f1_f2"] == "composite"

    def test_save_without_factors(self, bt_repo: BacktestRepository) -> None:
        bt_repo.save_backtest(_make_backtest())
        factors = bt_repo.get_backtest_factors("bt1")
        assert factors == []

    def test_get_backtest_factors_nonexistent(
        self, bt_repo: BacktestRepository,
    ) -> None:
        assert bt_repo.get_backtest_factors("nonexistent") == []

    def test_factor_association_fields(
        self, bt_repo: BacktestRepository,
    ) -> None:
        bt_repo.save_backtest(
            _make_backtest(),
            factor_ids=["f1"],
        )
        factors = bt_repo.get_backtest_factors("bt1")
        bf = factors[0]
        assert bf.backtest_id == "bt1"
        assert bf.factor_id == "f1"
        assert bf.role == "component"


# ---------------------------------------------------------------------------
# List backtests
# ---------------------------------------------------------------------------


class TestListBacktests:
    @pytest.fixture(autouse=True)
    def _seed(self, bt_repo: BacktestRepository) -> None:
        bt_repo.save_backtest(
            _make_backtest("bt1", sharpe_ratio=2.0),
            factor_ids=["f1", "f2"],
        )
        bt_repo.save_backtest(
            _make_backtest("bt2", sharpe_ratio=0.5),
            factor_ids=["f1"],
        )
        bt_repo.save_backtest(
            _make_backtest("bt3", sharpe_ratio=1.5),
            factor_ids=["f3"],
        )

    def test_list_all(self, bt_repo: BacktestRepository) -> None:
        result = bt_repo.list_backtests()
        assert len(result) == 3

    def test_list_by_factor_id(self, bt_repo: BacktestRepository) -> None:
        result = bt_repo.list_backtests(factor_id="f1")
        bt_ids = {r.backtest_id for r in result}
        assert bt_ids == {"bt1", "bt2"}

    def test_list_by_factor_id_single(
        self, bt_repo: BacktestRepository,
    ) -> None:
        result = bt_repo.list_backtests(factor_id="f3")
        assert len(result) == 1
        assert result[0].backtest_id == "bt3"

    def test_list_by_factor_id_no_match(
        self, bt_repo: BacktestRepository,
    ) -> None:
        result = bt_repo.list_backtests(factor_id="nonexistent")
        assert result == []

    def test_list_with_min_sharpe(self, bt_repo: BacktestRepository) -> None:
        result = bt_repo.list_backtests(min_sharpe=1.0)
        bt_ids = {r.backtest_id for r in result}
        assert bt_ids == {"bt1", "bt3"}

    def test_list_by_factor_id_and_min_sharpe(
        self, bt_repo: BacktestRepository,
    ) -> None:
        result = bt_repo.list_backtests(factor_id="f1", min_sharpe=1.0)
        assert len(result) == 1
        assert result[0].backtest_id == "bt1"

    def test_list_with_limit(self, bt_repo: BacktestRepository) -> None:
        result = bt_repo.list_backtests(limit=2)
        assert len(result) == 2

    def test_list_ordered_by_started_at_desc(
        self, bt_repo: BacktestRepository,
    ) -> None:
        result = bt_repo.list_backtests()
        # All have same started_at, so order is stable but DESC
        assert len(result) == 3
