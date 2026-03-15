# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for execution report CSV generation."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.backtest.config import ReportConfig, TearsheetConfig
from nautilus_quants.backtest.protocols import EXECUTION_STATES_CACHE_KEY
from nautilus_quants.backtest.reports import ReportGenerator
from nautilus_quants.execution.post_limit.state import OrderExecutionState, OrderState


def _make_state(
    order_id: str = "O-001",
    instrument: str = "BTCUSDT-PERP.BINANCE",
    side: OrderSide = OrderSide.BUY,
    total_qty: str = "1.0",
    filled_qty: str = "1.0",
    anchor_px: float = 50000.0,
    state: OrderState = OrderState.COMPLETED,
    chase_count: int = 0,
    limit_orders_submitted: int = 1,
    last_limit_price: float = 49999.0,
    fill_cost: float = 0.0,
    reduce_only: bool = False,
    created_ns: int = 1_000_000_000,
    completed_ns: int = 2_000_000_000,
    used_market_fallback: bool = False,
    timeout_secs: float | None = None,
    max_chase_attempts: int | None = None,
    chase_step_ticks: int | None = None,
    post_only: bool | None = None,
) -> OrderExecutionState:
    """Helper to create an OrderExecutionState for testing."""
    s = OrderExecutionState(
        primary_order_id=ClientOrderId(order_id),
        instrument_id=InstrumentId.from_str(instrument),
        side=side,
        total_quantity=Quantity.from_str(total_qty),
        anchor_px=anchor_px,
        reduce_only=reduce_only,
        state=state,
        chase_count=chase_count,
        limit_orders_submitted=limit_orders_submitted,
        last_limit_price=last_limit_price,
        fill_cost=fill_cost,
        created_ns=created_ns,
        completed_ns=completed_ns,
        used_market_fallback=used_market_fallback,
        filled_quantity=Quantity.from_str(filled_qty),
        timeout_secs=timeout_secs,
        max_chase_attempts=max_chase_attempts,
        chase_step_ticks=chase_step_ticks,
        post_only=post_only,
    )
    return s


def _make_generator(tmp_path: Path, cache_data: bytes | None = None) -> ReportGenerator:
    """Create a ReportGenerator with a mock engine."""
    engine = MagicMock()

    def mock_cache_get(key: str) -> bytes | None:
        if key == EXECUTION_STATES_CACHE_KEY:
            return cache_data
        return None

    engine.cache.get.side_effect = mock_cache_get
    engine.cache.orders.return_value = []
    engine.cache.positions.return_value = []
    engine.cache.accounts.return_value = []
    engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {}
    engine.portfolio.analyzer.get_performance_stats_returns.return_value = {}
    engine.portfolio.analyzer.get_performance_stats_general.return_value = {}

    config = ReportConfig(
        output_dir=str(tmp_path),
        formats=["csv"],
        tearsheet=TearsheetConfig(enabled=False),
    )
    return ReportGenerator(engine=engine, output_dir=tmp_path, config=config)


class TestGenerateExecutionReportCsv:
    """Tests for ReportGenerator.generate_execution_report_csv()."""

    def test_returns_none_when_no_data(self, tmp_path: Path) -> None:
        """Cache has no execution states data -> returns None."""
        generator = _make_generator(tmp_path, cache_data=None)
        result = generator.generate_execution_report_csv()
        assert result is None

    def test_returns_none_when_empty_states(self, tmp_path: Path) -> None:
        """Cache contains pickled empty dict -> returns None."""
        data = pickle.dumps({})
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()
        assert result is None

    def test_csv_columns_and_values(self, tmp_path: Path) -> None:
        """Verify CSV has correct columns and values for multiple states."""
        states = {
            ClientOrderId("O-001"): _make_state(
                order_id="O-001",
                total_qty="2.0",
                filled_qty="2.0",
                chase_count=1,
                limit_orders_submitted=2,
                anchor_px=50000.0,
                last_limit_price=50001.0,
                fill_cost=99998.0,  # avg_fill_px = 99998/2 = 49999
                created_ns=1_000_000_000,
                completed_ns=1_500_000_000,
            ),
            ClientOrderId("O-002"): _make_state(
                order_id="O-002",
                side=OrderSide.SELL,
                total_qty="0.5",
                filled_qty="0.5",
                anchor_px=60000.0,
                last_limit_price=59999.0,
                fill_cost=30001.0,  # avg_fill_px = 30001/0.5 = 60002
                reduce_only=True,
                state=OrderState.COMPLETED,
                created_ns=2_000_000_000,
                completed_ns=3_000_000_000,
                timeout_secs=5.0,
                max_chase_attempts=3,
                chase_step_ticks=2,
                post_only=True,
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        assert result is not None
        assert result.exists()
        assert result.name == "execution_report.csv"

        # Read CSV and verify
        with open(result) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

        expected_columns = {
            "primary_order_id",
            "instrument_id",
            "side",
            "total_quantity",
            "filled_quantity",
            "fill_ratio",
            "anchor_px",
            "last_limit_price",
            "avg_fill_px",
            "slippage",
            "slippage_bps",
            "reduce_only",
            "final_state",
            "chase_count",
            "limit_orders_submitted",
            "used_market_fallback",
            "created_ns",
            "completed_ns",
            "elapsed_ms",
            "timeout_secs",
            "max_chase_attempts",
            "chase_step_ticks",
            "post_only",
        }
        assert set(rows[0].keys()) == expected_columns

        # Verify first row values
        row0 = rows[0]
        assert row0["primary_order_id"] == "O-001"
        assert row0["side"] == "BUY"
        assert float(row0["total_quantity"]) == 2.0
        assert float(row0["filled_quantity"]) == 2.0
        assert row0["final_state"] == "COMPLETED"
        assert int(row0["chase_count"]) == 1
        assert int(row0["limit_orders_submitted"]) == 2

        # Verify second row with overrides
        row1 = rows[1]
        assert row1["primary_order_id"] == "O-002"
        assert row1["side"] == "SELL"
        assert row1["reduce_only"] == "True"
        assert float(row1["timeout_secs"]) == 5.0
        assert float(row1["max_chase_attempts"]) == 3.0
        assert float(row1["chase_step_ticks"]) == 2.0
        assert row1["post_only"] == "True"

    def test_derived_fields_computed(self, tmp_path: Path) -> None:
        """Verify fill_ratio, elapsed_ms, used_market_fallback computed correctly."""
        states = {
            ClientOrderId("O-PARTIAL"): _make_state(
                order_id="O-PARTIAL",
                total_qty="10.0",
                filled_qty="7.5",
                state=OrderState.COMPLETED,
                created_ns=1_000_000_000,
                completed_ns=1_100_000_000,
                used_market_fallback=True,
                chase_count=3,
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        assert result is not None
        with open(result) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        # fill_ratio = 7.5 / 10.0 = 0.75
        assert float(row["fill_ratio"]) == pytest.approx(0.75, abs=1e-6)

        # elapsed_ms = (1_100_000_000 - 1_000_000_000) / 1_000_000 = 100.0
        assert float(row["elapsed_ms"]) == pytest.approx(100.0, abs=0.001)

        # used_market_fallback = True
        assert row["used_market_fallback"] == "True"

    def test_generate_all_includes_execution_report(self, tmp_path: Path) -> None:
        """generate_all() should include execution_report when data is present."""
        states = {
            ClientOrderId("O-001"): _make_state(),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)

        reports = generator.generate_all()

        assert "execution_report" in reports
        assert reports["execution_report"].exists()

    def test_generate_all_excludes_when_no_data(self, tmp_path: Path) -> None:
        """generate_all() should not include execution_report when no data."""
        generator = _make_generator(tmp_path, cache_data=None)

        reports = generator.generate_all()

        assert "execution_report" not in reports


class TestSlippageCalculation:
    """Tests for avg_fill_px, slippage, and slippage_bps columns."""

    def test_buy_negative_slippage_price_improvement(self, tmp_path: Path) -> None:
        """BUY filled below anchor -> negative slippage (favorable)."""
        # anchor=2045, avg_fill=2044 -> slippage = 2044 - 2045 = -1
        states = {
            ClientOrderId("O-BUY"): _make_state(
                order_id="O-BUY",
                side=OrderSide.BUY,
                total_qty="1.0",
                filled_qty="1.0",
                anchor_px=2045.0,
                fill_cost=2044.0,  # avg_fill_px = 2044
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        with open(result) as f:
            row = list(csv.DictReader(f))[0]

        assert float(row["avg_fill_px"]) == pytest.approx(2044.0)
        assert float(row["slippage"]) == pytest.approx(-1.0)
        assert float(row["slippage_bps"]) == pytest.approx(-1.0 / 2045.0 * 10000, abs=0.01)

    def test_buy_positive_slippage_adverse(self, tmp_path: Path) -> None:
        """BUY filled above anchor -> positive slippage (adverse)."""
        # anchor=2045, avg_fill=2046 -> slippage = 2046 - 2045 = +1
        states = {
            ClientOrderId("O-BUY2"): _make_state(
                order_id="O-BUY2",
                side=OrderSide.BUY,
                total_qty="1.0",
                filled_qty="1.0",
                anchor_px=2045.0,
                fill_cost=2046.0,  # avg_fill_px = 2046
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        with open(result) as f:
            row = list(csv.DictReader(f))[0]

        assert float(row["avg_fill_px"]) == pytest.approx(2046.0)
        assert float(row["slippage"]) == pytest.approx(1.0)
        assert float(row["slippage_bps"]) > 0

    def test_sell_negative_slippage_price_improvement(self, tmp_path: Path) -> None:
        """SELL filled above anchor -> negative slippage (favorable)."""
        # anchor=2023, avg_fill=2024 -> slippage = 2023 - 2024 = -1
        states = {
            ClientOrderId("O-SELL"): _make_state(
                order_id="O-SELL",
                side=OrderSide.SELL,
                total_qty="1.0",
                filled_qty="1.0",
                anchor_px=2023.0,
                fill_cost=2024.0,  # avg_fill_px = 2024
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        with open(result) as f:
            row = list(csv.DictReader(f))[0]

        assert float(row["avg_fill_px"]) == pytest.approx(2024.0)
        assert float(row["slippage"]) == pytest.approx(-1.0)
        assert float(row["slippage_bps"]) < 0

    def test_sell_positive_slippage_adverse(self, tmp_path: Path) -> None:
        """SELL filled below anchor -> positive slippage (adverse)."""
        # anchor=2023, avg_fill=2022 -> slippage = 2023 - 2022 = +1
        states = {
            ClientOrderId("O-SELL2"): _make_state(
                order_id="O-SELL2",
                side=OrderSide.SELL,
                total_qty="1.0",
                filled_qty="1.0",
                anchor_px=2023.0,
                fill_cost=2022.0,  # avg_fill_px = 2022
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        with open(result) as f:
            row = list(csv.DictReader(f))[0]

        assert float(row["avg_fill_px"]) == pytest.approx(2022.0)
        assert float(row["slippage"]) == pytest.approx(1.0)
        assert float(row["slippage_bps"]) > 0

    def test_zero_filled_qty_gives_zero_slippage(self, tmp_path: Path) -> None:
        """Zero filled quantity -> avg_fill_px=0, slippage=0, slippage_bps=0."""
        states = {
            ClientOrderId("O-ZERO"): _make_state(
                order_id="O-ZERO",
                filled_qty="0.0",
                anchor_px=50000.0,
                fill_cost=0.0,
                state=OrderState.FAILED,
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        with open(result) as f:
            row = list(csv.DictReader(f))[0]

        assert float(row["avg_fill_px"]) == 0.0
        assert float(row["slippage"]) == 0.0
        assert float(row["slippage_bps"]) == 0.0

    def test_multi_fill_vwap(self, tmp_path: Path) -> None:
        """Multiple fills: fill_cost accumulates correctly for VWAP."""
        # 2 fills: 0.5 @ 2044 + 0.5 @ 2046 -> fill_cost = 1022 + 1023 = 2045
        # avg_fill_px = 2045 / 1.0 = 2045, anchor = 2045 -> slippage = 0
        states = {
            ClientOrderId("O-MULTI"): _make_state(
                order_id="O-MULTI",
                side=OrderSide.BUY,
                total_qty="1.0",
                filled_qty="1.0",
                anchor_px=2045.0,
                fill_cost=2045.0,  # avg_fill_px = 2045
            ),
        }
        data = pickle.dumps(states)
        generator = _make_generator(tmp_path, cache_data=data)
        result = generator.generate_execution_report_csv()

        with open(result) as f:
            row = list(csv.DictReader(f))[0]

        assert float(row["avg_fill_px"]) == pytest.approx(2045.0)
        assert float(row["slippage"]) == pytest.approx(0.0)
        assert float(row["slippage_bps"]) == pytest.approx(0.0)
