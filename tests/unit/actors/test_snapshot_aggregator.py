# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for SnapshotAggregatorActor helper functions and snapshot building."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import msgspec
import pytest

from nautilus_quants.actors.snapshot_aggregator import (
    SnapshotAggregatorActorConfig,
    _elapsed_ms,
    _empty_exec_agg,
    _tail_log_entries,
)
from nautilus_quants.execution.post_limit.state import (
    OrderExecutionStateSnapshot,
    OrderExecutionStateStore,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = SnapshotAggregatorActorConfig()
        assert cfg.venue_name == "OKX"
        assert cfg.currency == "USDT"
        assert cfg.interval_secs == 15
        assert cfg.strategy_ids == []
        assert cfg.log_directory == "logs"
        assert cfg.max_log_entries == 50

    def test_custom(self):
        cfg = SnapshotAggregatorActorConfig(
            venue_name="BINANCE",
            currency="BTC",
            interval_secs=30,
            strategy_ids=["S-001"],
            log_directory="logs/test",
            max_log_entries=10,
        )
        assert cfg.venue_name == "BINANCE"
        assert cfg.interval_secs == 30
        assert cfg.strategy_ids == ["S-001"]


# ---------------------------------------------------------------------------
# Elapsed ms helper
# ---------------------------------------------------------------------------


class TestElapsedMs:
    def test_completed(self):
        result = _elapsed_ms(1_000_000_000, 2_000_000_000, 3_000_000_000)
        assert result == pytest.approx(1000.0)

    def test_in_progress(self):
        result = _elapsed_ms(1_000_000_000, 0, 3_000_000_000)
        assert result == pytest.approx(2000.0)

    def test_no_timestamps(self):
        assert _elapsed_ms(0, 0, 3_000_000_000) == 0.0


# ---------------------------------------------------------------------------
# Empty exec aggregate
# ---------------------------------------------------------------------------


class TestEmptyExecAgg:
    def test_structure(self):
        agg = _empty_exec_agg()
        assert agg["active_sessions"] == 0
        assert agg["limit_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Execution state msgpack decoding
# ---------------------------------------------------------------------------


class TestExecutionSnapshotDecoding:
    """Test that we can decode msgpack execution states."""

    def _make_store(self, sessions: list[OrderExecutionStateSnapshot]) -> bytes:
        store = OrderExecutionStateStore(version=1, sessions=sessions)
        return msgspec.msgpack.encode(store)

    def test_empty_store(self):
        raw = self._make_store([])
        store = msgspec.msgpack.decode(raw, type=OrderExecutionStateStore)
        assert store.sessions == []
        assert store.version == 1

    def test_single_session(self):
        snap = OrderExecutionStateSnapshot(
            primary_order_id="O-001",
            instrument_id="BTC-USDT-SWAP.OKX",
            side="BUY",
            total_quantity="1.0",
            anchor_px=50000.0,
            reduce_only=False,
            state="WORKING_LIMIT",
            active_order_id="O-001-E1",
            active_order_kind="LIMIT",
            active_reserved_quantity="1.0",
            active_order_accepted=True,
            chase_count=2,
            spawn_sequence=3,
            timer_name="timeout_O-001",
            created_ns=1_000_000_000,
            completed_ns=0,
            used_market_fallback=False,
            residual_sweep_pending=False,
            timeout_secs=15.0,
            max_chase_attempts=3,
            chase_step_ticks=None,
            post_only=True,
            post_only_retreat_ticks=0,
            target_quote_quantity=50000.0,
            filled_quote_quantity=25000.0,
            contract_multiplier=1.0,
            transient_retry_count=0,
            limit_orders_submitted=3,
            last_limit_price=50000.0,
            filled_quantity="0.5",
            fill_cost=25000.0,
        )
        raw = self._make_store([snap])
        store = msgspec.msgpack.decode(raw, type=OrderExecutionStateStore)
        assert len(store.sessions) == 1
        s = store.sessions[0]
        assert s.instrument_id == "BTC-USDT-SWAP.OKX"
        assert s.chase_count == 2
        assert s.state == "WORKING_LIMIT"
        assert s.used_market_fallback is False


# ---------------------------------------------------------------------------
# Log tail
# ---------------------------------------------------------------------------


class TestLogTail:
    def _write_log(self, path: Path, lines: list[str]) -> None:
        with open(path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    def test_empty_dir(self, tmp_path):
        entries = _tail_log_entries(tmp_path, 60, 50)
        assert entries == []

    def test_parse_warn_and_error(self, tmp_path):
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000000000", time.gmtime())
        lines = [
            f"{now}Z [WARN] T-001.PostLimit: limit-child prepared",
            f"{now}Z [ERROR] T-001.redis::cache: timed out",
            f"{now}Z [INFO] T-001.SomeActor: normal message",
            f"{now}Z [DEBUG] T-001.h2: frame received",
        ]
        log_file = tmp_path / "T-001_2026-04-03.log"
        self._write_log(log_file, lines)

        entries = _tail_log_entries(tmp_path, 60, 50)
        assert len(entries) == 2
        assert entries[0]["level"] == "WARN"
        assert entries[0]["component"] == "PostLimit"
        assert "limit-child prepared" in entries[0]["message"]
        assert entries[1]["level"] == "ERROR"
        assert entries[1]["component"] == "redis::cache"

    def test_respects_time_window(self, tmp_path):
        old_ts = "2020-01-01T00:00:00.000000000"
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000000000", time.gmtime())
        lines = [
            f"{old_ts}Z [ERROR] T-001.old: ancient error",
            f"{now}Z [ERROR] T-001.new: recent error",
        ]
        log_file = tmp_path / "T-001_2026-04-03.log"
        self._write_log(log_file, lines)

        entries = _tail_log_entries(tmp_path, 60, 50)
        assert len(entries) == 1
        assert entries[0]["component"] == "new"

    def test_max_entries_limit(self, tmp_path):
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000000000", time.gmtime())
        lines = [f"{now}Z [WARN] T-001.comp{i}: msg {i}" for i in range(100)]
        log_file = tmp_path / "T-001_2026-04-03.log"
        self._write_log(log_file, lines)

        entries = _tail_log_entries(tmp_path, 60, 5)
        assert len(entries) == 5

    def test_message_truncation(self, tmp_path):
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000000000", time.gmtime())
        long_msg = "x" * 500
        lines = [f"{now}Z [ERROR] T-001.comp: {long_msg}"]
        log_file = tmp_path / "T-001_2026-04-03.log"
        self._write_log(log_file, lines)

        entries = _tail_log_entries(tmp_path, 60, 50)
        assert len(entries) == 1
        assert len(entries[0]["message"]) == 200


# ---------------------------------------------------------------------------
# Factor snapshot rank computation
# ---------------------------------------------------------------------------


class TestFactorRankComputation:
    """Test rank computation logic (extracted from _build_factor_snapshot)."""

    def test_rank_descending(self):
        """Ranks should be descending (highest value = rank 1)."""
        import math

        factors = {
            "alpha001": {"A": 0.5, "B": -0.3, "C": 0.8},
        }
        ranks: dict[str, dict[str, int]] = {}
        for fname, fvals in factors.items():
            numeric = {
                k: v for k, v in fvals.items() if isinstance(v, (int, float)) and not math.isnan(v)
            }
            sorted_items = sorted(numeric.items(), key=lambda x: (-x[1], x[0]))
            ranks[fname] = {inst: rank for rank, (inst, _) in enumerate(sorted_items, 1)}

        assert ranks["alpha001"]["C"] == 1  # 0.8 highest
        assert ranks["alpha001"]["A"] == 2  # 0.5
        assert ranks["alpha001"]["B"] == 3  # -0.3 lowest

    def test_nan_filtered(self):
        import math

        factors = {"f1": {"A": 1.0, "B": float("nan"), "C": 0.5}}
        numeric = {
            k: v
            for k, v in factors["f1"].items()
            if isinstance(v, (int, float)) and not math.isnan(v)
        }
        assert "B" not in numeric
        assert len(numeric) == 2
