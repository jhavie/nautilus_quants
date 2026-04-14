# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
SnapshotAggregatorActor — Unified Grafana monitoring snapshots.

Subscribes to FactorValues and RebalanceOrders via MessageBus, reads
execution states and portfolio data from cache, and periodically writes
5 JSON snapshots to Redis for Telegraf → Prometheus → Grafana consumption.

Usage (YAML ``engine.actors`` config):
    - actor_path: "nautilus_quants.actors.snapshot_aggregator:SnapshotAggregatorActor"
      config_path: "nautilus_quants.actors.snapshot_aggregator:SnapshotAggregatorActorConfig"
      config:
        venue_name: "OKX"
        currency: "USDT"
        interval_secs: 15
        log_directory: "logs/testnet_15m"
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import msgspec
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import ClientId, Venue
from nautilus_trader.model.objects import Currency

from nautilus_quants.execution.post_limit.state import OrderExecutionStateStore
from nautilus_quants.factors.factor_values import FactorValues
from nautilus_quants.portfolio.monitor.exposure import compute_portfolio_exposure
from nautilus_quants.portfolio.types import deserialize_risk_output
from nautilus_quants.strategies.cs.types import RebalanceOrders
from nautilus_quants.utils.cache_keys import (
    EXECUTION_STATES_CACHE_KEY,
    RISK_MODEL_STATE_CACHE_KEY,
    SNAPSHOT_EXECUTION_CACHE_KEY,
    SNAPSHOT_FACTOR_CACHE_KEY,
    SNAPSHOT_HEALTH_CACHE_KEY,
    SNAPSHOT_RISK_CACHE_KEY,
    SNAPSHOT_STRATEGY_CACHE_KEY,
    SNAPSHOT_VENUE_CACHE_KEY,
    STRATEGY_CONFIG_CACHE_KEY,
)
from nautilus_quants.utils.equity import compute_mtm_equity

# Regex for NautilusTrader log lines:
# 2026-04-03T00:45:56.369885345Z [ERROR] TESTNET-ALPHA101-001.redis::cache: timed out
_LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z "
    r"\[(WARN|ERROR)\] "
    r"[^.]+\.(.+?): "
    r"(.*)$",
)

_TAIL_BYTES = 256 * 1024  # Read last 256 KB of log file (first-run bootstrap only)
_ERROR_TTL_SECS = 300  # Keep ERROR entries for 5 minutes


class SnapshotAggregatorActorConfig(ActorConfig, frozen=True):
    """Configuration for SnapshotAggregatorActor.

    Parameters
    ----------
    venue_name : str, default "OKX"
        Venue name for portfolio/account lookup.
    currency : str, default "USDT"
        Settlement currency code.
    interval_secs : int, default 15
        Snapshot write interval in seconds.
    strategy_ids : list[str], default []
        Associated strategy IDs (for future filtering).
    log_directory : str, default "logs"
        Log directory path for health snapshot.
    max_log_entries : int, default 50
        Maximum WARN/ERROR log entries per health snapshot.
    """

    venue_name: str = "OKX"
    currency: str = "USDT"
    interval_secs: int = 15
    strategy_ids: list[str] = []
    log_directory: str = "logs"
    max_log_entries: int = 50


class SnapshotAggregatorActor(Actor):
    """Aggregate trading system state into JSON snapshots for Grafana.

    Writes 5 Redis keys every ``interval_secs`` seconds:
    - ``snapshot:venue``     — positions, account balance, PnL
    - ``snapshot:execution`` — PostLimit session states, chase/market stats
    - ``snapshot:factor``    — factor values and ranks
    - ``snapshot:strategy``  — rebalance decisions, position diff
    - ``snapshot:health``    — recent WARN/ERROR log entries
    """

    def __init__(self, config: SnapshotAggregatorActorConfig) -> None:
        super().__init__(config)
        self._venue = Venue(config.venue_name)
        self._currency = Currency.from_str(config.currency)
        self._log_directory = Path(config.log_directory)
        self._max_log_entries = config.max_log_entries
        self._interval_secs = config.interval_secs

        # In-memory caches (updated by on_data)
        self._last_factors: FactorValues | None = None
        self._last_factors_ts: int = 0
        self._last_rebalance: RebalanceOrders | None = None
        self._last_rebalance_ts: int = 0

        # Log tail state: incremental reading + ERROR retention
        self._log_file_path: str = ""
        self._log_file_offset: int = 0
        self._error_buffer: list[dict[str, str]] = []

    def on_start(self) -> None:
        """Subscribe to data and start snapshot timer."""
        self.subscribe_data(
            DataType(FactorValues),
            client_id=ClientId(self.id.value),
        )
        self.subscribe_data(
            DataType(RebalanceOrders),
            client_id=ClientId(self.id.value),
        )

        self.clock.set_timer(
            name="snapshot_aggregator",
            interval=timedelta(seconds=self._interval_secs),
            callback=self._on_snapshot,
        )
        self.log.info(
            f"SnapshotAggregatorActor started: interval={self._interval_secs}s, "
            f"venue={self._venue}, log_dir={self._log_directory}"
        )

    def on_data(self, data: object) -> None:
        """Cache latest FactorValues and RebalanceOrders from MessageBus."""
        if isinstance(data, FactorValues):
            self._last_factors = data
            self._last_factors_ts = data.ts_event
        elif isinstance(data, RebalanceOrders):
            self._last_rebalance = data
            self._last_rebalance_ts = data.ts_event

    # -------------------------------------------------------------------------
    # Timer callback
    # -------------------------------------------------------------------------

    def _on_snapshot(self, event: object) -> None:
        """Build and write all 6 snapshots to cache.

        The 6th snapshot (``snapshot:risk``) is produced only when a
        RiskModelActor has written ``risk_model:state`` to the cache.
        Missing risk state is not an error — risk monitoring is optional.
        """
        ts_wall = self.clock.timestamp_ns()

        snapshots: list[tuple[str, dict[str, Any]]] = [
            (SNAPSHOT_VENUE_CACHE_KEY, self._build_venue_snapshot(ts_wall)),
            (SNAPSHOT_EXECUTION_CACHE_KEY, self._build_execution_snapshot(ts_wall)),
            (SNAPSHOT_FACTOR_CACHE_KEY, self._build_factor_snapshot(ts_wall)),
            (SNAPSHOT_STRATEGY_CACHE_KEY, self._build_strategy_snapshot(ts_wall)),
            (SNAPSHOT_HEALTH_CACHE_KEY, self._build_health_snapshot(ts_wall)),
        ]
        risk_snapshot = self._build_risk_snapshot(ts_wall)
        if risk_snapshot is not None:
            snapshots.append((SNAPSHOT_RISK_CACHE_KEY, risk_snapshot))

        for key, snapshot in snapshots:
            try:
                self.cache.add(key, json.dumps(snapshot).encode())
            except Exception as e:
                self.log.warning(f"Failed to write {key}: {e}")

    # -------------------------------------------------------------------------
    # Venue snapshot
    # -------------------------------------------------------------------------

    def _build_venue_snapshot(self, ts_wall: int) -> dict[str, Any]:
        account_data: dict[str, Any] = {}
        account = self.portfolio.account(self._venue)
        if account is not None:
            balance_total = account.balance_total(self._currency)
            balance_free = account.balance_free(self._currency)
            account_data = {
                "balance_total": balance_total.as_double() if balance_total else 0.0,
                "balance_free": balance_free.as_double() if balance_free else 0.0,
            }

        unrealized_pnl = 0.0
        unrealized_dict = self.portfolio.unrealized_pnls(self._venue)
        if unrealized_dict:
            money = unrealized_dict.get(self._currency)
            if money is not None:
                unrealized_pnl = money.as_double()
        account_data["unrealized_pnl"] = unrealized_pnl

        equity = compute_mtm_equity(self.portfolio, self._venue, self._currency)
        account_data["equity"] = equity if equity is not None else 0.0

        positions = []
        long_count = 0
        short_count = 0
        gross_exposure = 0.0
        net_exposure = 0.0

        for pos in self.cache.positions_open():
            exposure = self.portfolio.net_exposure(pos.instrument_id)
            notional = exposure.as_double() if exposure is not None else 0.0
            abs_notional = abs(notional)

            pos_pnl = 0.0
            pos_unrealized = self.portfolio.unrealized_pnl(pos.instrument_id)
            if pos_unrealized is not None:
                pos_pnl = pos_unrealized.as_double()

            side = "LONG" if pos.is_long else "SHORT"
            if pos.is_long:
                long_count += 1
            else:
                short_count += 1

            gross_exposure += abs_notional
            net_exposure += notional

            positions.append(
                {
                    "instrument_id": str(pos.instrument_id),
                    "side": side,
                    "quantity": str(pos.quantity),
                    "avg_px_open": float(pos.avg_px_open),
                    "unrealized_pnl": pos_pnl,
                    "notional_value": abs_notional,
                }
            )

        return {
            "ts_wall": ts_wall,
            "account": account_data,
            "positions": positions,
            "summary": {
                "total_positions": len(positions),
                "long_count": long_count,
                "short_count": short_count,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
            },
        }

    # -------------------------------------------------------------------------
    # Execution snapshot
    # -------------------------------------------------------------------------

    def _build_execution_snapshot(self, ts_wall: int) -> dict[str, Any]:
        raw = self.cache.get(EXECUTION_STATES_CACHE_KEY)
        if raw is None:
            return {"ts_wall": ts_wall, "sessions": [], "aggregate": _empty_exec_agg()}

        try:
            store = msgspec.msgpack.decode(raw, type=OrderExecutionStateStore)
        except Exception:
            return {"ts_wall": ts_wall, "sessions": [], "aggregate": _empty_exec_agg()}

        sessions = []
        active = 0
        completed = 0
        failed = 0
        total_chases = 0
        market_fallback = 0
        limit_count = 0
        market_count = 0

        for s in store.sessions:
            elapsed_ms = _elapsed_ms(s.created_ns, s.completed_ns, ts_wall)

            sessions.append(
                {
                    "primary_order_id": s.primary_order_id,
                    "instrument_id": s.instrument_id,
                    "side": s.side,
                    "state": s.state,
                    "chase_count": s.chase_count,
                    "active_order_kind": s.active_order_kind,
                    "used_market_fallback": s.used_market_fallback,
                    "elapsed_ms": elapsed_ms,
                    "limit_orders_submitted": s.limit_orders_submitted,
                    "residual_sweep_pending": s.residual_sweep_pending,
                    "transient_retry_count": s.transient_retry_count,
                }
            )

            if s.state == "COMPLETED":
                completed += 1
            elif s.state == "FAILED":
                failed += 1
            else:
                active += 1

            total_chases += s.chase_count
            if s.used_market_fallback:
                market_fallback += 1
            if s.active_order_kind == "LIMIT":
                limit_count += 1
            elif s.active_order_kind in ("MARKET", "SWEEP"):
                market_count += 1

        total_orders = limit_count + market_count
        limit_ratio = limit_count / total_orders if total_orders > 0 else 1.0

        return {
            "ts_wall": ts_wall,
            "sessions": sessions,
            "aggregate": {
                "active_sessions": active,
                "completed_sessions": completed,
                "failed_sessions": failed,
                "total_chases": total_chases,
                "market_fallback_count": market_fallback,
                "limit_ratio": limit_ratio,
            },
        }

    # -------------------------------------------------------------------------
    # Factor snapshot
    # -------------------------------------------------------------------------

    def _build_factor_snapshot(self, ts_wall: int) -> dict[str, Any]:
        if self._last_factors is None:
            return {
                "ts_snapshot": 0,
                "ts_wall": ts_wall,
                "factors": {},
                "ranks": {},
                "meta": {"factor_names": [], "instrument_count": 0},
            }

        factors = self._last_factors.factors
        ranks: dict[str, dict[str, int]] = {}
        instruments: set[str] = set()

        for fname, fvals in factors.items():
            numeric = {
                k: v for k, v in fvals.items() if isinstance(v, (int, float)) and not math.isnan(v)
            }
            instruments.update(numeric.keys())
            sorted_items = sorted(numeric.items(), key=lambda x: (-x[1], x[0]))
            ranks[fname] = {inst: rank for rank, (inst, _) in enumerate(sorted_items, 1)}

        return {
            "ts_snapshot": self._last_factors_ts,
            "ts_wall": ts_wall,
            "factors": factors,
            "ranks": ranks,
            "meta": {
                "factor_names": list(factors.keys()),
                "instrument_count": len(instruments),
            },
        }

    # -------------------------------------------------------------------------
    # Strategy snapshot
    # -------------------------------------------------------------------------

    def _build_strategy_snapshot(self, ts_wall: int) -> dict[str, Any]:
        # Read decision engine config from cache (written by DecisionEngineActor)
        decision_engine: dict[str, Any] = {}
        config_raw = self.cache.get(STRATEGY_CONFIG_CACHE_KEY)
        if config_raw is not None:
            try:
                decision_engine = json.loads(config_raw)
            except Exception:
                pass

        # Last rebalance orders
        last_rebalance: dict[str, Any] = {}
        if self._last_rebalance is not None:
            orders = self._last_rebalance.orders
            last_rebalance = {
                "ts_event": self._last_rebalance_ts,
                "total_orders": len(orders),
                "orders": orders,
            }

        # Position diff: target vs actual
        actual_long: list[str] = []
        actual_short: list[str] = []
        for pos in self.cache.positions_open():
            inst_id = str(pos.instrument_id)
            if pos.is_long:
                actual_long.append(inst_id)
            else:
                actual_short.append(inst_id)

        target_long: list[str] = []
        target_short: list[str] = []
        if self._last_rebalance is not None:
            for order in self._last_rebalance.orders:
                inst = order.get("instrument_id", "")
                side = order.get("order_side", "")
                qty = order.get("target_quote_quantity", 0)
                if qty > 0:
                    if side == "BUY":
                        target_long.append(inst)
                    elif side == "SELL":
                        target_short.append(inst)

        actual_long_set = set(actual_long)
        actual_short_set = set(actual_short)
        target_long_set = set(target_long)
        target_short_set = set(target_short)

        seconds_since_rebalance = 0.0
        if self._last_rebalance_ts > 0:
            seconds_since_rebalance = (ts_wall - self._last_rebalance_ts) / 1e9

        return {
            "ts_wall": ts_wall,
            "decision_engine": decision_engine,
            "last_rebalance": last_rebalance,
            "position_diff": {
                "target_long": sorted(target_long),
                "target_short": sorted(target_short),
                "actual_long": sorted(actual_long),
                "actual_short": sorted(actual_short),
                "missing_long": sorted(target_long_set - actual_long_set),
                "missing_short": sorted(target_short_set - actual_short_set),
                "extra_long": sorted(actual_long_set - target_long_set),
                "extra_short": sorted(actual_short_set - target_short_set),
            },
            "staleness": {
                "seconds_since_last_rebalance": seconds_since_rebalance,
            },
        }

    # -------------------------------------------------------------------------
    # Risk snapshot (factor / sector exposures, from RiskModelActor)
    # -------------------------------------------------------------------------

    def _build_risk_snapshot(self, ts_wall: int) -> dict[str, Any] | None:
        """Build portfolio-level risk exposure snapshot.

        Returns None if no RiskModelActor is active (no cache payload),
        allowing the aggregator to run without portfolio/risk stack.

        Outputs raw exposure data only — no limit-breach detection lives here.
        Grafana reads the ``snapshot:risk`` JSON from Redis and defines its
        own alert rules (thresholds, ratios, composite conditions), avoiding
        duplicate alerting channels.

        Sector aggregation uses ``RiskModelOutput.sector_map`` populated by
        FundamentalRiskModel. Statistical (PCA) snapshots contribute only
        covariance diagnostics (factor_exposures is empty because PC_i names
        are not interpretable).
        """
        payload = self.cache.get(RISK_MODEL_STATE_CACHE_KEY)
        if payload is None:
            return None
        try:
            output = deserialize_risk_output(payload)
        except Exception as exc:
            self.log.warning(f"Failed to deserialize risk snapshot: {exc}")
            return None

        # Current portfolio weights by equity share (signed: long>0, short<0)
        weights = self._collect_position_weights()
        exposure = compute_portfolio_exposure(weights, output)

        return {
            "ts_ns": ts_wall,
            "model_type": output.model_type,
            "model_timestamp_ns": output.timestamp_ns,
            "n_instruments": output.n_instruments,
            "n_factors": output.n_factors,
            "is_interpretable": output.is_interpretable,
            "is_decomposed": output.is_decomposed,
            "covariance_trace": float(output.covariance.trace()),
            # Portfolio-level statistics from compute_portfolio_exposure
            "gross": exposure["gross"],
            "net": exposure["net"],
            "long": exposure["long"],
            "short": exposure["short"],
            # Named-factor and sector exposures (Grafana defines alert rules)
            "factor_exposures": exposure["factor_exposures"],
            "sector_exposures": exposure["sector_exposures"],
        }

    def _collect_position_weights(self) -> dict[str, float]:
        """Compute current portfolio weights as fraction of equity (signed)."""
        equity = compute_mtm_equity(self.portfolio, self._venue, self._currency)
        if equity is None or equity <= 0:
            return {}
        weights: dict[str, float] = {}
        for pos in self.cache.positions_open():
            exposure = self.portfolio.net_exposure(pos.instrument_id)
            notional = exposure.as_double() if exposure is not None else 0.0
            if notional == 0.0:
                continue
            inst_id = str(pos.instrument_id)
            # Signed: positive for long, negative for short
            signed = notional if pos.is_long else -notional
            weights[inst_id] = weights.get(inst_id, 0.0) + signed / equity
        return weights

    # -------------------------------------------------------------------------
    # Health snapshot (log tail)
    # -------------------------------------------------------------------------

    def _build_health_snapshot(self, ts_wall: int) -> dict[str, Any]:
        new_entries, self._log_file_path, self._log_file_offset = _read_log_incremental(
            self._log_directory,
            self._log_file_path,
            self._log_file_offset,
        )

        # Append new ERRORs to sticky buffer; pass WARNs through
        now = time.time()
        for entry in new_entries:
            if entry["level"] == "ERROR":
                entry["_expire"] = now + _ERROR_TTL_SECS
                self._error_buffer.append(entry)

        # Evict expired ERRORs
        self._error_buffer = [e for e in self._error_buffer if e["_expire"] > now]

        # Merge: sticky errors + current-cycle warns (dedup by ts+message)
        seen: set[str] = set()
        entries: list[dict[str, str]] = []
        for e in self._error_buffer:
            key = f"{e['ts']}:{e['message']}"
            if key not in seen:
                seen.add(key)
                entries.append({k: v for k, v in e.items() if k != "_expire"})

        for e in new_entries:
            if e["level"] == "WARN":
                key = f"{e['ts']}:{e['message']}"
                if key not in seen:
                    seen.add(key)
                    entries.append(e)

        entries = entries[: self._max_log_entries]
        warn_count = sum(1 for e in entries if e["level"] == "WARN")
        error_count = sum(1 for e in entries if e["level"] == "ERROR")

        return {
            "ts_wall": ts_wall,
            "warn_count": warn_count,
            "error_count": error_count,
            "entries": entries,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_exec_agg() -> dict[str, Any]:
    return {
        "active_sessions": 0,
        "completed_sessions": 0,
        "failed_sessions": 0,
        "total_chases": 0,
        "market_fallback_count": 0,
        "limit_ratio": 1.0,
    }


def _elapsed_ms(created_ns: int, completed_ns: int, now_ns: int) -> float:
    if completed_ns > 0 and created_ns > 0:
        return (completed_ns - created_ns) / 1e6
    if created_ns > 0:
        return (now_ns - created_ns) / 1e6
    return 0.0


def _read_log_incremental(
    log_dir: Path,
    prev_path: str,
    prev_offset: int,
) -> tuple[list[dict[str, str]], str, int]:
    """Read new WARN/ERROR entries since last offset.

    Returns (entries, current_file_path, new_offset).
    On first call (prev_offset=0) bootstraps from last 256 KB.
    """
    try:
        log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime)
    except Exception:
        return [], prev_path, prev_offset
    if not log_files:
        return [], prev_path, prev_offset

    log_file = log_files[-1]
    current_path = str(log_file)

    try:
        file_size = log_file.stat().st_size
    except Exception:
        return [], current_path, prev_offset

    # Log file rotated — reset offset, bootstrap from tail
    skip_partial = False
    if current_path != prev_path:
        prev_offset = max(0, file_size - _TAIL_BYTES)
        if prev_offset > 0:
            skip_partial = True  # Mid-file seek, skip partial first line

    # Nothing new
    if file_size <= prev_offset:
        return [], current_path, prev_offset

    try:
        with open(log_file, "rb") as f:
            f.seek(prev_offset)
            if skip_partial:
                f.readline()
            read_start = f.tell()
            raw = f.read()
            new_offset = read_start + len(raw)
            tail = raw.decode("utf-8", errors="replace")
    except Exception:
        return [], current_path, prev_offset

    entries: list[dict[str, str]] = []
    for line in tail.splitlines():
        m = _LOG_RE.match(line)
        if m is None:
            continue
        ts_str, level, component, message = m.groups()
        entries.append(
            {
                "ts": ts_str[:23] + "Z",
                "level": level,
                "component": component,
                "message": message[:200],
            }
        )

    return entries, current_path, new_offset
