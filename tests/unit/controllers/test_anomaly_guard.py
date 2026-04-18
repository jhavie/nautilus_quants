# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for AnomalyGuardController."""

from __future__ import annotations

import json
from collections import deque
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from nautilus_quants.controllers.anomaly_guard import AnomalyGuardController, AnomalyGuardControllerConfig, AnomalyRule

HOUR_NS = 3_600_000_000_000
MIN_NS = 60_000_000_000
SEC_NS = 1_000_000_000

_EQUITY_PATH = "nautilus_quants.controllers.anomaly_guard.compute_mtm_equity"
_REALIZED_PATH = "nautilus_quants.controllers.anomaly_guard._realized_balance"


def _make_mock(
    *,
    rules: list[AnomalyRule] | None = None,
    countdown_period: str = "5m",
    cancel_on_recovery: bool = True,
    guarded_strategy_ids: list[str] | None = None,
    snapshot_staleness_secs: int = 120,
    now_ns: int = 100 * HOUR_NS,
    halted: bool = False,
    countdown_active: bool = False,
) -> MagicMock:
    """Create a MagicMock that behaves like an AnomalyGuardController instance."""
    rules = rules or []
    mock = MagicMock(spec=AnomalyGuardController)
    mock.config = AnomalyGuardControllerConfig(
        interval="30m",
        venue_name="OKX",
        currency="USDT",
        guarded_strategy_ids=guarded_strategy_ids or [],
        snapshot_staleness_secs=snapshot_staleness_secs,
        countdown_period=countdown_period,
        cancel_on_recovery=cancel_on_recovery,
        anomaly_rules=rules,
    )
    mock._venue = MagicMock()
    mock._currency = MagicMock()
    mock._halted = halted
    mock._countdown_active = countdown_active
    mock._stopped_strategies = []
    mock._trader = MagicMock()
    mock._trader.strategies.return_value = []
    mock.portfolio = MagicMock()
    mock.cache = MagicMock()
    mock.cache.get.return_value = None
    mock.log = MagicMock()
    mock.stop_strategy = MagicMock()
    mock.clock = MagicMock()
    mock.clock.timestamp_ns.return_value = now_ns

    # Mirror __init__ state setup.
    mock._rules = tuple(r if r.name else msgspec.structs.replace(r, name=f"rule_{i}") for i, r in enumerate(rules))
    mock._hits = {r.name: 0 for r in mock._rules}
    mock._equity_history = {r.name: deque() for r in mock._rules if r.kind == "equity_drawdown"}

    # Wire private methods to real implementations where we exercise them.
    mock._evaluate_rule = lambda rule, now, snap: (AnomalyGuardController._evaluate_rule(mock, rule, now, snap))
    mock._eval_equity_drawdown = lambda rule, now: (AnomalyGuardController._eval_equity_drawdown(mock, rule, now))
    mock._eval_position_count = lambda rule, snap: (AnomalyGuardController._eval_position_count(mock, rule, snap))
    mock._eval_notional_drift = lambda rule, snap: (AnomalyGuardController._eval_notional_drift(mock, rule, snap))
    mock._compute_equity_for_basis = lambda basis: (AnomalyGuardController._compute_equity_for_basis(mock, basis))
    mock._read_venue_snapshot = lambda now: (AnomalyGuardController._read_venue_snapshot(mock, now))
    mock._start_countdown = lambda now: (AnomalyGuardController._start_countdown(mock, now))
    mock._cancel_countdown = lambda: AnomalyGuardController._cancel_countdown(mock)
    mock._halt_guarded_strategies = lambda **kw: (AnomalyGuardController._halt_guarded_strategies(mock, **kw))
    return mock


def _venue_snapshot_payload(
    *,
    ts_wall: int,
    long_count: int = 0,
    short_count: int = 0,
    positions: list[dict] | None = None,
) -> bytes:
    return json.dumps(
        {
            "ts_wall": ts_wall,
            "summary": {
                "total_positions": long_count + short_count,
                "long_count": long_count,
                "short_count": short_count,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
            },
            "positions": positions or [],
            "account": {},
        }
    ).encode()


# -----------------------------------------------------------------------------
# Struct / config basics
# -----------------------------------------------------------------------------


class TestAnomalyRule:
    def test_defaults(self) -> None:
        rule = AnomalyRule()
        assert rule.name == ""
        assert rule.kind == ""
        assert rule.sustained_periods == 1
        assert rule.cooldown_period == ""
        assert rule.equity_basis == "mark_to_market"
        assert rule.drawdown_ratio == 0.7
        assert rule.tolerance == 0
        assert rule.min_violators == 1

    def test_frozen(self) -> None:
        rule = AnomalyRule()
        with pytest.raises(AttributeError):
            rule.name = "x"  # type: ignore[misc]


class TestAnomalyGuardControllerConfig:
    def test_defaults(self) -> None:
        cfg = AnomalyGuardControllerConfig()
        assert cfg.interval == "30m"
        assert cfg.countdown_period == "5m"
        assert cfg.cancel_on_recovery is True
        assert cfg.snapshot_staleness_secs == 120
        assert cfg.anomaly_rules == []

    def test_frozen(self) -> None:
        cfg = AnomalyGuardControllerConfig()
        with pytest.raises(Exception):
            cfg.interval = "1h"  # type: ignore[misc]


# -----------------------------------------------------------------------------
# equity_drawdown rule — mark_to_market vs realized_balance
# -----------------------------------------------------------------------------


class TestEquityDrawdownMarkToMarket:
    RULE = AnomalyRule(
        name="monthly",
        kind="equity_drawdown",
        equity_basis="mark_to_market",
        drawdown_window="72h",
        drawdown_ratio=0.7,
        cooldown_period="72h",
        sustained_periods=1,
    )

    def test_hits_when_ratio_below_threshold(self) -> None:
        mock = _make_mock(rules=[self.RULE], now_ns=100 * HOUR_NS)
        mock._equity_history["monthly"] = deque(
            [
                (40 * HOUR_NS, 10000.0),  # outside 72h from now=100h (cutoff=28h)
                (50 * HOUR_NS, 9000.0),
                (95 * HOUR_NS, 9500.0),
            ]
        )
        with patch(_EQUITY_PATH, return_value=6000.0):  # 6000/9500 = 0.63 < 0.7
            hit, diag = mock._eval_equity_drawdown(self.RULE, 100 * HOUR_NS)
        assert hit is True
        assert "0.63" in diag or "ratio=" in diag

    def test_miss_within_threshold(self) -> None:
        mock = _make_mock(rules=[self.RULE], now_ns=100 * HOUR_NS)
        mock._equity_history["monthly"] = deque([(95 * HOUR_NS, 9500.0)])
        with patch(_EQUITY_PATH, return_value=7000.0):  # 7000/9500 ≈ 0.737 > 0.7
            hit, _ = mock._eval_equity_drawdown(self.RULE, 100 * HOUR_NS)
        assert hit is False

    def test_window_prunes_old_entries(self) -> None:
        mock = _make_mock(rules=[self.RULE], now_ns=100 * HOUR_NS)
        mock._equity_history["monthly"] = deque(
            [
                (10 * HOUR_NS, 20000.0),  # very old — must be pruned
                (50 * HOUR_NS, 8000.0),
                (95 * HOUR_NS, 9000.0),
            ]
        )
        with patch(_EQUITY_PATH, return_value=7500.0):
            mock._eval_equity_drawdown(self.RULE, 100 * HOUR_NS)
        timestamps = [ts for ts, _ in mock._equity_history["monthly"]]
        assert 10 * HOUR_NS not in timestamps

    def test_uses_mtm_equity(self) -> None:
        mock = _make_mock(rules=[self.RULE], now_ns=100 * HOUR_NS)
        with patch(_EQUITY_PATH, return_value=9999.0) as mtm, patch(_REALIZED_PATH) as realized:
            mock._eval_equity_drawdown(self.RULE, 100 * HOUR_NS)
        mtm.assert_called_once()
        realized.assert_not_called()


class TestEquityDrawdownRealizedBalance:
    RULE = AnomalyRule(
        name="daily",
        kind="equity_drawdown",
        equity_basis="realized_balance",
        drawdown_window="1d",
        drawdown_ratio=0.95,
        cooldown_period="24h",
        sustained_periods=1,
    )

    def test_uses_realized_balance_not_mtm(self) -> None:
        mock = _make_mock(rules=[self.RULE], now_ns=100 * HOUR_NS)
        with patch(_EQUITY_PATH) as mtm, patch(_REALIZED_PATH, return_value=10000.0) as realized:
            mock._eval_equity_drawdown(self.RULE, 100 * HOUR_NS)
        realized.assert_called_once()
        mtm.assert_not_called()

    def test_hits_when_below_threshold(self) -> None:
        mock = _make_mock(rules=[self.RULE], now_ns=100 * HOUR_NS)
        mock._equity_history["daily"] = deque([(99 * HOUR_NS, 10000.0)])
        with patch(_REALIZED_PATH, return_value=9400.0):  # 0.94 < 0.95
            hit, _ = mock._eval_equity_drawdown(self.RULE, 100 * HOUR_NS)
        assert hit is True

    def test_returns_miss_when_balance_unavailable(self) -> None:
        mock = _make_mock(rules=[self.RULE], now_ns=100 * HOUR_NS)
        with patch(_REALIZED_PATH, return_value=None):
            hit, _ = mock._eval_equity_drawdown(self.RULE, 100 * HOUR_NS)
        assert hit is False


# -----------------------------------------------------------------------------
# position_count rule
# -----------------------------------------------------------------------------


class TestPositionCountRule:
    RULE = AnomalyRule(
        name="pc",
        kind="position_count",
        expected_long=20,
        expected_short=20,
        tolerance=2,
        sustained_periods=3,
        cooldown_period="2h",
    )

    def test_within_tolerance_is_miss(self) -> None:
        mock = _make_mock(rules=[self.RULE])
        snap = {"summary": {"long_count": 21, "short_count": 19}}
        hit, _ = mock._eval_position_count(self.RULE, snap)
        assert hit is False

    def test_long_deviation_beyond_tolerance_hits(self) -> None:
        mock = _make_mock(rules=[self.RULE])
        snap = {"summary": {"long_count": 23, "short_count": 20}}
        hit, diag = mock._eval_position_count(self.RULE, snap)
        assert hit is True
        assert "long=23/20" in diag

    def test_short_deviation_beyond_tolerance_hits(self) -> None:
        mock = _make_mock(rules=[self.RULE])
        snap = {"summary": {"long_count": 20, "short_count": 16}}
        hit, diag = mock._eval_position_count(self.RULE, snap)
        assert hit is True
        assert "short=16/20" in diag

    def test_user_scenario_20_20_vs_39_open(self) -> None:
        """Planned 20 long + 20 short but only 19 long + 20 short open."""
        mock = _make_mock(
            rules=[
                AnomalyRule(
                    name="pc",
                    kind="position_count",
                    expected_long=20,
                    expected_short=20,
                    tolerance=0,
                    sustained_periods=1,
                    cooldown_period="2h",
                )
            ]
        )
        snap = {"summary": {"long_count": 19, "short_count": 20}}
        hit, _ = mock._eval_position_count(mock._rules[0], snap)
        assert hit is True

    def test_missing_snapshot_returns_miss(self) -> None:
        mock = _make_mock(rules=[self.RULE])
        hit, _ = mock._eval_position_count(self.RULE, None)
        assert hit is False


# -----------------------------------------------------------------------------
# notional_drift rule
# -----------------------------------------------------------------------------


class TestNotionalDriftRule:
    RULE = AnomalyRule(
        name="nd",
        kind="notional_drift",
        target_notional_usd=200.0,
        max_drift_ratio=1.5,
        min_drift_ratio=0.5,
        min_violators=3,
        sustained_periods=2,
        cooldown_period="4h",
    )

    def test_all_within_band_miss(self) -> None:
        mock = _make_mock(rules=[self.RULE])
        snap = {
            "positions": [
                {"instrument_id": "A", "notional_value": 210.0},
                {"instrument_id": "B", "notional_value": 190.0},
                {"instrument_id": "C", "notional_value": 250.0},
            ]
        }
        hit, _ = mock._eval_notional_drift(self.RULE, snap)
        assert hit is False

    def test_user_scenario_planned_200_actual_400(self) -> None:
        """Planned 200 per pos but 3+ positions at 400 USDT."""
        mock = _make_mock(rules=[self.RULE])
        snap = {
            "positions": [
                {"instrument_id": "A", "notional_value": 400.0},
                {"instrument_id": "B", "notional_value": 410.0},
                {"instrument_id": "C", "notional_value": 395.0},
                {"instrument_id": "D", "notional_value": 200.0},
            ]
        }
        hit, diag = mock._eval_notional_drift(self.RULE, snap)
        assert hit is True
        assert "violators=3" in diag

    def test_below_min_violators_miss(self) -> None:
        mock = _make_mock(rules=[self.RULE])
        snap = {
            "positions": [
                {"instrument_id": "A", "notional_value": 400.0},  # violator
                {"instrument_id": "B", "notional_value": 200.0},
            ]
        }
        hit, _ = mock._eval_notional_drift(self.RULE, snap)
        assert hit is False

    def test_zero_target_disables_rule(self) -> None:
        mock = _make_mock(
            rules=[
                AnomalyRule(
                    name="nd",
                    kind="notional_drift",
                    target_notional_usd=0.0,
                    max_drift_ratio=1.5,
                    min_violators=1,
                )
            ]
        )
        snap = {"positions": [{"instrument_id": "A", "notional_value": 99999.0}]}
        hit, _ = mock._eval_notional_drift(mock._rules[0], snap)
        assert hit is False

    def test_missing_snapshot_returns_miss(self) -> None:
        mock = _make_mock(rules=[self.RULE])
        hit, _ = mock._eval_notional_drift(self.RULE, None)
        assert hit is False


# -----------------------------------------------------------------------------
# Snapshot staleness
# -----------------------------------------------------------------------------


class TestSnapshotStaleness:
    def test_skips_stale_snapshot(self) -> None:
        rule = AnomalyRule(
            name="pc",
            kind="position_count",
            expected_long=20,
            expected_short=20,
            tolerance=0,
        )
        mock = _make_mock(rules=[rule], snapshot_staleness_secs=60, now_ns=100 * HOUR_NS)
        mock.cache.get.return_value = _venue_snapshot_payload(
            ts_wall=100 * HOUR_NS - 5 * MIN_NS,  # 5 min stale (>60s)
            long_count=0,
            short_count=0,
        )
        snap = mock._read_venue_snapshot(100 * HOUR_NS)
        assert snap is None

    def test_returns_fresh_snapshot(self) -> None:
        rule = AnomalyRule(
            name="pc",
            kind="position_count",
            expected_long=20,
            expected_short=20,
        )
        mock = _make_mock(rules=[rule], snapshot_staleness_secs=120, now_ns=100 * HOUR_NS)
        mock.cache.get.return_value = _venue_snapshot_payload(
            ts_wall=100 * HOUR_NS - 30 * SEC_NS,
            long_count=20,
            short_count=20,
        )
        snap = mock._read_venue_snapshot(100 * HOUR_NS)
        assert snap is not None
        assert snap["summary"]["long_count"] == 20

    def test_missing_cache_key(self) -> None:
        rule = AnomalyRule(
            name="pc",
            kind="position_count",
            expected_long=1,
            expected_short=1,
        )
        mock = _make_mock(rules=[rule])
        mock.cache.get.return_value = None
        snap = mock._read_venue_snapshot(100 * HOUR_NS)
        assert snap is None

    def test_no_snapshot_rules_skips_cache_read(self) -> None:
        rule = AnomalyRule(
            name="ed",
            kind="equity_drawdown",
            drawdown_window="72h",
            drawdown_ratio=0.7,
        )
        mock = _make_mock(rules=[rule])
        snap = mock._read_venue_snapshot(100 * HOUR_NS)
        assert snap is None
        mock.cache.get.assert_not_called()


# -----------------------------------------------------------------------------
# Sustained-periods debounce + countdown lifecycle
# -----------------------------------------------------------------------------


class TestCountdownLifecycle:
    def _drawdown_rule(self, *, sustained: int = 3, cooldown: str = "2h") -> AnomalyRule:
        return AnomalyRule(
            name="dd",
            kind="equity_drawdown",
            equity_basis="mark_to_market",
            drawdown_window="72h",
            drawdown_ratio=0.7,
            sustained_periods=sustained,
            cooldown_period=cooldown,
        )

    def test_single_hit_does_not_start_countdown(self) -> None:
        rule = self._drawdown_rule(sustained=3)
        mock = _make_mock(rules=[rule])
        mock._equity_history["dd"] = deque([(99 * HOUR_NS, 10000.0)])
        with patch(_EQUITY_PATH, return_value=6000.0):  # 0.6 < 0.7 — HIT
            AnomalyGuardController._on_check(mock, MagicMock())
        assert mock._hits["dd"] == 1
        assert not mock._countdown_active
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_sustained_hits_trigger_countdown(self) -> None:
        rule = self._drawdown_rule(sustained=2)
        mock = _make_mock(rules=[rule])
        mock._equity_history["dd"] = deque([(99 * HOUR_NS, 10000.0)])
        with patch(_EQUITY_PATH, return_value=6000.0):
            AnomalyGuardController._on_check(mock, MagicMock())
            AnomalyGuardController._on_check(mock, MagicMock())
        assert mock._countdown_active
        assert mock.clock.set_time_alert_ns.call_count == 1
        call = mock.clock.set_time_alert_ns.call_args.kwargs
        assert call["name"] == "anomaly_guard_countdown"

    def test_miss_resets_hits(self) -> None:
        rule = self._drawdown_rule(sustained=3)
        mock = _make_mock(rules=[rule])
        mock._equity_history["dd"] = deque([(99 * HOUR_NS, 10000.0)])
        with patch(_EQUITY_PATH, return_value=6000.0):
            AnomalyGuardController._on_check(mock, MagicMock())
        assert mock._hits["dd"] == 1
        with patch(_EQUITY_PATH, return_value=9000.0):  # 0.9 > 0.7
            AnomalyGuardController._on_check(mock, MagicMock())
        assert mock._hits["dd"] == 0

    def test_cancel_on_recovery(self) -> None:
        rule = self._drawdown_rule(sustained=1, cooldown="2h")
        mock = _make_mock(rules=[rule], cancel_on_recovery=True)
        mock._equity_history["dd"] = deque([(99 * HOUR_NS, 10000.0)])
        with patch(_EQUITY_PATH, return_value=6000.0):
            AnomalyGuardController._on_check(mock, MagicMock())
        assert mock._countdown_active
        with patch(_EQUITY_PATH, return_value=9500.0):  # recovery
            AnomalyGuardController._on_check(mock, MagicMock())
        assert not mock._countdown_active
        mock.clock.cancel_time_alert.assert_called_once_with("anomaly_guard_countdown")

    def test_cancel_disabled_keeps_countdown(self) -> None:
        rule = self._drawdown_rule(sustained=1)
        mock = _make_mock(rules=[rule], cancel_on_recovery=False)
        mock._equity_history["dd"] = deque([(99 * HOUR_NS, 10000.0)])
        with patch(_EQUITY_PATH, return_value=6000.0):
            AnomalyGuardController._on_check(mock, MagicMock())
        with patch(_EQUITY_PATH, return_value=9500.0):
            AnomalyGuardController._on_check(mock, MagicMock())
        assert mock._countdown_active
        mock.clock.cancel_time_alert.assert_not_called()


# -----------------------------------------------------------------------------
# Countdown expiry → halt; cooldown → resume
# -----------------------------------------------------------------------------


class TestCountdownExpiry:
    def test_halt_with_longest_cooldown(self) -> None:
        daily = AnomalyRule(
            name="daily",
            kind="equity_drawdown",
            drawdown_window="1d",
            drawdown_ratio=0.95,
            sustained_periods=1,
            cooldown_period="24h",
        )
        monthly = AnomalyRule(
            name="monthly",
            kind="equity_drawdown",
            drawdown_window="30d",
            drawdown_ratio=0.8,
            sustained_periods=1,
            cooldown_period="72h",
        )
        mock = _make_mock(rules=[daily, monthly], countdown_active=True)
        mock._hits = {"daily": 1, "monthly": 1}
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]

        AnomalyGuardController._on_countdown_expired(mock, MagicMock())

        assert mock._halted
        assert not mock._countdown_active
        mock.stop_strategy.assert_called_once_with(s1)
        # Cooldown timer uses monthly's 72h
        assert mock.clock.set_time_alert_ns.called
        call = mock.clock.set_time_alert_ns.call_args.kwargs
        assert call["name"] == "anomaly_guard_cooldown"

    def test_permanent_halt_when_any_rule_has_no_cooldown(self) -> None:
        strict = AnomalyRule(
            name="strict",
            kind="position_count",
            expected_long=20,
            expected_short=20,
            tolerance=0,
            sustained_periods=1,
            cooldown_period="",
        )
        mock = _make_mock(rules=[strict], countdown_active=True)
        mock._hits = {"strict": 1}
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]

        AnomalyGuardController._on_countdown_expired(mock, MagicMock())

        assert mock._halted
        mock.stop_strategy.assert_called_once()
        # No cooldown timer scheduled for permanent halt.
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_no_halt_when_already_halted(self) -> None:
        rule = AnomalyRule(
            name="pc",
            kind="position_count",
            expected_long=0,
            expected_short=0,
            sustained_periods=1,
            cooldown_period="2h",
        )
        mock = _make_mock(rules=[rule], countdown_active=True, halted=True)
        mock._hits = {"pc": 1}
        AnomalyGuardController._on_countdown_expired(mock, MagicMock())
        mock.stop_strategy.assert_not_called()

    def test_safety_net_no_eligible_rules(self) -> None:
        """Race: countdown timer fires but hits have already reset."""
        rule = AnomalyRule(
            name="pc",
            kind="position_count",
            expected_long=20,
            expected_short=20,
            sustained_periods=1,
            cooldown_period="2h",
        )
        mock = _make_mock(rules=[rule], countdown_active=True)
        mock._hits = {"pc": 0}
        AnomalyGuardController._on_countdown_expired(mock, MagicMock())
        assert not mock._halted
        mock.stop_strategy.assert_not_called()


class TestHaltAndCooldown:
    def test_halt_stops_only_guarded(self) -> None:
        mock = _make_mock(
            rules=[AnomalyRule(name="r", kind="equity_drawdown")],
            guarded_strategy_ids=["S-001"],
        )
        s1 = MagicMock(is_running=True)
        s1.id = "S-001"
        s1.__str__ = lambda self: "S-001"
        s2 = MagicMock(is_running=True)
        s2.id = "S-002"
        s2.__str__ = lambda self: "S-002"
        mock._trader.strategies.return_value = [s1, s2]
        AnomalyGuardController._halt_guarded_strategies(mock, cooldown_period="")
        mock.stop_strategy.assert_called_once_with(s1)

    def test_halt_stops_all_when_guarded_empty(self) -> None:
        mock = _make_mock(rules=[AnomalyRule(name="r", kind="equity_drawdown")])
        s1 = MagicMock(is_running=True)
        s1.id = "S-001"
        s1.__str__ = lambda self: "S-001"
        s2 = MagicMock(is_running=True)
        s2.id = "S-002"
        s2.__str__ = lambda self: "S-002"
        mock._trader.strategies.return_value = [s1, s2]
        AnomalyGuardController._halt_guarded_strategies(mock, cooldown_period="1h")
        assert mock.stop_strategy.call_count == 2

    def test_cooldown_expiry_resumes_and_resets(self) -> None:
        rule = AnomalyRule(
            name="dd",
            kind="equity_drawdown",
            drawdown_window="72h",
            drawdown_ratio=0.7,
            sustained_periods=2,
            cooldown_period="24h",
        )
        mock = _make_mock(rules=[rule], halted=True)
        mock._hits = {"dd": 5}
        mock._equity_history["dd"] = deque([(0, 9000.0), (HOUR_NS, 8000.0)])
        s1 = MagicMock(id="S-001")
        mock._stopped_strategies = [s1]

        AnomalyGuardController._on_cooldown_expired(mock, MagicMock())

        s1.resume.assert_called_once()
        assert not mock._halted
        assert mock._hits == {"dd": 0}
        assert list(mock._equity_history["dd"]) == []
        assert mock._stopped_strategies == []


# -----------------------------------------------------------------------------
# Multi-rule interaction + halt gating
# -----------------------------------------------------------------------------


class TestMultiRuleInteraction:
    def test_halted_short_circuits_check(self) -> None:
        rule = AnomalyRule(
            name="pc",
            kind="position_count",
            expected_long=20,
            expected_short=20,
            sustained_periods=1,
            cooldown_period="2h",
        )
        mock = _make_mock(rules=[rule], halted=True)
        AnomalyGuardController._on_check(mock, MagicMock())
        mock.cache.get.assert_not_called()

    def test_no_rules_short_circuits_check(self) -> None:
        mock = _make_mock(rules=[])
        AnomalyGuardController._on_check(mock, MagicMock())
        mock.cache.get.assert_not_called()

    def test_one_rule_eligible_triggers_countdown_among_many(self) -> None:
        dd_sustained_2 = AnomalyRule(
            name="dd",
            kind="equity_drawdown",
            drawdown_window="72h",
            drawdown_ratio=0.7,
            sustained_periods=2,
            cooldown_period="24h",
        )
        pc_sustained_1 = AnomalyRule(
            name="pc",
            kind="position_count",
            expected_long=20,
            expected_short=20,
            tolerance=0,
            sustained_periods=1,
            cooldown_period="2h",
        )
        mock = _make_mock(rules=[dd_sustained_2, pc_sustained_1])
        mock._equity_history["dd"] = deque([(99 * HOUR_NS, 10000.0)])
        mock.cache.get.return_value = _venue_snapshot_payload(
            ts_wall=100 * HOUR_NS,
            long_count=22,
            short_count=20,
        )
        with patch(_EQUITY_PATH, return_value=9500.0):  # 0.95 > 0.7 — dd miss
            AnomalyGuardController._on_check(mock, MagicMock())
        # dd miss, pc hit (long dev=2>0)
        assert mock._hits["dd"] == 0
        assert mock._hits["pc"] == 1
        assert mock._countdown_active  # pc eligible with sustained=1
