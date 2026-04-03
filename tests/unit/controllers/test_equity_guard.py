# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for EquityGuardController with multi-rule drawdown."""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.controllers.equity_guard import (
    DrawdownRule,
    EquityGuardController,
    EquityGuardControllerConfig,
)


HOUR_NS = 3_600_000_000_000  # 1 hour in nanoseconds


def _make_mock(
    *,
    drawdown_rules: list[DrawdownRule] | None = None,
    min_equity_ratio: float = 0.1,
    guarded_strategy_ids: list[str] | None = None,
    initial_balance: float | None = 10000.0,
    halted: bool = False,
    now_ns: int = 100 * HOUR_NS,
) -> MagicMock:
    """Create a mock that behaves like EquityGuardController."""
    rules = drawdown_rules or []
    mock = MagicMock(spec=EquityGuardController)
    mock.config = EquityGuardControllerConfig(
        interval="1h",
        venue_name="OKX",
        currency="USDT",
        min_equity_ratio=min_equity_ratio,
        guarded_strategy_ids=guarded_strategy_ids or [],
        drawdown_rules=rules,
    )
    mock._venue = MagicMock()
    mock._currency = MagicMock()
    mock._initial_balance = initial_balance
    mock._halted = halted
    mock._stopped_strategies = []
    mock._equity_history = deque()
    mock._trader = MagicMock()
    mock._trader.strategies.return_value = []
    mock.portfolio = MagicMock()
    mock.log = MagicMock()
    mock.stop_strategy = MagicMock()
    mock.clock = MagicMock()
    mock.clock.timestamp_ns.return_value = now_ns

    # Resolve rules with auto-naming (mirrors __init__ logic)
    mock._rules = tuple(
        DrawdownRule(
            name=r.name or f"rule_{i}",
            drawdown_window=r.drawdown_window,
            drawdown_ratio=r.drawdown_ratio,
            cooldown_period=r.cooldown_period,
        )
        if not r.name
        else r
        for i, r in enumerate(rules)
    )

    # Pre-compute max window (mirrors __init__ logic)
    mock._max_window_ns = EquityGuardController._compute_max_window_ns(mock)

    # Wire _peak_in_window to real implementation
    mock._peak_in_window = lambda now, w: (
        EquityGuardController._peak_in_window(mock, now, w)
    )

    return mock


def _wire_halt(mock: MagicMock) -> None:
    """Wire _halt_guarded_strategies to real implementation on mock."""
    mock._halt_guarded_strategies = lambda equity, ratio, **kw: (
        EquityGuardController._halt_guarded_strategies(mock, equity, ratio, **kw)
    )


class TestDrawdownRule:
    """Tests for DrawdownRule struct."""

    def test_default_values(self) -> None:
        rule = DrawdownRule()
        assert rule.name == ""
        assert rule.drawdown_window == ""
        assert rule.drawdown_ratio == 0.7
        assert rule.cooldown_period == ""

    def test_custom_values(self) -> None:
        rule = DrawdownRule(
            name="daily",
            drawdown_window="1d",
            drawdown_ratio=0.95,
            cooldown_period="24h",
        )
        assert rule.name == "daily"
        assert rule.drawdown_window == "1d"
        assert rule.drawdown_ratio == 0.95
        assert rule.cooldown_period == "24h"

    def test_frozen(self) -> None:
        rule = DrawdownRule()
        with pytest.raises(AttributeError):
            rule.name = "test"  # type: ignore[misc]


class TestEquityGuardControllerConfig:
    """Tests for EquityGuardControllerConfig."""

    def test_default_values(self) -> None:
        config = EquityGuardControllerConfig()
        assert config.interval == "1h"
        assert config.venue_name == "SIM"
        assert config.currency == "USD"
        assert config.min_equity_ratio == 0.1
        assert config.guarded_strategy_ids == []
        assert config.drawdown_rules == []

    def test_no_legacy_fields(self) -> None:
        config = EquityGuardControllerConfig()
        assert not hasattr(config, "drawdown_window")
        assert not hasattr(config, "drawdown_ratio")
        assert not hasattr(config, "cooldown_period")

    def test_with_drawdown_rules(self) -> None:
        rules = [
            DrawdownRule(name="monthly", drawdown_window="30d",
                         drawdown_ratio=0.8, cooldown_period="72h"),
            DrawdownRule(name="daily", drawdown_window="1d",
                         drawdown_ratio=0.95, cooldown_period="24h"),
        ]
        config = EquityGuardControllerConfig(drawdown_rules=rules)
        assert len(config.drawdown_rules) == 2
        assert config.drawdown_rules[0].name == "monthly"
        assert config.drawdown_rules[1].name == "daily"

    def test_config_is_frozen(self) -> None:
        config = EquityGuardControllerConfig()
        with pytest.raises(Exception):
            config.interval = "4h"  # type: ignore[misc]


class TestOnCheckEarlyExits:
    """Tests for _on_check early exit paths."""

    def test_skips_when_already_halted(self) -> None:
        mock = _make_mock(halted=True)
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())
        mock_equity.assert_not_called()

    def test_skips_when_initial_balance_none(self) -> None:
        mock = _make_mock(initial_balance=None)
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())
        mock_equity.assert_not_called()

    def test_skips_when_initial_balance_zero(self) -> None:
        mock = _make_mock(initial_balance=0.0)
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())
        mock_equity.assert_not_called()

    def test_skips_when_equity_none(self) -> None:
        mock = _make_mock()
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=None,
        ):
            EquityGuardController._on_check(mock, MagicMock())
        assert not mock._halted


class TestAbsoluteGuard:
    """Tests for Layer 2 absolute guard (min_equity_ratio)."""

    def test_halts_when_below_threshold(self) -> None:
        mock = _make_mock(min_equity_ratio=0.1)
        s1 = MagicMock(id="Strategy-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=500.0,  # 5% < 10%
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.stop_strategy.assert_called_once_with(s1)

    def test_no_halt_when_healthy(self) -> None:
        mock = _make_mock(min_equity_ratio=0.1)
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=5000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())
        assert not mock._halted

    def test_permanent_halt_no_cooldown_timer(self) -> None:
        mock = _make_mock(
            min_equity_ratio=0.3,
            drawdown_rules=[
                DrawdownRule(name="test", drawdown_window="72h",
                             drawdown_ratio=0.7, cooldown_period="24h"),
            ],
        )
        s1 = MagicMock(id="CSStrategy-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        # equity=2000, initial=10000 → ratio=0.2 < 0.3 → absolute guard
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=2000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_not_called()


class TestHaltStrategies:
    """Tests for _halt_guarded_strategies targeting and cooldown timer."""

    def test_stops_all_when_empty_guarded_list(self) -> None:
        mock = _make_mock(guarded_strategy_ids=[])
        s1 = MagicMock(id="S-001", is_running=True)
        s2 = MagicMock(id="S-002", is_running=True)
        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=500.0, ratio=0.05,
        )
        assert mock._halted
        assert mock.stop_strategy.call_count == 2

    def test_stops_only_guarded(self) -> None:
        mock = _make_mock(guarded_strategy_ids=["S-001"])
        s1 = MagicMock(id="S-001", is_running=True)
        s1.__str__ = lambda self: "S-001"
        s2 = MagicMock(id="S-002", is_running=True)
        s2.__str__ = lambda self: "S-002"
        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=500.0, ratio=0.05,
        )
        mock.stop_strategy.assert_called_once_with(s1)

    def test_skips_already_stopped(self) -> None:
        mock = _make_mock()
        s1 = MagicMock(id="S-001", is_running=False)
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=500.0, ratio=0.05,
        )
        mock.stop_strategy.assert_not_called()

    def test_cooldown_timer_set(self) -> None:
        mock = _make_mock(now_ns=1_000_000_000_000)
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6, cooldown_period="24h",
        )

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_called_once()
        call_kw = mock.clock.set_time_alert_ns.call_args.kwargs
        assert call_kw["name"] == "equity_guard_cooldown"
        expected = 1_000_000_000_000 + 86_400_000_000_000
        assert call_kw["alert_time_ns"] == expected

    def test_no_timer_when_permanent(self) -> None:
        mock = _make_mock()
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6, permanent=True,
        )
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_no_timer_when_cooldown_empty(self) -> None:
        mock = _make_mock()
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6, cooldown_period="",
        )
        mock.clock.set_time_alert_ns.assert_not_called()


class TestCooldownExpiry:
    """Tests for _on_cooldown_expired."""

    def test_resumes_strategies(self) -> None:
        mock = _make_mock()
        s1, s2 = MagicMock(id="S-001"), MagicMock(id="S-002")
        mock._stopped_strategies = [s1, s2]
        mock._halted = True

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=7000.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        s1.resume.assert_called_once()
        s2.resume.assert_called_once()

    def test_resets_state(self) -> None:
        mock = _make_mock()
        mock._halted = True
        mock._initial_balance = 10000.0
        mock._stopped_strategies = [MagicMock()]

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=7000.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert not mock._halted
        assert mock._initial_balance == 7000.0
        assert mock._stopped_strategies == []

    def test_clears_history(self) -> None:
        mock = _make_mock()
        mock._halted = True
        mock._equity_history = deque([
            (50 * HOUR_NS, 5000.0),
            (60 * HOUR_NS, 4000.0),
        ])
        mock._stopped_strategies = [MagicMock()]

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3500.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert len(mock._equity_history) == 1
        assert mock._equity_history[0][1] == 3500.0

    def test_stays_halted_when_equity_none(self) -> None:
        mock = _make_mock()
        mock._halted = True
        mock._initial_balance = 10000.0
        s1 = MagicMock()
        mock._stopped_strategies = [s1]

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=None,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert mock._halted
        assert mock._initial_balance == 10000.0
        s1.resume.assert_not_called()

    def test_multiple_cycles(self) -> None:
        mock = _make_mock(now_ns=1_000_000_000_000)
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]

        # Cycle 1: halt with cooldown
        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6, cooldown_period="24h",
        )
        assert mock._halted

        # Cycle 1: cooldown → resume
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=6500.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())
        assert not mock._halted
        assert mock._initial_balance == 6500.0

        # Cycle 2: halt again
        s1.is_running = True
        mock._stopped_strategies = []
        EquityGuardController._halt_guarded_strategies(
            mock, equity=4000.0, ratio=0.615, cooldown_period="24h",
        )
        assert mock._halted
        assert mock.clock.set_time_alert_ns.call_count == 2


class TestSingleDrawdownRule:
    """Tests for single drawdown rule behavior."""

    def test_triggers_halt(self) -> None:
        mock = _make_mock(
            drawdown_rules=[
                DrawdownRule(name="test", drawdown_window="72h",
                             drawdown_ratio=0.7, cooldown_period="24h"),
            ],
        )
        mock._equity_history = deque([
            (90 * HOUR_NS, 4000.0),
            (95 * HOUR_NS, 5000.0),  # peak
            (99 * HOUR_NS, 4500.0),
        ])
        s1 = MagicMock(id="CSStrategy-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        # 3400/5000 = 0.68 < 0.7 → trigger
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3400.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.stop_strategy.assert_called_once()
        mock.clock.set_time_alert_ns.assert_called_once()

    def test_no_trigger_within_threshold(self) -> None:
        mock = _make_mock(
            drawdown_rules=[
                DrawdownRule(name="test", drawdown_window="72h",
                             drawdown_ratio=0.7),
            ],
        )
        mock._equity_history = deque([(90 * HOUR_NS, 5000.0)])

        # 4000/5000 = 0.8 > 0.7 → no trigger
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=4000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted

    def test_window_prunes_old_entries(self) -> None:
        mock = _make_mock(
            drawdown_rules=[
                DrawdownRule(name="test", drawdown_window="72h",
                             drawdown_ratio=0.7),
            ],
        )
        # t=10h is outside 72h window from t=100h (cutoff=28h)
        mock._equity_history = deque([
            (10 * HOUR_NS, 9000.0),   # outside, should be pruned
            (50 * HOUR_NS, 4000.0),   # inside
            (90 * HOUR_NS, 4500.0),   # inside, peak
        ])

        # 4000/4500 = 0.889 > 0.7 → no trigger
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=4000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted
        timestamps = [ts for ts, _ in mock._equity_history]
        assert 10 * HOUR_NS not in timestamps

    def test_alltime_window_no_pruning(self) -> None:
        mock = _make_mock(
            drawdown_rules=[
                DrawdownRule(name="alltime", drawdown_window="",
                             drawdown_ratio=0.7),
            ],
        )
        mock._equity_history = deque([
            (1 * HOUR_NS, 8000.0),    # very old
            (50 * HOUR_NS, 4000.0),
            (90 * HOUR_NS, 4500.0),
        ])
        s1 = MagicMock(id="CSStrategy-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        # 5000/8000 = 0.625 < 0.7 → trigger (using all-time peak)
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=5000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        timestamps = [ts for ts, _ in mock._equity_history]
        assert 1 * HOUR_NS in timestamps  # not pruned

    def test_permanent_when_no_cooldown(self) -> None:
        mock = _make_mock(
            drawdown_rules=[
                DrawdownRule(name="no_cd", drawdown_window="72h",
                             drawdown_ratio=0.7, cooldown_period=""),
            ],
        )
        mock._equity_history = deque([(90 * HOUR_NS, 5000.0)])
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3000.0,  # 0.6 < 0.7
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_not_called()


class TestMultiDrawdownRules:
    """Tests for multi-rule drawdown behavior."""

    DAILY = DrawdownRule(
        name="daily", drawdown_window="1d",
        drawdown_ratio=0.95, cooldown_period="24h",
    )
    MONTHLY = DrawdownRule(
        name="monthly", drawdown_window="30d",
        drawdown_ratio=0.8, cooldown_period="72h",
    )

    def test_only_daily_triggers(self) -> None:
        """Daily 5% drop triggers, monthly 20% does not."""
        mock = _make_mock(drawdown_rules=[self.DAILY, self.MONTHLY])
        # Within last 1d: peak=10000. Within 30d: peak=10000.
        # equity=9400 → daily: 0.94 < 0.95 ✓, monthly: 0.94 > 0.8 ✗
        mock._equity_history = deque([
            (1 * HOUR_NS, 9500.0),     # old, within 30d
            (99 * HOUR_NS, 10000.0),   # within 1d window, peak
        ])
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=9400.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        # Should use daily cooldown (24h)
        call_kw = mock.clock.set_time_alert_ns.call_args.kwargs
        expected = 100 * HOUR_NS + 24 * HOUR_NS
        assert call_kw["alert_time_ns"] == expected

    def test_both_trigger_longest_cooldown_wins(self) -> None:
        """Both rules trigger → use longest cooldown (72h from monthly)."""
        mock = _make_mock(drawdown_rules=[self.DAILY, self.MONTHLY])
        mock._equity_history = deque([
            (1 * HOUR_NS, 10000.0),    # 30d peak
            (99 * HOUR_NS, 10000.0),   # 1d peak
        ])
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        # 7500/10000 = 0.75 → daily: < 0.95 ✓, monthly: < 0.8 ✓
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=7500.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        call_kw = mock.clock.set_time_alert_ns.call_args.kwargs
        expected = 100 * HOUR_NS + 72 * HOUR_NS  # 72h, not 24h
        assert call_kw["alert_time_ns"] == expected

    def test_triggered_rule_no_cooldown_permanent(self) -> None:
        """If any triggered rule has no cooldown → permanent halt."""
        no_cd_rule = DrawdownRule(
            name="strict", drawdown_window="1d",
            drawdown_ratio=0.95, cooldown_period="",
        )
        mock = _make_mock(drawdown_rules=[no_cd_rule, self.MONTHLY])
        mock._equity_history = deque([(99 * HOUR_NS, 10000.0)])
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=9000.0,  # 0.9 < 0.95 → strict triggers
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_no_rules_only_absolute_guard(self) -> None:
        """With empty drawdown_rules, only absolute guard is active."""
        mock = _make_mock(drawdown_rules=[], min_equity_ratio=0.5)

        # equity=6000/10000=0.6 > 0.5 → no halt
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=6000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())
        assert not mock._halted

        # equity=4000/10000=0.4 < 0.5 → absolute guard
        _wire_halt(mock)
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=4000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())
        assert mock._halted

    def test_absolute_guard_overrides_rules(self) -> None:
        """Absolute guard triggers before drawdown rules are evaluated."""
        mock = _make_mock(
            drawdown_rules=[self.DAILY, self.MONTHLY],
            min_equity_ratio=0.3,
        )
        mock._equity_history = deque([(90 * HOUR_NS, 5000.0)])
        s1 = MagicMock(id="S-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        # equity=2000, initial=10000 → abs_ratio=0.2 < 0.3 → permanent
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=2000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_window_pruning_uses_max_window(self) -> None:
        """History pruning uses the longest window among all rules."""
        # daily=1d, monthly=30d → max window = 30d = 720h
        mock = _make_mock(drawdown_rules=[self.DAILY, self.MONTHLY])
        # now=100h, max_window=720h → cutoff is negative → nothing pruned
        # But if now=800h, cutoff=80h → entries before 80h pruned
        mock.clock.timestamp_ns.return_value = 800 * HOUR_NS
        mock._max_window_ns = EquityGuardController._compute_max_window_ns(mock)

        mock._equity_history = deque([
            (10 * HOUR_NS, 9000.0),    # outside 30d from 800h (cutoff=80h)
            (100 * HOUR_NS, 8000.0),   # inside 30d
            (790 * HOUR_NS, 9500.0),   # inside both
        ])

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=9400.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        timestamps = [ts for ts, _ in mock._equity_history]
        assert 10 * HOUR_NS not in timestamps
        assert 100 * HOUR_NS in timestamps

    def test_alltime_rule_disables_pruning(self) -> None:
        """If any rule has all-time window, no pruning occurs."""
        alltime_rule = DrawdownRule(
            name="alltime", drawdown_window="",
            drawdown_ratio=0.7, cooldown_period="24h",
        )
        mock = _make_mock(drawdown_rules=[self.DAILY, alltime_rule])
        assert mock._max_window_ns is None  # no pruning

        mock._equity_history = deque([
            (1 * HOUR_NS, 9000.0),
            (90 * HOUR_NS, 8000.0),
        ])
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=8500.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        timestamps = [ts for ts, _ in mock._equity_history]
        assert 1 * HOUR_NS in timestamps  # not pruned

    def test_cooldown_resets_all_rules(self) -> None:
        """After cooldown, history is cleared and all rules start fresh."""
        mock = _make_mock(drawdown_rules=[self.DAILY, self.MONTHLY])
        mock._halted = True
        mock._equity_history = deque([
            (50 * HOUR_NS, 5000.0),
            (60 * HOUR_NS, 4000.0),
        ])
        mock._stopped_strategies = [MagicMock()]

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3500.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert not mock._halted
        assert len(mock._equity_history) == 1
        assert mock._equity_history[0][1] == 3500.0
        assert mock._initial_balance == 3500.0

    def test_auto_naming_for_unnamed_rules(self) -> None:
        """Rules without names get auto-numbered."""
        rules = [
            DrawdownRule(drawdown_window="1d", drawdown_ratio=0.95),
            DrawdownRule(drawdown_window="30d", drawdown_ratio=0.8),
        ]
        mock = _make_mock(drawdown_rules=rules)
        assert mock._rules[0].name == "rule_0"
        assert mock._rules[1].name == "rule_1"

    def test_named_rules_keep_names(self) -> None:
        """Named rules preserve their names."""
        mock = _make_mock(drawdown_rules=[self.DAILY, self.MONTHLY])
        assert mock._rules[0].name == "daily"
        assert mock._rules[1].name == "monthly"

    def test_both_layers_independent_lifecycle(self) -> None:
        """Drawdown rule halt → cooldown → resume → absolute guard works."""
        mock = _make_mock(
            drawdown_rules=[
                DrawdownRule(name="test", drawdown_window="72h",
                             drawdown_ratio=0.7, cooldown_period="24h"),
            ],
            min_equity_ratio=0.3,
        )
        mock._equity_history = deque([(90 * HOUR_NS, 5000.0)])
        s1 = MagicMock(id="CSStrategy-001", is_running=True)
        mock._trader.strategies.return_value = [s1]
        _wire_halt(mock)

        # Drawdown rule: 3400/5000 = 0.68 < 0.7 → halt with cooldown
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3400.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_called_once()

        # Cooldown → resume
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3000.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert not mock._halted
        assert mock._initial_balance == 3000.0

        # Absolute guard: 800/3000 = 0.27 < 0.3 → permanent halt
        s1.is_running = True
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=800.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        assert mock.clock.set_time_alert_ns.call_count == 1  # no new timer
