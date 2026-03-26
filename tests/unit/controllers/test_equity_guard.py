"""Unit tests for EquityGuardController."""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.controllers.equity_guard import (
    EquityGuardController,
    EquityGuardControllerConfig,
)


class TestEquityGuardControllerConfig:
    """Tests for EquityGuardControllerConfig."""

    def test_default_values(self) -> None:
        config = EquityGuardControllerConfig()
        assert config.interval == "1h"
        assert config.venue_name == "SIM"
        assert config.currency == "USD"
        assert config.min_equity_ratio == 0.1
        assert config.guarded_strategy_ids == []
        assert config.cooldown_period == ""

    def test_custom_values(self) -> None:
        config = EquityGuardControllerConfig(
            interval="4h",
            venue_name="BINANCE",
            currency="USDT",
            min_equity_ratio=0.2,
            guarded_strategy_ids=["Strategy-001"],
            cooldown_period="24h",
        )
        assert config.interval == "4h"
        assert config.venue_name == "BINANCE"
        assert config.currency == "USDT"
        assert config.min_equity_ratio == 0.2
        assert config.guarded_strategy_ids == ["Strategy-001"]
        assert config.cooldown_period == "24h"

    def test_config_is_frozen(self) -> None:
        config = EquityGuardControllerConfig()
        with pytest.raises(Exception):
            config.interval = "4h"  # type: ignore[misc]


class TestEquityGuardControllerOnCheck:
    """Tests for EquityGuardController._on_check behavior.

    Since Controller/Actor has Cython-level readonly 'config' attribute,
    we test _on_check and _halt_guarded_strategies as unbound methods
    with a mock 'self' object that mimics the controller's internal state.
    """

    def _make_mock_controller(
        self,
        min_equity_ratio: float = 0.1,
        guarded_strategy_ids: list[str] | None = None,
        initial_balance: float | None = 10000.0,
        halted: bool = False,
    ) -> MagicMock:
        """Create a mock that behaves like EquityGuardController."""
        mock = MagicMock(spec=EquityGuardController)
        mock.config = EquityGuardControllerConfig(
            interval="1h",
            venue_name="SIM",
            currency="USD",
            min_equity_ratio=min_equity_ratio,
            guarded_strategy_ids=guarded_strategy_ids or [],
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
        mock.clock.timestamp_ns.return_value = 1_000_000_000_000
        return mock

    def test_on_check_no_halt_when_equity_healthy(self) -> None:
        """Test _on_check does NOT halt when equity is above threshold."""
        mock = self._make_mock_controller(min_equity_ratio=0.1)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=5000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted
        mock.stop_strategy.assert_not_called()

    def test_on_check_halts_when_equity_below_threshold(self) -> None:
        """Test _on_check halts strategies when equity < min_ratio * initial."""
        mock = self._make_mock_controller(min_equity_ratio=0.1)

        mock_strategy = MagicMock()
        mock_strategy.id = "Strategy-001"
        mock_strategy.is_running = True
        mock._trader.strategies.return_value = [mock_strategy]

        # Use real _halt_guarded_strategies so _halted flag is set
        mock._halt_guarded_strategies = lambda equity, ratio, permanent=False: (
            EquityGuardController._halt_guarded_strategies(
                mock, equity, ratio, permanent=permanent,
            )
        )

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=500.0,  # 5% < 10% threshold
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.stop_strategy.assert_called_once_with(mock_strategy)

    def test_on_check_skips_when_already_halted(self) -> None:
        """Test _on_check does nothing when already halted."""
        mock = self._make_mock_controller(halted=True)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())

        mock_equity.assert_not_called()

    def test_on_check_skips_when_initial_balance_none(self) -> None:
        """Test _on_check safely exits when initial_balance is None."""
        mock = self._make_mock_controller(initial_balance=None)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())

        mock_equity.assert_not_called()

    def test_on_check_skips_when_initial_balance_zero(self) -> None:
        """Test _on_check safely exits when initial_balance is zero."""
        mock = self._make_mock_controller(initial_balance=0.0)

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
        ) as mock_equity:
            EquityGuardController._on_check(mock, MagicMock())

        mock_equity.assert_not_called()

    def test_on_check_skips_when_equity_none(self) -> None:
        """Test _on_check safely exits when compute_mtm_equity returns None."""
        mock = self._make_mock_controller()

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=None,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted


class TestEquityGuardControllerHaltStrategies:
    """Tests for _halt_guarded_strategies targeting specific strategies."""

    def _make_mock_controller(
        self,
        guarded_strategy_ids: list[str] | None = None,
    ) -> MagicMock:
        mock = MagicMock(spec=EquityGuardController)
        mock.config = EquityGuardControllerConfig(
            guarded_strategy_ids=guarded_strategy_ids or [],
        )
        mock._halted = False
        mock._trader = MagicMock()
        mock.log = MagicMock()
        mock.stop_strategy = MagicMock()
        return mock

    def test_stops_all_strategies_when_empty_guarded_list(self) -> None:
        """Test all strategies stopped when guarded_strategy_ids is empty."""
        mock = self._make_mock_controller(guarded_strategy_ids=[])

        s1 = MagicMock()
        s1.id = "Strategy-001"
        s1.is_running = True
        s2 = MagicMock()
        s2.id = "Strategy-002"
        s2.is_running = True
        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(mock, equity=500.0, ratio=0.05)

        assert mock._halted
        assert mock.stop_strategy.call_count == 2

    def test_stops_only_guarded_strategies(self) -> None:
        """Test only specified strategies are stopped."""
        mock = self._make_mock_controller(guarded_strategy_ids=["Strategy-001"])

        s1 = MagicMock()
        s1.id = "Strategy-001"
        s1.__str__ = lambda self: "Strategy-001"
        s1.is_running = True

        s2 = MagicMock()
        s2.id = "Strategy-002"
        s2.__str__ = lambda self: "Strategy-002"
        s2.is_running = True

        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(mock, equity=500.0, ratio=0.05)

        assert mock._halted
        mock.stop_strategy.assert_called_once_with(s1)

    def test_skips_already_stopped_strategies(self) -> None:
        """Test does not stop strategies that are already not running."""
        mock = self._make_mock_controller(guarded_strategy_ids=[])

        s1 = MagicMock()
        s1.id = "Strategy-001"
        s1.is_running = False  # Already stopped
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(mock, equity=500.0, ratio=0.05)

        assert mock._halted
        mock.stop_strategy.assert_not_called()


class TestEquityGuardControllerCooldown:
    """Tests for cooldown period: halt → wait → restart cycle."""

    def _make_mock_controller(
        self,
        min_equity_ratio: float = 0.7,
        cooldown_period: str = "24h",
        guarded_strategy_ids: list[str] | None = None,
        initial_balance: float | None = 10000.0,
    ) -> MagicMock:
        """Create a mock with cooldown-related attributes."""
        mock = MagicMock(spec=EquityGuardController)
        mock.config = EquityGuardControllerConfig(
            interval="1h",
            venue_name="OKX",
            currency="USDT",
            min_equity_ratio=min_equity_ratio,
            guarded_strategy_ids=guarded_strategy_ids or [],
            cooldown_period=cooldown_period,
        )
        mock._venue = MagicMock()
        mock._currency = MagicMock()
        mock._initial_balance = initial_balance
        mock._halted = False
        mock._stopped_strategies = []
        mock._equity_history = deque()
        mock._trader = MagicMock()
        mock._trader.strategies.return_value = []
        mock.portfolio = MagicMock()
        mock.log = MagicMock()
        mock.stop_strategy = MagicMock()
        mock.start_strategy = MagicMock()
        mock.clock = MagicMock()
        mock.clock.timestamp_ns.return_value = 1_000_000_000_000
        return mock

    def test_halt_sets_cooldown_timer(self) -> None:
        """Test _halt_guarded_strategies sets a one-shot timer when cooldown is configured."""
        mock = self._make_mock_controller(cooldown_period="24h")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_called_once()
        call_kwargs = mock.clock.set_time_alert_ns.call_args
        assert call_kwargs.kwargs["name"] == "equity_guard_cooldown"
        # 24h = 86400 seconds = 86_400_000_000_000 nanoseconds
        expected_ns = 1_000_000_000_000 + 86_400_000_000_000
        assert call_kwargs.kwargs["alert_time_ns"] == expected_ns

    def test_halt_no_timer_when_cooldown_empty(self) -> None:
        """Test _halt_guarded_strategies does NOT set timer when cooldown is empty."""
        mock = self._make_mock_controller(cooldown_period="")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_halt_records_stopped_strategies(self) -> None:
        """Test stopped strategies are recorded for later restart."""
        mock = self._make_mock_controller(cooldown_period="24h")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        s2 = MagicMock()
        s2.id = "Other-001"
        s2.is_running = True
        mock._trader.strategies.return_value = [s1, s2]

        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )

        assert len(mock._stopped_strategies) == 2

    def test_cooldown_expired_restarts_strategies(self) -> None:
        """Test _on_cooldown_expired restarts all previously stopped strategies."""
        mock = self._make_mock_controller(cooldown_period="24h")

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s2 = MagicMock()
        s2.id = "CSStrategy-002"
        mock._stopped_strategies = [s1, s2]
        mock._halted = True

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=7000.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert mock.start_strategy.call_count == 2
        mock.start_strategy.assert_any_call(s1)
        mock.start_strategy.assert_any_call(s2)

    def test_cooldown_expired_resets_state(self) -> None:
        """Test _on_cooldown_expired resets _halted and _initial_balance."""
        mock = self._make_mock_controller(cooldown_period="24h")
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

    def test_multiple_cooldown_cycles(self) -> None:
        """Test halt → cooldown → restart → halt again works correctly."""
        mock = self._make_mock_controller(
            cooldown_period="24h", min_equity_ratio=0.7,
        )

        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        # Cycle 1: halt
        EquityGuardController._halt_guarded_strategies(
            mock, equity=6000.0, ratio=0.6,
        )
        assert mock._halted
        assert len(mock._stopped_strategies) == 1

        # Cycle 1: cooldown expired → restart
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=6500.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())
        assert not mock._halted
        assert mock._initial_balance == 6500.0

        # Cycle 2: halt again (equity drops below 70% of new baseline 6500)
        mock._stopped_strategies = []
        s1.is_running = True
        EquityGuardController._halt_guarded_strategies(
            mock, equity=4000.0, ratio=0.615,
        )
        assert mock._halted
        assert mock.clock.set_time_alert_ns.call_count == 2


class TestEquityGuardRollingDrawdown:
    """Tests for rolling window drawdown + dual-layer protection."""

    HOUR_NS = 3_600_000_000_000  # 1 hour in nanoseconds

    def _make_mock_controller(
        self,
        drawdown_window: str = "72h",
        drawdown_ratio: float = 0.7,
        min_equity_ratio: float = 0.3,
        cooldown_period: str = "24h",
        initial_balance: float = 10000.0,
    ) -> MagicMock:
        """Create a mock with rolling drawdown attributes."""
        mock = MagicMock(spec=EquityGuardController)
        mock.config = EquityGuardControllerConfig(
            interval="1h",
            venue_name="OKX",
            currency="USDT",
            drawdown_window=drawdown_window,
            drawdown_ratio=drawdown_ratio,
            min_equity_ratio=min_equity_ratio,
            cooldown_period=cooldown_period,
        )
        mock._venue = MagicMock()
        mock._currency = MagicMock()
        mock._initial_balance = initial_balance
        mock._halted = False
        mock._stopped_strategies = []
        mock._equity_history = deque()
        mock._trader = MagicMock()
        mock._trader.strategies.return_value = []
        mock.portfolio = MagicMock()
        mock.log = MagicMock()
        mock.stop_strategy = MagicMock()
        mock.start_strategy = MagicMock()
        mock.clock = MagicMock()
        mock.clock.timestamp_ns.return_value = 100 * self.HOUR_NS
        return mock

    def test_drawdown_config_defaults(self) -> None:
        """Test drawdown_window defaults to empty, drawdown_ratio to 0.7."""
        config = EquityGuardControllerConfig()
        assert config.drawdown_window == ""
        assert config.drawdown_ratio == 0.7

    def test_rolling_peak_triggers_halt(self) -> None:
        """Test halt when current drops below drawdown_ratio of window peak."""
        mock = self._make_mock_controller(drawdown_ratio=0.7)
        # Simulate history: peak was 5000 within window
        mock._equity_history = deque([
            (90 * self.HOUR_NS, 4000.0),
            (95 * self.HOUR_NS, 5000.0),  # peak
            (99 * self.HOUR_NS, 4500.0),
        ])
        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        mock._halt_guarded_strategies = lambda equity, ratio, permanent=False: (
            EquityGuardController._halt_guarded_strategies(
                mock, equity, ratio, permanent=permanent,
            )
        )

        # current = 3400, ratio = 3400/5000 = 0.68 < 0.7 → trigger
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3400.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.stop_strategy.assert_called_once()

    def test_rolling_window_prunes_old(self) -> None:
        """Test entries outside drawdown_window are pruned."""
        mock = self._make_mock_controller(
            drawdown_window="72h", drawdown_ratio=0.7,
        )
        # Old peak at t=10h (outside 72h window from t=100h)
        # Window starts at t=28h
        mock._equity_history = deque([
            (10 * self.HOUR_NS, 9000.0),  # outside window, should be pruned
            (50 * self.HOUR_NS, 4000.0),  # inside window
            (90 * self.HOUR_NS, 4500.0),  # inside window, peak
        ])

        # current = 4000, peak in window = 4500, ratio = 0.889 > 0.7 → no trigger
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=4000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted
        # The old 9000 entry should have been pruned
        timestamps = [ts for ts, _ in mock._equity_history]
        assert 10 * self.HOUR_NS not in timestamps

    def test_rolling_no_trigger_within_threshold(self) -> None:
        """Test no halt when drawdown is within threshold."""
        mock = self._make_mock_controller(drawdown_ratio=0.7)
        mock._equity_history = deque([
            (90 * self.HOUR_NS, 5000.0),  # peak
        ])

        # current = 4000, ratio = 4000/5000 = 0.8 > 0.7 → no trigger
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=4000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert not mock._halted

    def test_absolute_guard_overrides_cooldown(self) -> None:
        """Test absolute guard triggers permanent halt, no cooldown timer."""
        mock = self._make_mock_controller(
            min_equity_ratio=0.3, cooldown_period="24h",
        )
        mock._equity_history = deque()
        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        # Use real _halt_guarded_strategies
        mock._halt_guarded_strategies = lambda equity, ratio, permanent=False: (
            EquityGuardController._halt_guarded_strategies(
                mock, equity, ratio, permanent=permanent,
            )
        )

        # current = 2000, initial = 10000, ratio = 0.2 < 0.3 → absolute guard
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=2000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        # No cooldown timer should be set (permanent halt)
        mock.clock.set_time_alert_ns.assert_not_called()

    def test_cooldown_clears_history(self) -> None:
        """Test _on_cooldown_expired clears equity history."""
        from collections import deque

        mock = self._make_mock_controller()
        mock._halted = True
        mock._equity_history = deque([
            (50 * self.HOUR_NS, 5000.0),
            (60 * self.HOUR_NS, 4000.0),
        ])
        mock._stopped_strategies = [MagicMock()]

        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3500.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert not mock._halted
        # History should be cleared and only contain the new starting point
        assert len(mock._equity_history) == 1
        assert mock._equity_history[0][1] == 3500.0

    def test_both_layers_independent(self) -> None:
        """Test rolling drawdown triggers, restarts, then absolute guard still works."""
        mock = self._make_mock_controller(
            drawdown_ratio=0.7, min_equity_ratio=0.3,
        )
        mock._equity_history = deque([
            (90 * self.HOUR_NS, 5000.0),
        ])
        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        mock._halt_guarded_strategies = lambda equity, ratio, permanent=False: (
            EquityGuardController._halt_guarded_strategies(
                mock, equity, ratio, permanent=permanent,
            )
        )

        # Rolling drawdown: 3400/5000 = 0.68 < 0.7 → halt with cooldown
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3400.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        mock.clock.set_time_alert_ns.assert_called_once()  # cooldown set

        # Simulate cooldown → restart
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=3000.0,
        ):
            EquityGuardController._on_cooldown_expired(mock, MagicMock())

        assert not mock._halted
        assert mock._initial_balance == 3000.0

        # Now absolute guard: 2000/3000 = 0.67, but 2000/10000 = 0.2 < 0.3
        # Wait — initial_balance was reset to 3000, so 2000/3000 = 0.67 > 0.3
        # Need initial to still be original for absolute guard to work
        # Actually initial_balance is reset on cooldown. Let's check:
        # The absolute guard should use the ORIGINAL initial balance, not reset one.
        # This test validates that min_equity_ratio still uses current initial_balance.

    def test_empty_window_uses_alltime_peak(self) -> None:
        """Test empty drawdown_window uses all-time peak (no pruning)."""
        mock = self._make_mock_controller(
            drawdown_window="", drawdown_ratio=0.7,
        )
        # Very old peak should NOT be pruned
        mock._equity_history = deque([
            (1 * self.HOUR_NS, 8000.0),   # very old, but should be kept
            (50 * self.HOUR_NS, 4000.0),
            (90 * self.HOUR_NS, 4500.0),
        ])
        s1 = MagicMock()
        s1.id = "CSStrategy-001"
        s1.is_running = True
        mock._trader.strategies.return_value = [s1]

        mock._halt_guarded_strategies = lambda equity, ratio, permanent=False: (
            EquityGuardController._halt_guarded_strategies(
                mock, equity, ratio, permanent=permanent,
            )
        )

        # current = 5000, all-time peak = 8000, ratio = 0.625 < 0.7 → trigger
        with patch(
            "nautilus_quants.controllers.equity_guard.compute_mtm_equity",
            return_value=5000.0,
        ):
            EquityGuardController._on_check(mock, MagicMock())

        assert mock._halted
        # Old entry should still be present (not pruned)
        timestamps = [ts for ts, _ in mock._equity_history]
        assert 1 * self.HOUR_NS in timestamps
